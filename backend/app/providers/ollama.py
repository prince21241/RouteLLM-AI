"""Ollama chat client.

Calls ``POST /api/chat`` with streaming disabled. A local server gets no
credential. When an API key is configured, it is sent as a bearer token.
Input tokens come from
``prompt_eval_count`` and output tokens from ``eval_count``.
"""

from typing import Self
from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr, ValidationError

from app.api.schemas import LLMResponse, Provider, ReportedUsage
from app.config import Settings
from app.providers.base import LLMProvider
from app.providers.client import (
    ManagedAsyncClient,
    ProviderSession,
    optional_token_count,
    reported_model,
    require_configured_text,
    require_token_count,
    safe_token,
)
from app.providers.errors import ProviderConfigurationError, ProviderResponseError

DEFAULT_MODEL = "llama3.2"
DEFAULT_TIMEOUT_SECONDS = 60.0
_CHAT_PATH = "/api/chat"


class OllamaProvider(ProviderSession, LLMProvider):
    """Non-streaming Ollama text generation."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        root = _http_base_url(base_url)
        self._model = require_configured_text(model, "OLLAMA_MODEL")
        self._url = f"{root}{_CHAT_PATH}"
        self._api_key = _optional_api_key(api_key)
        self._http = ManagedAsyncClient(
            client=client,
            timeout_seconds=timeout_seconds,
            timeout_label="OLLAMA_TIMEOUT_SECONDS",
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> Self:
        """Build a client from process settings.

        Raises when the base URL is blank. Serving ``/health`` does not
        connect to Ollama.
        """
        return cls(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
            api_key=None
            if settings.ollama_api_key is None
            else settings.ollama_api_key.get_secret_value(),
            client=client,
        )

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self._model!r})"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate text with streaming disabled. There are no retries."""
        payload, latency_ms = await self._http.post_json(
            self._url,
            headers=_authorization(self._api_key),
            payload=_request_body(prompt, system_prompt, model=self._model),
            provider_name="Ollama",
        )
        return _normalize(payload, fallback_model=self._model, latency_ms=latency_ms)


def _optional_api_key(api_key: str | None) -> SecretStr | None:
    if api_key is None or api_key.strip() == "":
        return None
    return SecretStr(api_key.strip())


def _authorization(api_key: SecretStr | None) -> dict[str, str]:
    if api_key is None:
        return {}
    return {"Authorization": f"Bearer {api_key.get_secret_value()}"}


def _http_base_url(base_url: str) -> str:
    candidate = base_url.strip() if isinstance(base_url, str) else ""
    if candidate == "":
        raise ProviderConfigurationError("OLLAMA_BASE_URL is not configured")
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
        raise ProviderConfigurationError("OLLAMA_BASE_URL is not a valid HTTP URL")
    return candidate.rstrip("/")


def _request_body(
    prompt: str,
    system_prompt: str | None,
    *,
    model: str,
) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return {"model": model, "messages": messages, "stream": False}


def _normalize(
    payload: dict[str, object],
    *,
    fallback_model: str,
    latency_ms: float,
) -> LLMResponse:
    content = _message_content(payload)
    usage = _reported_usage(payload)
    try:
        return LLMResponse(
            provider=Provider.OLLAMA,
            model=reported_model(payload, fallback_model),
            content=content,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            estimated_cost=None,
            cached_input_tokens=usage.cached_input_tokens,
        )
    except ValidationError:
        raise ProviderResponseError("Ollama response could not be normalized") from None


def _reported_usage(payload: dict[str, object]) -> ReportedUsage:
    """Read Ollama counts. Cached prompt tokens are not added to the input total."""
    return ReportedUsage(
        input_tokens=require_token_count(payload, "prompt_eval_count", "Ollama"),
        output_tokens=require_token_count(payload, "eval_count", "Ollama"),
        cached_input_tokens=optional_token_count(payload, "prompt_eval_cached_count", "Ollama"),
    )


def _message_content(payload: dict[str, object]) -> str:
    if payload.get("done") is not True:
        raise ProviderResponseError(
            "Ollama response was incomplete",
            usage=_usage_or_none(payload),
        )
    reason = payload.get("done_reason")
    if reason is not None and reason != "stop":
        label = safe_token(reason) or "unexpected"
        raise ProviderResponseError(
            f"Ollama response was incomplete ({label})",
            usage=_usage_or_none(payload),
        )
    message = payload.get("message")
    if not isinstance(message, dict):
        raise ProviderResponseError("Ollama response did not include text")
    content = message.get("content")
    if not isinstance(content, str) or content.strip() == "":
        raise ProviderResponseError("Ollama response did not include text")
    return content


def _usage_or_none(payload: dict[str, object]) -> ReportedUsage | None:
    try:
        return _reported_usage(payload)
    except ProviderResponseError:
        return None
