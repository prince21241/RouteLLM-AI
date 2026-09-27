"""Anthropic Messages API client.

Requests are non-streaming. A system prompt is sent with the top-level
``system`` field. Refusal text on a completed message is normal content.
``max_tokens`` stops and other unfinished stop reasons are incomplete
responses, not transport failures.
"""

from typing import Self

import httpx
from pydantic import SecretStr, ValidationError

from app.api.schemas import LLMResponse, Provider
from app.config import Settings
from app.providers.base import LLMProvider
from app.providers.client import (
    ManagedAsyncClient,
    ProviderSession,
    reported_model,
    require_configured_text,
    require_positive_int,
    require_token_count,
    safe_token,
)
from app.providers.errors import ProviderConfigurationError, ProviderResponseError

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 30.0
ANTHROPIC_VERSION = "2023-06-01"
_MESSAGES_PATH = "/v1/messages"
_COMPLETE_STOPS = frozenset({"end_turn", "stop_sequence", "refusal"})
_INCOMPLETE_STOPS = frozenset(
    {"max_tokens", "pause_turn", "model_context_window_exceeded"}
)


class AnthropicProvider(ProviderSession, LLMProvider):
    """Non-streaming Anthropic text generation."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.anthropic.com",
    ) -> None:
        key = require_configured_text(api_key, "ANTHROPIC_API_KEY")
        self._model = require_configured_text(model, "ANTHROPIC_MODEL")
        self._max_tokens = require_positive_int(max_tokens, "ANTHROPIC_MAX_TOKENS")
        root = require_configured_text(base_url, "Anthropic base URL").rstrip("/")
        self._url = f"{root}{_MESSAGES_PATH}"
        self._api_key = SecretStr(key)
        self._http = ManagedAsyncClient(
            client=client,
            timeout_seconds=timeout_seconds,
            timeout_label="ANTHROPIC_TIMEOUT_SECONDS",
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> Self:
        """Build a client from process settings.

        Raises when the Anthropic key is unset. Serving ``/health`` does not
        call this.
        """
        if settings.anthropic_api_key is None:
            raise ProviderConfigurationError("ANTHROPIC_API_KEY is not configured")
        return cls(
            settings.anthropic_api_key.get_secret_value(),
            model=settings.anthropic_model,
            max_tokens=settings.anthropic_max_tokens,
            timeout_seconds=settings.anthropic_timeout_seconds,
            client=client,
        )

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate text with the Messages API. There are no retries."""
        payload, latency_ms = await self._http.post_json(
            self._url,
            headers={
                "x-api-key": self._api_key.get_secret_value(),
                "anthropic-version": ANTHROPIC_VERSION,
            },
            payload=_request_body(
                prompt,
                system_prompt,
                model=self._model,
                max_tokens=self._max_tokens,
            ),
            provider_name="Anthropic",
        )
        return _normalize(payload, fallback_model=self._model, latency_ms=latency_ms)


def _request_body(
    prompt: str,
    system_prompt: str | None,
    *,
    model: str,
    max_tokens: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if system_prompt is not None:
        body["system"] = system_prompt
    return body


def _normalize(
    payload: dict[str, object],
    *,
    fallback_model: str,
    latency_ms: float,
) -> LLMResponse:
    _require_complete_stop(payload)
    content = _text_blocks(payload.get("content"))
    try:
        return LLMResponse(
            provider=Provider.ANTHROPIC,
            model=reported_model(payload, fallback_model),
            content=content,
            input_tokens=require_token_count(
                payload.get("usage"),
                "input_tokens",
                "Anthropic",
            ),
            output_tokens=require_token_count(
                payload.get("usage"),
                "output_tokens",
                "Anthropic",
            ),
            latency_ms=latency_ms,
            estimated_cost=None,
        )
    except ValidationError:
        raise ProviderResponseError("Anthropic response could not be normalized") from None


def _require_complete_stop(payload: dict[str, object]) -> None:
    stop_reason = payload.get("stop_reason")
    if safe_token(stop_reason) in _COMPLETE_STOPS:
        return
    label = safe_token(stop_reason)
    if label in _INCOMPLETE_STOPS:
        raise ProviderResponseError(f"Anthropic response was incomplete ({label})")
    raise ProviderResponseError(
        f"Anthropic response was not completed ({label or 'unexpected'})"
    )


def _text_blocks(content: object) -> str:
    if not isinstance(content, list):
        raise ProviderResponseError("Anthropic response did not include text")
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    if not parts or "".join(parts).strip() == "":
        raise ProviderResponseError("Anthropic response did not include text")
    return "".join(parts)
