"""OpenAI Responses API client.

Text is collected from message ``output_text`` items. Reasoning items can
appear ahead of the message, so the first output item is not assumed to be
the answer. ``reasoning.effort`` stays ``minimal`` for the verified
``gpt-5-nano`` configuration. ``max_output_tokens`` covers visible output
and reasoning tokens.
"""

from typing import Self

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
    require_positive_int,
    require_token_count,
    safe_token,
)
from app.providers.errors import ProviderConfigurationError, ProviderResponseError

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MAX_OUTPUT_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 30.0
_RESPONSES_PATH = "/v1/responses"
_INCOMPLETE_REASONS = frozenset({"max_output_tokens", "content_filter"})


class OpenAIProvider(ProviderSession, LLMProvider):
    """Non-streaming OpenAI text generation."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.openai.com",
    ) -> None:
        key = require_configured_text(api_key, "OPENAI_API_KEY")
        self._model = require_configured_text(model, "OPENAI_MODEL")
        self._max_output_tokens = require_positive_int(
            max_output_tokens,
            "OPENAI_MAX_OUTPUT_TOKENS",
        )
        root = require_configured_text(base_url, "OpenAI base URL").rstrip("/")
        self._url = f"{root}{_RESPONSES_PATH}"
        self._api_key = SecretStr(key)
        self._http = ManagedAsyncClient(
            client=client,
            timeout_seconds=timeout_seconds,
            timeout_label="OPENAI_TIMEOUT_SECONDS",
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> Self:
        """Build a client from process settings.

        Raises when the OpenAI key is unset. Importing or serving ``/health``
        does not call this.
        """
        if settings.openai_api_key is None:
            raise ProviderConfigurationError("OPENAI_API_KEY is not configured")
        return cls(
            settings.openai_api_key.get_secret_value(),
            model=settings.openai_model,
            max_output_tokens=settings.openai_max_output_tokens,
            timeout_seconds=settings.openai_timeout_seconds,
            client=client,
        )

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self._model!r})"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate text with the Responses API. There are no retries."""
        payload, latency_ms = await self._http.post_json(
            self._url,
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
            payload=_request_body(
                prompt,
                system_prompt,
                model=self._model,
                max_output_tokens=self._max_output_tokens,
            ),
            provider_name="OpenAI",
        )
        return _normalize(payload, fallback_model=self._model, latency_ms=latency_ms)


def _request_body(
    prompt: str,
    system_prompt: str | None,
    *,
    model: str,
    max_output_tokens: int,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "minimal"},
    }
    if system_prompt is not None:
        body["instructions"] = system_prompt
    return body


def _normalize(
    payload: dict[str, object],
    *,
    fallback_model: str,
    latency_ms: float,
) -> LLMResponse:
    _require_completed(payload)
    content = _message_text(payload)
    usage = _reported_usage(payload)
    try:
        return LLMResponse(
            provider=Provider.OPENAI,
            model=reported_model(payload, fallback_model),
            content=content,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            latency_ms=latency_ms,
            estimated_cost=None,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_input_tokens=usage.cache_write_input_tokens,
            reasoning_tokens=usage.reasoning_tokens,
        )
    except ValidationError:
        raise ProviderResponseError("OpenAI response could not be normalized") from None


def _require_completed(payload: dict[str, object]) -> None:
    status = payload.get("status")
    if status == "completed":
        return
    if status == "incomplete":
        reason = _incomplete_reason(payload)
        detail = f" ({reason})" if reason is not None else ""
        raise ProviderResponseError(
            f"OpenAI response was incomplete{detail}",
            usage=_usage_or_none(payload),
        )
    if status == "failed":
        code = _failure_code(payload)
        detail = f" ({code})" if code is not None else ""
        raise ProviderResponseError(
            f"OpenAI response failed{detail}",
            usage=_usage_or_none(payload),
        )
    label = safe_token(status) or "unexpected"
    raise ProviderResponseError(f"OpenAI response was not completed ({label})")


def _reported_usage(payload: dict[str, object]) -> ReportedUsage:
    """Read OpenAI usage. Cached tokens are a subset of ``input_tokens``."""
    usage = payload.get("usage")
    input_tokens = require_token_count(usage, "input_tokens", "OpenAI")
    output_tokens = require_token_count(usage, "output_tokens", "OpenAI")
    details = usage.get("input_tokens_details") if isinstance(usage, dict) else None
    output_details = usage.get("output_tokens_details") if isinstance(usage, dict) else None
    return ReportedUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=optional_token_count(details, "cached_tokens", "OpenAI"),
        cache_write_input_tokens=optional_token_count(details, "cache_write_tokens", "OpenAI"),
        reasoning_tokens=optional_token_count(output_details, "reasoning_tokens", "OpenAI"),
    )


def _usage_or_none(payload: dict[str, object]) -> ReportedUsage | None:
    try:
        return _reported_usage(payload)
    except ProviderResponseError:
        return None


def _incomplete_reason(payload: dict[str, object]) -> str | None:
    details = payload.get("incomplete_details")
    if not isinstance(details, dict):
        return None
    reason = safe_token(details.get("reason"))
    if reason in _INCOMPLETE_REASONS:
        return reason
    return None


def _failure_code(payload: dict[str, object]) -> str | None:
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    return safe_token(error.get("code"))


def _message_text(payload: dict[str, object]) -> str:
    output = payload.get("output")
    if not isinstance(output, list):
        raise ProviderResponseError("OpenAI response did not include text")
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            text = _block_text(block)
            if text is not None:
                parts.append(text)
    if not parts or "".join(parts).strip() == "":
        raise ProviderResponseError("OpenAI response did not include text")
    return "".join(parts)


def _block_text(block: object) -> str | None:
    if not isinstance(block, dict):
        return None
    if block.get("type") == "output_text" and isinstance(block.get("text"), str):
        return block["text"]
    if block.get("type") == "refusal" and isinstance(block.get("refusal"), str):
        return block["refusal"]
    return None
