"""Shared async HTTP client for provider calls."""

import json
import math
import re
import time
from collections.abc import Mapping
from typing import Self

import httpx

from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUpstreamError,
)

_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def monotonic_now() -> float:
    """Return a monotonic timestamp in seconds."""
    return time.perf_counter()


def require_configured_text(value: str, label: str) -> str:
    """Return stripped text or raise when configuration is blank."""
    if not isinstance(value, str) or value.strip() == "":
        raise ProviderConfigurationError(f"{label} is not configured")
    return value.strip()


def require_positive_int(value: int, label: str) -> int:
    """Return a positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ProviderConfigurationError(f"{label} must be a positive integer")
    return value


def require_positive_timeout(value: float, label: str) -> float:
    """Return a finite timeout greater than zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderConfigurationError(f"{label} must be greater than zero")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ProviderConfigurationError(f"{label} must be greater than zero")
    return timeout


def require_token_count(usage: object, field: str, provider_name: str) -> int:
    """Read one API-reported token count.

    A missing count is an error. It is not replaced with zero.
    """
    if not isinstance(usage, dict) or field not in usage or usage[field] is None:
        raise ProviderResponseError(f"{provider_name} response did not include {field}")
    value = usage[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(f"{provider_name} response included invalid {field}")
    return value


def optional_token_count(usage: object, field: str, provider_name: str) -> int | None:
    """Read an optional token count.

    A missing field stays ``None``. A present but invalid value is an error.
    """
    if not isinstance(usage, dict) or field not in usage or usage[field] is None:
        return None
    value = usage[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderResponseError(f"{provider_name} response included invalid {field}")
    return value


def safe_token(value: object) -> str | None:
    """Return a short status or error code, never a free-form upstream message."""
    if isinstance(value, str) and _SAFE_TOKEN.fullmatch(value):
        return value
    return None


def reported_model(payload: Mapping[str, object], fallback: str) -> str:
    """Use the model id reported by the provider when it is present."""
    model = payload.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return fallback


class ProviderSession:
    """Closes the HTTP client owned by a provider."""

    _http: "ManagedAsyncClient"

    async def aclose(self) -> None:
        """Close the client created by this provider."""
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


class ManagedAsyncClient:
    """Reusable httpx client. Injected clients stay open for the caller."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None,
        timeout_seconds: float,
        timeout_label: str,
    ) -> None:
        timeout = require_positive_timeout(timeout_seconds, timeout_label)
        self._timeout = httpx.Timeout(timeout)
        self._owns_client = client is None
        self._client = (
            client
            if client is not None
            else httpx.AsyncClient(timeout=self._timeout, follow_redirects=False)
        )

    async def aclose(self) -> None:
        """Close the owned client. A second call is a no-op."""
        if self._owns_client and not self._client.is_closed:
            await self._client.aclose()

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, object],
        provider_name: str,
    ) -> tuple[dict[str, object], float]:
        """POST JSON once and return the object body with elapsed milliseconds."""
        started = monotonic_now()
        try:
            response = await self._client.post(
                url,
                headers=dict(headers),
                json=dict(payload),
                timeout=self._timeout,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            raise ProviderTimeoutError(f"{provider_name} request timed out") from None
        except httpx.NetworkError:
            raise ProviderTimeoutError(f"{provider_name} connection failed") from None
        except httpx.TransportError:
            raise ProviderTimeoutError(f"{provider_name} connection failed") from None
        except RuntimeError as exc:
            if "client has been closed" in str(exc):
                raise ProviderError(f"{provider_name} client is closed") from None
            raise
        latency_ms = (monotonic_now() - started) * 1000
        _raise_for_status(response, provider_name)
        return _json_object(response, provider_name), latency_ms


def _raise_for_status(response: httpx.Response, provider_name: str) -> None:
    code = response.status_code
    if 200 <= code < 300:
        return
    if code in (401, 403):
        raise ProviderAuthenticationError(
            f"{provider_name} authentication failed (HTTP {code})"
        )
    if code == 429:
        raise ProviderRateLimitError(
            f"{provider_name} rate limit exceeded (HTTP {code})"
        )
    if code == 408:
        raise ProviderTimeoutError(f"{provider_name} request timed out (HTTP {code})")
    if code >= 500:
        raise ProviderUpstreamError(f"{provider_name} upstream error (HTTP {code})")
    raise ProviderResponseError(
        f"{provider_name} returned an unexpected HTTP status (HTTP {code})"
    )


def _json_object(response: httpx.Response, provider_name: str) -> dict[str, object]:
    try:
        body = response.json()
    except json.JSONDecodeError:
        raise ProviderResponseError(
            f"{provider_name} response was not valid JSON"
        ) from None
    if not isinstance(body, dict):
        raise ProviderResponseError(f"{provider_name} response was not a JSON object")
    return body
