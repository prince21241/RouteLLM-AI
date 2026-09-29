"""Classify provider failures for fallback decisions.

The category and message are safe to store. Classification does not read
credentials, headers, or raw provider bodies.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.api.schemas import ReportedUsage
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderSafetyError,
    ProviderTimeoutError,
    ProviderUpstreamError,
)


class ErrorCategory(StrEnum):
    """Normalized provider failure category."""

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    RATE_LIMIT = "rate_limit"
    TRANSIENT_SERVER = "transient_server"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    SAFETY_REFUSAL = "safety_refusal"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


_FALLBACK_ELIGIBLE = frozenset(
    {
        ErrorCategory.TIMEOUT,
        ErrorCategory.CONNECTION,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TRANSIENT_SERVER,
    }
)


@dataclass(frozen=True)
class ProviderFailure:
    """One classified provider failure."""

    category: ErrorCategory
    message: str
    fallback_eligible: bool
    usage: ReportedUsage | None = None


def classify_provider_error(exc: ProviderError) -> ProviderFailure:
    """Map a provider exception to a category. Unknown usage stays unknown."""
    message = str(exc)[:500]
    usage = exc.usage if isinstance(exc, ProviderResponseError) else None
    if not isinstance(usage, ReportedUsage):
        usage = None
    category = _category(exc, message)
    return ProviderFailure(
        category=category,
        message=message,
        fallback_eligible=category in _FALLBACK_ELIGIBLE,
        usage=usage,
    )


def _category(exc: ProviderError, message: str) -> ErrorCategory:
    if isinstance(exc, ProviderSafetyError):
        return ErrorCategory.SAFETY_REFUSAL
    if isinstance(exc, ProviderTimeoutError):
        if "connection failed" in message:
            return ErrorCategory.CONNECTION
        return ErrorCategory.TIMEOUT
    if isinstance(exc, ProviderRateLimitError):
        return ErrorCategory.RATE_LIMIT
    if isinstance(exc, ProviderUpstreamError):
        return ErrorCategory.TRANSIENT_SERVER
    if isinstance(exc, ProviderAuthenticationError):
        if "HTTP 403" in message:
            return ErrorCategory.PERMISSION
        return ErrorCategory.AUTHENTICATION
    if isinstance(exc, ProviderConfigurationError):
        return ErrorCategory.CONFIGURATION
    if isinstance(exc, ProviderResponseError):
        lowered = message.lower()
        if "content_filter" in lowered or "safety" in lowered or "refusal" in lowered:
            return ErrorCategory.SAFETY_REFUSAL
        return ErrorCategory.INVALID_REQUEST
    return ErrorCategory.UNKNOWN
