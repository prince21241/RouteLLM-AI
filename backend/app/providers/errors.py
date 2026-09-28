"""Typed provider failures.

Messages name the provider and the failure category. They do not include
credentials, request headers, request bodies, or raw upstream payloads.
"""


class ProviderError(Exception):
    """Base error for a provider call."""


class ProviderConfigurationError(ProviderError):
    """Required provider configuration is missing or invalid."""


class ProviderAuthenticationError(ProviderError):
    """The provider rejected the credentials or the caller's permissions."""


class ProviderRateLimitError(ProviderError):
    """The provider rate-limited the request."""


class ProviderTimeoutError(ProviderError):
    """The request timed out or the connection failed."""


class ProviderUpstreamError(ProviderError):
    """The provider returned a server error."""


class ProviderResponseError(ProviderError):
    """The provider response was malformed, empty, or incomplete.

    ``usage`` is set when the failed payload still reported token counts.
    It stays ``None`` when those counts were absent.
    """

    def __init__(self, message: str, *, usage: object | None = None) -> None:
        super().__init__(message)
        self.usage = usage
