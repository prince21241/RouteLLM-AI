"""Validate dashboard query parameters.

``from`` is inclusive and ``to`` is exclusive. Both are UTC instants.
A date without a time is UTC midnight. A date range may not exceed 366 days.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_PROVIDERS = ("openai", "anthropic", "ollama")
_STATUSES = ("pending", "succeeded", "failed")
_MAX_RANGE = timedelta(days=366)


class DashboardQueryError(ValueError):
    """The dashboard query parameters cannot be applied."""


@dataclass(frozen=True)
class DashboardFilter:
    """One validated set of dashboard filters."""

    start: datetime | None = None
    end: datetime | None = None
    model: str | None = None
    provider: str | None = None
    status: str | None = None
    limit: int = 20
    offset: int = 0


def parse_dashboard_filter(
    *,
    start: str | None,
    end: str | None,
    model: str | None,
    provider: str | None,
    status: str | None,
    limit: int = 20,
    offset: int = 0,
) -> DashboardFilter:
    """Parse query text into a filter. Blank values mean no constraint."""
    parsed_start = _instant(start, name="from")
    parsed_end = _instant(end, name="to")
    if parsed_start is not None and parsed_end is not None:
        if parsed_end <= parsed_start:
            raise DashboardQueryError("to must be later than from")
        if parsed_end - parsed_start > _MAX_RANGE:
            raise DashboardQueryError("Date range must be 366 days or less")
    if limit < 1 or limit > 100:
        raise DashboardQueryError("limit must be from 1 through 100")
    if offset < 0:
        raise DashboardQueryError("offset must be zero or greater")
    return DashboardFilter(
        start=parsed_start,
        end=parsed_end,
        model=_model(model),
        provider=_choice(provider, allowed=_PROVIDERS, name="provider"),
        status=_choice(status, allowed=_STATUSES, name="status"),
        limit=limit,
        offset=offset,
    )


def _instant(value: str | None, *, name: str) -> datetime | None:
    text = _blank(value)
    if text is None:
        return None
    if len(text) == 10:
        text = f"{text}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise DashboardQueryError(f"{name} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise DashboardQueryError(f"{name} must include a time zone")
    return parsed.astimezone(timezone.utc)


def _model(value: str | None) -> str | None:
    text = _blank(value)
    if text is None:
        return None
    if len(text) > 128 or any(ord(character) < 32 for character in text):
        raise DashboardQueryError("model must be a single model id")
    return text


def _choice(value: str | None, *, allowed: tuple[str, ...], name: str) -> str | None:
    text = _blank(value)
    if text is None:
        return None
    if text not in allowed:
        choices = ", ".join(allowed)
        raise DashboardQueryError(f"{name} must be {choices}")
    return text


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None
