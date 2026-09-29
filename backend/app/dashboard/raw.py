"""Aggregate rows returned by the database before API formatting."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID


@dataclass(frozen=True)
class CostSums:
    """Sums split by cost completeness. Unknown amounts stay out of the sums."""

    complete: Decimal | None
    estimated: Decimal | None
    unknown_count: int


@dataclass(frozen=True)
class LatencySums:
    """Latency over recorded samples. Missing samples are counted, not averaged."""

    p50_ms: Decimal | None
    average_ms: Decimal | None
    known_count: int
    missing_count: int


@dataclass(frozen=True)
class DaySums:
    """One UTC day of stored chat requests."""

    day: date
    requests: int
    cost_complete: Decimal | None
    cost_estimated: Decimal | None
    cost_unknown_requests: int


@dataclass(frozen=True)
class OverviewRaw:
    """Database aggregates for the overview. Counts are not coerced from null sums."""

    requests: int
    succeeded_requests: int
    failed_requests: int
    pending_requests: int
    other_status_requests: int
    escalated_requests: int
    fallback_used_requests: int
    fallback_known_requests: int
    fallback_unknown_requests: int
    quality_pass: int
    quality_fail: int
    quality_unknown: int
    quality_error: int
    quality_other: int
    request_latency: LatencySums
    attempt_count: int
    routing_attempts: int
    fallback_attempts: int
    escalation_attempts: int
    purpose_unknown_attempts: int
    attempt_latency: LatencySums
    request_cost: CostSums
    judge_cost: CostSums
    savings_estimate: Decimal | None
    savings_known_requests: int
    savings_unknown_requests: int
    days: tuple[DaySums, ...]


@dataclass(frozen=True)
class BreakdownRaw:
    """Attempt totals for one provider and configured model."""

    provider: str
    model: str
    attempt_count: int
    request_count: int
    failed_attempts: int
    cost_complete: Decimal | None
    cost_estimated: Decimal | None
    cost_unknown_attempts: int
    latency_p50_ms: Decimal | None
    latency_known_attempts: int
    latency_missing_attempts: int


@dataclass(frozen=True)
class JudgeRaw:
    """Judge-call totals for one recorded judge model."""

    judge_model: str | None
    evaluations: int
    cost_complete: Decimal | None
    cost_estimated: Decimal | None
    cost_unknown_evaluations: int


@dataclass(frozen=True)
class DashboardListItem:
    """One history row. The preview is not the stored prompt."""

    request_id: UUID
    created_at: datetime
    status: str
    provider: str
    model_id: str
    final_provider: str | None
    final_model_id: str | None
    prompt_preview: str
    prompt_truncated: bool
    total_cost: Decimal | None
    cost_completeness: str
    estimated_savings: Decimal | None
    end_to_end_latency_ms: Decimal | None
    quality_verdict: str | None
    escalated: bool
    fallback_used: bool | None
    attempt_count: int
