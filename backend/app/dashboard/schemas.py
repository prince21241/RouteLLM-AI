"""Dashboard API models. Money stays a decimal string. Missing values stay null."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FilterView(BaseModel):
    """Filters that produced a dashboard response."""

    model_config = ConfigDict(extra="forbid")

    start: datetime | None
    end: datetime | None
    model: str | None
    provider: str | None
    status: str | None


class StatusCounts(BaseModel):
    """Stored request statuses. Pending is not an error."""

    model_config = ConfigDict(extra="forbid")

    succeeded: int = Field(ge=0)
    failed: int = Field(ge=0)
    pending: int = Field(ge=0)
    other: int = Field(ge=0)


class QualityCounts(BaseModel):
    """Request quality verdicts. A missing verdict is unknown."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    passed: int = Field(ge=0, alias="pass")
    failed: int = Field(ge=0, alias="fail")
    unknown: int = Field(ge=0)
    error: int = Field(ge=0)
    other: int = Field(ge=0)


class AttemptPurposeCounts(BaseModel):
    """Generation attempts by purpose. A missing purpose stays unknown."""

    model_config = ConfigDict(extra="forbid")

    routing: int = Field(ge=0)
    fallback: int = Field(ge=0)
    escalation: int = Field(ge=0)
    unknown: int = Field(ge=0)


class LatencyView(BaseModel):
    """Milliseconds over recorded samples only."""

    model_config = ConfigDict(extra="forbid")

    p50: float | None
    average: float | None
    known_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)


class CostView(BaseModel):
    """Complete and estimated sums. Unknown rows are counted and not added."""

    model_config = ConfigDict(extra="forbid")

    complete: str | None
    estimated: str | None
    unknown_count: int = Field(ge=0)


class DayView(BaseModel):
    """One UTC day. A request count of zero is an empty day, not a zero cost."""

    model_config = ConfigDict(extra="forbid")

    day: str
    requests: int = Field(ge=0)
    complete: str | None
    estimated: str | None
    unknown_count: int = Field(ge=0)


class OverviewResponse(BaseModel):
    """Overview metrics for stored chat requests."""

    model_config = ConfigDict(extra="forbid")

    timezone: Literal["UTC"]
    filters: FilterView
    request_count: int = Field(ge=0)
    attempt_count: int = Field(ge=0)
    status_counts: StatusCounts
    attempt_purposes: AttemptPurposeCounts
    failed_requests: int = Field(ge=0)
    error_rate: str | None
    escalated_requests: int = Field(ge=0)
    escalation_rate: str | None
    fallback_used_requests: int = Field(ge=0)
    fallback_known_requests: int = Field(ge=0)
    fallback_unknown_requests: int = Field(ge=0)
    fallback_rate: str | None
    quality: QualityCounts
    request_latency_ms: LatencyView
    attempt_latency_ms: LatencyView
    request_cost: CostView
    judge_cost: CostView
    savings_estimate: str | None
    savings_basis: Literal["same_token_volume"]
    savings_known_requests: int = Field(ge=0)
    savings_unknown_requests: int = Field(ge=0)
    by_day: list[DayView]
    notes: list[str]


class ModelBreakdown(BaseModel):
    """Attempt cost and latency for the provider and model that incurred them."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    attempt_count: int = Field(ge=0)
    request_count: int = Field(ge=0)
    failed_attempts: int = Field(ge=0)
    cost_complete: str | None
    cost_estimated: str | None
    cost_unknown_attempts: int = Field(ge=0)
    latency_p50_ms: float | None
    latency_known_attempts: int = Field(ge=0)
    latency_missing_attempts: int = Field(ge=0)


class JudgeBreakdown(BaseModel):
    """Recorded judge cost for one judge model. This is not a generation attempt."""

    model_config = ConfigDict(extra="forbid")

    judge_model: str | None
    evaluations: int = Field(ge=0)
    cost_complete: str | None
    cost_estimated: str | None
    cost_unknown_evaluations: int = Field(ge=0)


class BreakdownResponse(BaseModel):
    """Provider, model, and judge breakdowns for the same request filters."""

    model_config = ConfigDict(extra="forbid")

    timezone: Literal["UTC"]
    filters: FilterView
    models: list[ModelBreakdown]
    judges: list[JudgeBreakdown]
    notes: list[str]


class DashboardHistoryItem(BaseModel):
    """A history row with a truncated prompt preview."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: datetime
    status: str
    provider: str
    model: str
    final_provider: str | None
    final_model: str | None
    prompt_preview: str
    prompt_truncated: bool
    total_cost: str | None
    cost_completeness: str
    estimated_savings: str | None
    savings_basis: Literal["same_token_volume"]
    end_to_end_latency_ms: float | None
    quality_verdict: str | None
    escalated: bool
    fallback_used: bool | None
    attempt_count: int = Field(ge=0)


class DashboardHistoryResponse(BaseModel):
    """A filtered, newest-first page of stored chat requests."""

    model_config = ConfigDict(extra="forbid")

    timezone: Literal["UTC"]
    filters: FilterView
    items: list[DashboardHistoryItem]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)
