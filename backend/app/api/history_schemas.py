"""History responses. List items omit prompts and completions."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RequestSummaryResponse(BaseModel):
    """One row in the history list."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: datetime
    status: str
    provider: str
    model: str
    complexity_tier: str
    total_cost: str | None
    cost_completeness: str
    estimated_savings: str | None
    end_to_end_latency_ms: float | None


class RequestListResponse(BaseModel):
    """A newest-first page of request summaries."""

    model_config = ConfigDict(extra="forbid")

    items: list[RequestSummaryResponse]
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    total: int = Field(ge=0)


class AttemptResponse(BaseModel):
    """One stored provider attempt, including the pricing snapshot."""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(ge=1)
    provider: str
    configured_model_id: str
    reported_model_id: str | None
    status: str
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    cache_read_input_tokens: int | None
    cache_write_5m_tokens: int | None
    cache_write_1h_tokens: int | None
    reasoning_tokens: int | None
    provider_latency_ms: float | None
    estimated_cost: str | None
    cost_completeness: str
    error_message: str | None
    pricing_snapshot: dict[str, object] | None
    created_at: datetime
    purpose: str | None = None
    error_category: str | None = None


class RequestDetailResponse(BaseModel):
    """A stored request and its attempts."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    prompt: str
    system_prompt: str | None
    created_at: datetime
    completed_at: datetime | None
    status: str
    complexity_score: float
    complexity_tier: str
    complexity_reasons: list[str]
    provider: str
    model: str
    selected_model_tier: str
    selection_reason: str
    degraded: bool
    escalated: bool
    response: str | None
    total_cost: str | None
    cost_completeness: str
    end_to_end_latency_ms: float | None
    baseline_model: str | None
    baseline_cost: str | None
    estimated_savings: str | None
    savings_basis: str
    attempts: list[AttemptResponse]
    quality_verdict: str | None = None
    quality_score: float | None = None
    quality_reasons: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    escalation_error: str | None = None
    final_model: str | None = None
    returned_attempt: int | None = None
    evaluations: list["EvaluationResponse"] = Field(default_factory=list)
    fallback_used: bool | None = None
    fallback_reason: str | None = None
    fallback_skips: list[dict[str, str]] = Field(default_factory=list)
    final_provider: str | None = None


class EvaluationResponse(BaseModel):
    """One stored quality check, including a judge call when one ran."""

    model_config = ConfigDict(extra="forbid")

    attempt_number: int = Field(ge=1)
    source: str
    method: str
    verdict: str
    score: float | None
    reasons: list[str]
    judge_model: str | None
    judge_input_tokens: int | None
    judge_output_tokens: int | None
    judge_cost: str | None
    judge_cost_completeness: str | None
    judge_latency_ms: float | None
    error_message: str | None
    created_at: datetime


RequestDetailResponse.model_rebuild()
