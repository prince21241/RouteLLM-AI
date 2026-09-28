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
