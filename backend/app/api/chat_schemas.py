"""Request and response models for POST /api/v1/chat."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas import Provider, QualityTier


class ChatRequest(BaseModel):
    """One chat turn. Blank prompts are rejected by the chat service."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    system_prompt: str | None = None


class RoutingDetails(BaseModel):
    """Pre-request routing. Complexity and the selected model stay separate."""

    model_config = ConfigDict(extra="forbid")

    complexity_score: float = Field(ge=0, le=1)
    complexity_tier: QualityTier
    complexity_reasons: list[str]
    provider: Provider
    model: str = Field(min_length=1)
    selected_model_tier: QualityTier
    selection_reason: str = Field(min_length=1)
    degraded: bool
    escalated: bool = False


class ChatMetrics(BaseModel):
    """Usage and cost for the request.

    ``latency_ms`` is the provider latency of the answer that was returned.
    ``cost`` is the total of every generation attempt and any judge call.
    With quality checks disabled there is one attempt and no judge, so
    ``cost`` matches that attempt. ``quality_score`` stays empty unless a
    quality check ran.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cost: str | None = None
    cost_completeness: Literal["unknown", "estimated", "complete"] = "unknown"
    quality_score: float | None = Field(default=None, ge=0, le=1)


class ChatResponse(BaseModel):
    """Normalized completion plus the routing decision that produced it.

    ``baseline_cost`` and ``estimated_savings`` use the same decimal-string
    format as ``metrics.cost``. ``savings_basis`` is always
    ``same_token_volume``: the premium price is applied to the returned
    answer's observed token counts. It is not the measured difference
    between a router run and a baseline run. ``end_to_end_latency_ms``
    includes routing, quality checks, escalation, and persistence.
    ``metrics.latency_ms`` stays the returned answer's provider latency.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    response: str
    routing: RoutingDetails
    metrics: ChatMetrics
    baseline_model: str | None = None
    baseline_cost: str | None = None
    estimated_savings: str | None = None
    savings_basis: Literal["same_token_volume"] = "same_token_volume"
    persistence_status: Literal["not_configured", "stored", "failed"]
    persistence_warning: str | None = None
    end_to_end_latency_ms: float = Field(ge=0)
    quality_verdict: Literal["pass", "fail", "unknown", "error"] | None = None
    quality_reasons: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    escalation_error: str | None = None
    returned_model: str | None = None
    returned_attempt: int = Field(default=1, ge=1)
