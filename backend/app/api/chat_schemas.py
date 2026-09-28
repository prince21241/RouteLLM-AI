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
    escalated: Literal[False] = False


class ChatMetrics(BaseModel):
    """Usage from the single provider call.

    ``latency_ms`` is the provider-call latency returned by the client.
    It is not the full HTTP round trip. ``cost`` is a decimal string when
    the verified price book can price the model, and ``null`` when the cost
    is unknown. ``quality_score`` stays empty.
    """

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cost: str | None = None
    cost_completeness: Literal["unknown", "estimated", "complete"] = "unknown"
    quality_score: Literal[None] = None


class ChatResponse(BaseModel):
    """Normalized completion plus the routing decision that produced it.

    ``baseline_cost`` and ``estimated_savings`` use the same decimal-string
    format as ``metrics.cost``. ``savings_basis`` is always
    ``same_token_volume``: the premium price is applied to this response's
    observed token counts, not to a second model call.
    ``end_to_end_latency_ms`` includes routing and persistence.
    ``metrics.latency_ms`` stays the provider-call latency.
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
