"""Request and response models for POST /api/v1/chat."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.api.schemas import Provider, QualityTier


class ServerFeature(BaseModel):
    """A behavior controlled by server settings, not by the chat form."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool
    per_request_override: Literal[False] = False


class ChatModelChoice(BaseModel):
    """One catalog model. Enabled means the server can route to it."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    provider: Provider
    quality_tier: QualityTier
    enabled: bool
    local: bool


class ChatOptionsResponse(BaseModel):
    """Routing choices the chat page may offer. Credentials are omitted."""

    model_config = ConfigDict(extra="forbid")

    requests_are_independent: Literal[True] = True
    max_input_characters: int = Field(ge=1)
    context_note: str = (
        "Each send is a new request. Earlier messages are not included, "
        "and the model is not given this conversation. The character limit "
        "covers the user prompt and system prompt together. It is not a "
        "model context window."
    )
    model_selection_supported: Literal[False] = False
    provider_restriction_supported: Literal[True] = True
    models: list[ChatModelChoice]
    quality_evaluation: ServerFeature
    escalation: ServerFeature
    fallback: ServerFeature
    routing_strategy: Literal["rule_based", "ml"] = "rule_based"
    effective_routing_strategy: Literal["rule_based", "ml"] = "rule_based"
    ml_diagnostic: str | None = None


class ChatRequest(BaseModel):
    """One chat turn. Blank prompts are rejected by the chat service.

    ``provider`` limits routing and fallback to that provider. Omit it to
    allow the router to choose any enabled provider.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    system_prompt: str | None = None
    provider: Provider | None = None


class FallbackSkip(BaseModel):
    """A configured fallback model that was not called."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    reason: str


class RoutingMetadata(BaseModel):
    """How the initial model was chosen. Confidence is not answer quality."""

    model_config = ConfigDict(extra="forbid")

    requested_strategy: Literal["rule_based", "ml"]
    effective_strategy: Literal["rule_based", "ml"]
    artifact_version: str | None = None
    predicted_model: str | None = None
    confidence: float | None = None
    rules_used: bool
    rules_reason: str | None = None
    initial_model_id: str = Field(min_length=1)


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
    routing_metadata: RoutingMetadata | None = None


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
    fallback_used: bool = False
    fallback_reason: str | None = None
    fallback_skips: list[FallbackSkip] = Field(default_factory=list)
    final_provider: str | None = None
