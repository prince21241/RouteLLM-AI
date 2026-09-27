"""Shared models for providers, the registry, and API responses."""

from decimal import Decimal
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class QualityTier(StrEnum):
    """Relative quality band stored with a model."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ModelConfig(BaseModel):
    """Static metadata for one model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
    )

    provider: Provider
    model_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    quality_tier: QualityTier
    input_cost_per_million_tokens: Decimal = Field(ge=0)
    output_cost_per_million_tokens: Decimal = Field(ge=0)
    context_window: int = Field(gt=0)
    enabled: bool
    local: bool


class LLMResponse(BaseModel):
    """Completion returned by a provider.

    ``estimated_cost`` stays empty until shared pricing fills it in.
    """

    model_config = ConfigDict(extra="forbid")

    provider: Provider
    model: str = Field(min_length=1)
    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    estimated_cost: Annotated[Decimal, Field(ge=0)] | None = None
