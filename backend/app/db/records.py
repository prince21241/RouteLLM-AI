"""Values passed between the chat service and the request store."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID


@dataclass(frozen=True)
class PendingRequest:
    """A request row written before the provider call."""

    request_id: UUID
    prompt: str
    system_prompt: str | None
    created_at: datetime
    complexity_score: Decimal
    complexity_tier: str
    complexity_reasons: list[str]
    provider: str
    model_id: str
    selected_model_tier: str
    selection_reason: str
    degraded: bool


@dataclass(frozen=True)
class AttemptOutcome:
    """The single Phase 4 attempt, written after the provider call returns."""

    request_id: UUID
    status: str
    completed_at: datetime
    provider: str
    configured_model_id: str
    reported_model_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None
    cache_write_input_tokens: int | None
    cache_read_input_tokens: int | None
    cache_write_5m_tokens: int | None
    cache_write_1h_tokens: int | None
    reasoning_tokens: int | None
    provider_latency_ms: Decimal | None
    estimated_cost: Decimal | None
    cost_completeness: str
    error_message: str | None
    pricing_snapshot: dict[str, object] | None
    response_text: str | None
    total_cost: Decimal | None
    end_to_end_latency_ms: Decimal
    premium_baseline_model: str | None
    premium_baseline_cost: Decimal | None
    estimated_savings: Decimal | None


@dataclass(frozen=True)
class StoredAttempt:
    """One attempt read back from the database."""

    attempt_number: int
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
    provider_latency_ms: Decimal | None
    estimated_cost: Decimal | None
    cost_completeness: str
    error_message: str | None
    pricing_snapshot: dict[str, object] | None
    created_at: datetime


@dataclass(frozen=True)
class StoredSummary:
    """History list item. Prompts and responses are omitted."""

    request_id: UUID
    created_at: datetime
    status: str
    provider: str
    model_id: str
    complexity_tier: str
    total_cost: Decimal | None
    cost_completeness: str
    estimated_savings: Decimal | None
    end_to_end_latency_ms: Decimal | None


@dataclass(frozen=True)
class StoredRequest:
    """Full request plus its attempts."""

    request_id: UUID
    prompt: str
    system_prompt: str | None
    created_at: datetime
    completed_at: datetime | None
    status: str
    complexity_score: Decimal
    complexity_tier: str
    complexity_reasons: list[str]
    provider: str
    model_id: str
    selected_model_tier: str
    selection_reason: str
    degraded: bool
    escalated: bool
    response_text: str | None
    total_cost: Decimal | None
    cost_completeness: str
    end_to_end_latency_ms: Decimal | None
    premium_baseline_model: str | None
    premium_baseline_cost: Decimal | None
    estimated_savings: Decimal | None
    attempts: list[StoredAttempt]
