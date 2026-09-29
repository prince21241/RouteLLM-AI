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
    attempt_number: int = 1
    escalated: bool = False
    quality_verdict: str | None = None
    quality_score: Decimal | None = None
    quality_reasons: list[str] | None = None
    escalation_reason: str | None = None
    escalation_error: str | None = None
    final_model_id: str | None = None
    returned_attempt_number: int = 1
    evaluations: tuple["EvaluationWrite", ...] = ()


@dataclass(frozen=True)
class EvaluationWrite:
    """A quality result stored with the request that produced the answer."""

    attempt_number: int
    source: str
    method: str
    verdict: str
    score: Decimal | None
    reasons: list[str]
    judge_model: str | None
    judge_input_tokens: int | None
    judge_output_tokens: int | None
    judge_cost: Decimal | None
    judge_cost_completeness: str | None
    judge_latency_ms: Decimal | None
    error_message: str | None
    created_at: datetime


@dataclass(frozen=True)
class StoredEvaluation:
    """One quality result read back from the database."""

    attempt_number: int
    source: str
    method: str
    verdict: str
    score: Decimal | None
    reasons: list[str]
    judge_model: str | None
    judge_input_tokens: int | None
    judge_output_tokens: int | None
    judge_cost: Decimal | None
    judge_cost_completeness: str | None
    judge_latency_ms: Decimal | None
    error_message: str | None
    created_at: datetime


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
    quality_verdict: str | None = None
    quality_score: Decimal | None = None
    quality_reasons: list[str] | None = None
    escalation_reason: str | None = None
    escalation_error: str | None = None
    final_model_id: str | None = None
    returned_attempt_number: int | None = None
    evaluations: list[StoredEvaluation] | None = None
