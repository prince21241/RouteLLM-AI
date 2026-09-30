"""Short transactions for the request lifecycle.

Each method opens a session, commits, and closes it before returning.
Callers must not hold a session across a provider network call.
"""

import uuid

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.dashboard.filters import DashboardFilter
from app.dashboard.raw import BreakdownRaw, DashboardListItem, JudgeRaw, OverviewRaw
from app.db.analytics import fetch_breakdown, fetch_overview, fetch_requests
from app.db.models import AttemptRow, EvaluationRow, RequestRow
from app.db.records import (
    AttemptOutcome,
    EvaluationWrite,
    PendingRequest,
    StoredAttempt,
    StoredEvaluation,
    StoredRequest,
    StoredSummary,
)
from app.pricing.service import CostCompleteness


class RequestStore:
    """PostgreSQL store for requests and attempts."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def ping(self) -> None:
        """Check that the database accepts a query."""
        async with self._sessions() as session:
            await session.execute(text("SELECT 1"))

    async def create_pending(self, pending: PendingRequest) -> None:
        """Insert a pending request and commit before any provider call."""
        async with self._sessions() as session:
            async with session.begin():
                session.add(
                    RequestRow(
                        id=pending.request_id,
                        prompt=pending.prompt,
                        system_prompt=pending.system_prompt,
                        created_at=pending.created_at,
                        updated_at=pending.created_at,
                        completed_at=None,
                        status="pending",
                        complexity_score=pending.complexity_score,
                        complexity_tier=pending.complexity_tier,
                        complexity_reasons=list(pending.complexity_reasons),
                        provider=pending.provider,
                        model_id=pending.model_id,
                        selected_model_tier=pending.selected_model_tier,
                        selection_reason=pending.selection_reason,
                        degraded=pending.degraded,
                        escalated=False,
                        response_text=None,
                        total_cost=None,
                        cost_completeness=CostCompleteness.UNKNOWN.value,
                        end_to_end_latency_ms=None,
                        premium_baseline_model=None,
                        premium_baseline_cost=None,
                        estimated_savings=None,
                        routing_metadata=pending.routing_metadata,
                    )
                )

    async def record_success(self, outcome: AttemptOutcome) -> None:
        """Store a successful attempt and the final request fields."""
        await self._finish(outcome)

    async def record_failure(self, outcome: AttemptOutcome) -> None:
        """Store a failed attempt. Unknown costs stay null."""
        await self._finish(outcome)

    async def save_attempt(self, outcome: AttemptOutcome) -> None:
        """Insert one attempt without closing the request."""
        async with self._sessions() as session:
            async with session.begin():
                request = await session.get(RequestRow, outcome.request_id)
                if request is None:
                    raise LookupError("request row is missing")
                session.add(_attempt(outcome))

    async def finalize_request(self, outcome: AttemptOutcome) -> None:
        """Write the final request fields and any evaluation rows."""
        async with self._sessions() as session:
            async with session.begin():
                request = await session.get(RequestRow, outcome.request_id)
                if request is None:
                    raise LookupError("request row is missing")
                _apply_final(request, outcome)
                for evaluation in outcome.evaluations:
                    session.add(_evaluation(outcome.request_id, evaluation))

    async def list_requests(self, *, limit: int, offset: int) -> tuple[list[StoredSummary], int]:
        """Return newest requests first. ``id`` breaks timestamp ties."""
        async with self._sessions() as session:
            total = await session.scalar(select(func.count()).select_from(RequestRow))
            rows = (
                await session.scalars(
                    select(RequestRow)
                    .order_by(RequestRow.created_at.desc(), RequestRow.id.desc())
                    .limit(limit)
                    .offset(offset)
                )
            ).all()
            summaries = [_summary(row) for row in rows]
        return summaries, int(total or 0)

    async def dashboard_overview(self, filt: DashboardFilter) -> OverviewRaw:
        """Aggregate stored chat requests in the database."""
        async with self._sessions() as session:
            return await fetch_overview(session, filt)

    async def dashboard_breakdown(
        self,
        filt: DashboardFilter,
    ) -> tuple[tuple[BreakdownRaw, ...], tuple[JudgeRaw, ...]]:
        """Aggregate attempts and judge calls for the filtered requests."""
        async with self._sessions() as session:
            return await fetch_breakdown(session, filt)

    async def dashboard_requests(
        self,
        filt: DashboardFilter,
    ) -> tuple[tuple[DashboardListItem, ...], int]:
        """Return one filtered history page and the filtered request total."""
        async with self._sessions() as session:
            return await fetch_requests(session, filt)

    async def get_request(self, request_id: uuid.UUID) -> StoredRequest | None:
        """Return one request and its attempts, or ``None`` when it is missing."""
        async with self._sessions() as session:
            row = await session.scalar(
                select(RequestRow)
                .where(RequestRow.id == request_id)
                .options(
                    selectinload(RequestRow.attempts),
                    selectinload(RequestRow.evaluations),
                )
            )
            if row is None:
                return None
            return _detail(row)

    async def _finish(self, outcome: AttemptOutcome) -> None:
        async with self._sessions() as session:
            async with session.begin():
                request = await session.get(RequestRow, outcome.request_id)
                if request is None:
                    raise LookupError("request row is missing")
                _apply_final(request, outcome)
                session.add(_attempt(outcome))
                for evaluation in outcome.evaluations:
                    session.add(_evaluation(outcome.request_id, evaluation))


def _apply_final(request: RequestRow, outcome: AttemptOutcome) -> None:
    request.status = outcome.status
    request.updated_at = outcome.completed_at
    request.completed_at = outcome.completed_at
    request.response_text = outcome.response_text
    request.total_cost = outcome.total_cost
    request.cost_completeness = outcome.cost_completeness
    request.end_to_end_latency_ms = outcome.end_to_end_latency_ms
    request.premium_baseline_model = outcome.premium_baseline_model
    request.premium_baseline_cost = outcome.premium_baseline_cost
    request.estimated_savings = outcome.estimated_savings
    request.escalated = outcome.escalated
    request.quality_verdict = outcome.quality_verdict
    request.quality_score = outcome.quality_score
    request.quality_reasons = outcome.quality_reasons
    request.escalation_reason = outcome.escalation_reason
    request.escalation_error = outcome.escalation_error
    request.final_model_id = outcome.final_model_id
    request.returned_attempt_number = outcome.returned_attempt_number
    request.fallback_used = outcome.fallback_used
    request.fallback_reason = outcome.fallback_reason
    request.fallback_skips = [
        {"model_id": model_id, "reason": reason} for model_id, reason in outcome.fallback_skips
    ]
    request.final_provider = outcome.final_provider


def _evaluation(request_id: uuid.UUID, evaluation: EvaluationWrite) -> EvaluationRow:
    return EvaluationRow(
        id=uuid.uuid4(),
        request_id=request_id,
        attempt_number=evaluation.attempt_number,
        source=evaluation.source,
        method=evaluation.method,
        verdict=evaluation.verdict,
        score=evaluation.score,
        reasons=list(evaluation.reasons),
        judge_model=evaluation.judge_model,
        judge_input_tokens=evaluation.judge_input_tokens,
        judge_output_tokens=evaluation.judge_output_tokens,
        judge_cost=evaluation.judge_cost,
        judge_cost_completeness=evaluation.judge_cost_completeness,
        judge_latency_ms=evaluation.judge_latency_ms,
        error_message=evaluation.error_message,
        created_at=evaluation.created_at,
    )


def _attempt(outcome: AttemptOutcome) -> AttemptRow:
    return AttemptRow(
        id=uuid.uuid4(),
        request_id=outcome.request_id,
        attempt_number=outcome.attempt_number,
        provider=outcome.provider,
        configured_model_id=outcome.configured_model_id,
        reported_model_id=outcome.reported_model_id,
        status=outcome.status,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        cached_input_tokens=outcome.cached_input_tokens,
        cache_write_input_tokens=outcome.cache_write_input_tokens,
        cache_read_input_tokens=outcome.cache_read_input_tokens,
        cache_write_5m_tokens=outcome.cache_write_5m_tokens,
        cache_write_1h_tokens=outcome.cache_write_1h_tokens,
        reasoning_tokens=outcome.reasoning_tokens,
        provider_latency_ms=outcome.provider_latency_ms,
        estimated_cost=outcome.estimated_cost,
        cost_completeness=outcome.cost_completeness,
        error_message=outcome.error_message,
        pricing_snapshot=outcome.pricing_snapshot,
        purpose=outcome.purpose,
        error_category=outcome.error_category,
        created_at=outcome.completed_at,
    )


def _summary(row: RequestRow) -> StoredSummary:
    return StoredSummary(
        request_id=row.id,
        created_at=row.created_at,
        status=row.status,
        provider=row.provider,
        model_id=row.model_id,
        complexity_tier=row.complexity_tier,
        total_cost=row.total_cost,
        cost_completeness=row.cost_completeness,
        estimated_savings=row.estimated_savings,
        end_to_end_latency_ms=row.end_to_end_latency_ms,
    )


def _detail(row: RequestRow) -> StoredRequest:
    return StoredRequest(
        request_id=row.id,
        prompt=row.prompt,
        system_prompt=row.system_prompt,
        created_at=row.created_at,
        completed_at=row.completed_at,
        status=row.status,
        complexity_score=row.complexity_score,
        complexity_tier=row.complexity_tier,
        complexity_reasons=list(row.complexity_reasons),
        provider=row.provider,
        model_id=row.model_id,
        selected_model_tier=row.selected_model_tier,
        selection_reason=row.selection_reason,
        degraded=row.degraded,
        escalated=row.escalated,
        response_text=row.response_text,
        total_cost=row.total_cost,
        cost_completeness=row.cost_completeness,
        end_to_end_latency_ms=row.end_to_end_latency_ms,
        premium_baseline_model=row.premium_baseline_model,
        premium_baseline_cost=row.premium_baseline_cost,
        estimated_savings=row.estimated_savings,
        attempts=[_stored_attempt(attempt) for attempt in row.attempts],
        quality_verdict=row.quality_verdict,
        quality_score=row.quality_score,
        quality_reasons=None if row.quality_reasons is None else list(row.quality_reasons),
        escalation_reason=row.escalation_reason,
        escalation_error=row.escalation_error,
        final_model_id=row.final_model_id,
        returned_attempt_number=row.returned_attempt_number,
        evaluations=[_stored_evaluation(item) for item in row.evaluations],
        fallback_used=row.fallback_used,
        fallback_reason=row.fallback_reason,
        fallback_skips=_skips(row.fallback_skips),
        final_provider=row.final_provider,
        routing_metadata=None if row.routing_metadata is None else dict(row.routing_metadata),
    )


def _stored_attempt(row: AttemptRow) -> StoredAttempt:
    snapshot = row.pricing_snapshot
    return StoredAttempt(
        attempt_number=row.attempt_number,
        provider=row.provider,
        configured_model_id=row.configured_model_id,
        reported_model_id=row.reported_model_id,
        status=row.status,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cached_input_tokens=row.cached_input_tokens,
        cache_write_input_tokens=row.cache_write_input_tokens,
        cache_read_input_tokens=row.cache_read_input_tokens,
        cache_write_5m_tokens=row.cache_write_5m_tokens,
        cache_write_1h_tokens=row.cache_write_1h_tokens,
        reasoning_tokens=row.reasoning_tokens,
        provider_latency_ms=row.provider_latency_ms,
        estimated_cost=row.estimated_cost,
        cost_completeness=row.cost_completeness,
        error_message=row.error_message,
        pricing_snapshot=dict(snapshot) if isinstance(snapshot, dict) else None,
        created_at=row.created_at,
        purpose=row.purpose,
        error_category=row.error_category,
    )


def _skips(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        return []
    skips: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model_id")
        reason = item.get("reason")
        if isinstance(model_id, str) and isinstance(reason, str):
            skips.append((model_id, reason))
    return skips


def _stored_evaluation(row: EvaluationRow) -> StoredEvaluation:
    return StoredEvaluation(
        attempt_number=row.attempt_number,
        source=row.source,
        method=row.method,
        verdict=row.verdict,
        score=row.score,
        reasons=list(row.reasons),
        judge_model=row.judge_model,
        judge_input_tokens=row.judge_input_tokens,
        judge_output_tokens=row.judge_output_tokens,
        judge_cost=row.judge_cost,
        judge_cost_completeness=row.judge_cost_completeness,
        judge_latency_ms=row.judge_latency_ms,
        error_message=row.error_message,
        created_at=row.created_at,
    )
