"""GET /api/v1/requests and GET /api/v1/requests/{request_id}."""

import logging
from decimal import Decimal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query

from app.api.history_schemas import (
    AttemptResponse,
    EvaluationResponse,
    RequestDetailResponse,
    RequestListResponse,
    RequestSummaryResponse,
)
from app.db.records import StoredAttempt, StoredEvaluation, StoredRequest, StoredSummary
from app.db.store import RequestStore
from app.pricing.money import money_to_api

logger = logging.getLogger("app.history")


def register_history_routes(app: FastAPI) -> None:
    """Add the history routes. They read ``app.state.store`` per request."""

    @app.get("/api/v1/requests", response_model=RequestListResponse)
    async def list_requests(
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> RequestListResponse:
        store = _store(app)
        try:
            items, total = await store.list_requests(limit=limit, offset=offset)
        except Exception as exc:
            _raise_unavailable(exc)
        return RequestListResponse(
            items=[_summary(item) for item in items],
            limit=limit,
            offset=offset,
            total=total,
        )

    @app.get("/api/v1/requests/{request_id}", response_model=RequestDetailResponse)
    async def get_request(request_id: str) -> RequestDetailResponse:
        parsed = _parse_id(request_id)
        store = _store(app)
        try:
            row = await store.get_request(parsed)
        except Exception as exc:
            _raise_unavailable(exc)
        if row is None:
            raise HTTPException(status_code=404, detail="Request not found")
        return _detail(row)


def _store(app: FastAPI) -> RequestStore:
    store = getattr(app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Request history is unavailable")
    return store


def _parse_id(request_id: str) -> UUID:
    try:
        return UUID(request_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Request not found") from None


def _raise_unavailable(exc: Exception) -> None:
    logger.error("History query failed (%s)", type(exc).__name__)
    raise HTTPException(status_code=503, detail="Request history is unavailable") from None


def _summary(item: StoredSummary) -> RequestSummaryResponse:
    return RequestSummaryResponse(
        request_id=str(item.request_id),
        created_at=item.created_at,
        status=item.status,
        provider=item.provider,
        model=item.model_id,
        complexity_tier=item.complexity_tier,
        total_cost=money_to_api(item.total_cost),
        cost_completeness=item.cost_completeness,
        estimated_savings=money_to_api(item.estimated_savings),
        end_to_end_latency_ms=_latency(item.end_to_end_latency_ms),
    )


def _detail(row: StoredRequest) -> RequestDetailResponse:
    return RequestDetailResponse(
        request_id=str(row.request_id),
        prompt=row.prompt,
        system_prompt=row.system_prompt,
        created_at=row.created_at,
        completed_at=row.completed_at,
        status=row.status,
        complexity_score=float(row.complexity_score),
        complexity_tier=row.complexity_tier,
        complexity_reasons=row.complexity_reasons,
        provider=row.provider,
        model=row.model_id,
        selected_model_tier=row.selected_model_tier,
        selection_reason=row.selection_reason,
        degraded=row.degraded,
        escalated=row.escalated,
        response=row.response_text,
        total_cost=money_to_api(row.total_cost),
        cost_completeness=row.cost_completeness,
        end_to_end_latency_ms=_latency(row.end_to_end_latency_ms),
        baseline_model=row.premium_baseline_model,
        baseline_cost=money_to_api(row.premium_baseline_cost),
        estimated_savings=money_to_api(row.estimated_savings),
        savings_basis="same_token_volume",
        attempts=[_attempt(attempt) for attempt in row.attempts],
        quality_verdict=row.quality_verdict,
        quality_score=None if row.quality_score is None else float(row.quality_score),
        quality_reasons=list(row.quality_reasons or []),
        escalation_reason=row.escalation_reason,
        escalation_error=row.escalation_error,
        final_model=row.final_model_id,
        returned_attempt=row.returned_attempt_number,
        evaluations=[_evaluation(item) for item in (row.evaluations or [])],
    )


def _attempt(row: StoredAttempt) -> AttemptResponse:
    return AttemptResponse(
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
        provider_latency_ms=_latency(row.provider_latency_ms),
        estimated_cost=money_to_api(row.estimated_cost),
        cost_completeness=row.cost_completeness,
        error_message=row.error_message,
        pricing_snapshot=row.pricing_snapshot,
        created_at=row.created_at,
    )


def _evaluation(row: StoredEvaluation) -> EvaluationResponse:
    return EvaluationResponse(
        attempt_number=row.attempt_number,
        source=row.source,
        method=row.method,
        verdict=row.verdict,
        score=None if row.score is None else float(row.score),
        reasons=list(row.reasons),
        judge_model=row.judge_model,
        judge_input_tokens=row.judge_input_tokens,
        judge_output_tokens=row.judge_output_tokens,
        judge_cost=money_to_api(row.judge_cost),
        judge_cost_completeness=row.judge_cost_completeness,
        judge_latency_ms=_latency(row.judge_latency_ms),
        error_message=row.error_message,
        created_at=row.created_at,
    )


def _latency(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value)
