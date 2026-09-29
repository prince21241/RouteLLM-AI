"""Database-side dashboard aggregates.

The existing ``ix_requests_created_at_id`` index supports the time range and
the newest-first page. Equality filters on provider, model, and status are
not indexed separately: this is a local history table, and those predicates
do not justify another index yet.
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Numeric, and_, func, not_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.dashboard.filters import DashboardFilter
from app.dashboard.raw import (
    BreakdownRaw,
    CostSums,
    DashboardListItem,
    DaySums,
    JudgeRaw,
    LatencySums,
    OverviewRaw,
)
from app.db.models import AttemptRow, EvaluationRow, RequestRow

_PREVIEW = 80
_MONEY = Numeric(14, 3)


def request_clauses(filt: DashboardFilter) -> list[object]:
    """SQL predicates for the initial routed request, not each attempt."""
    clauses: list[object] = []
    if filt.start is not None:
        clauses.append(RequestRow.created_at >= filt.start)
    if filt.end is not None:
        clauses.append(RequestRow.created_at < filt.end)
    if filt.model is not None:
        clauses.append(RequestRow.model_id == filt.model)
    if filt.provider is not None:
        clauses.append(RequestRow.provider == filt.provider)
    if filt.status is not None:
        clauses.append(RequestRow.status == filt.status)
    return clauses


def request_totals_statement(filt: DashboardFilter):
    """Aggregate logical requests. Attempt rows are not counted here."""
    complete, estimated, unknown = _cost_parts(RequestRow.cost_completeness, RequestRow.total_cost)
    quality_unknown = or_(
        RequestRow.quality_verdict.is_(None),
        RequestRow.quality_verdict == "unknown",
    )
    return select(
        func.count().label("requests"),
        _count(RequestRow.status == "succeeded").label("succeeded_requests"),
        _count(RequestRow.status == "failed").label("failed_requests"),
        _count(RequestRow.status == "pending").label("pending_requests"),
        _count(RequestRow.status.notin_(("succeeded", "failed", "pending"))).label("other_status_requests"),
        _count(RequestRow.escalated.is_(True)).label("escalated_requests"),
        _count(RequestRow.fallback_used.is_(True)).label("fallback_used_requests"),
        _count(RequestRow.fallback_used.is_not(None)).label("fallback_known_requests"),
        _count(RequestRow.fallback_used.is_(None)).label("fallback_unknown_requests"),
        _count(RequestRow.quality_verdict == "pass").label("quality_pass"),
        _count(RequestRow.quality_verdict == "fail").label("quality_fail"),
        _count(quality_unknown).label("quality_unknown"),
        _count(RequestRow.quality_verdict == "error").label("quality_error"),
        _count(
            and_(
                RequestRow.quality_verdict.is_not(None),
                RequestRow.quality_verdict.notin_(("pass", "fail", "unknown", "error")),
            )
        ).label("quality_other"),
        _latency_p50(RequestRow.end_to_end_latency_ms).label("request_latency_p50"),
        _latency_avg(RequestRow.end_to_end_latency_ms).label("request_latency_avg"),
        _count(RequestRow.end_to_end_latency_ms.is_not(None)).label("request_latency_known"),
        _count(RequestRow.end_to_end_latency_ms.is_(None)).label("request_latency_missing"),
        func.sum(RequestRow.total_cost).filter(complete).label("cost_complete"),
        func.sum(RequestRow.total_cost).filter(estimated).label("cost_estimated"),
        _count(unknown).label("cost_unknown"),
        func.sum(RequestRow.estimated_savings)
        .filter(RequestRow.estimated_savings.is_not(None))
        .label("savings_estimate"),
        _count(RequestRow.estimated_savings.is_not(None)).label("savings_known"),
        _count(RequestRow.estimated_savings.is_(None)).label("savings_unknown"),
    ).where(*request_clauses(filt))


def attempt_totals_statement(filt: DashboardFilter):
    """Aggregate generation attempts that belong to the filtered requests."""
    latency = AttemptRow.provider_latency_ms
    return (
        select(
            func.count().label("attempt_count"),
            _count(AttemptRow.purpose == "routing").label("routing_attempts"),
            _count(AttemptRow.purpose == "fallback").label("fallback_attempts"),
            _count(AttemptRow.purpose == "escalation").label("escalation_attempts"),
            _count(AttemptRow.purpose.is_(None)).label("purpose_unknown_attempts"),
            _latency_p50(latency).label("attempt_latency_p50"),
            _latency_avg(latency).label("attempt_latency_avg"),
            _count(latency.is_not(None)).label("attempt_latency_known"),
            _count(latency.is_(None)).label("attempt_latency_missing"),
        )
        .select_from(AttemptRow)
        .where(AttemptRow.request_id.in_(_filtered_ids(filt)))
    )


def day_statement(filt: DashboardFilter):
    """Sum recorded request costs by UTC day. Empty days are not invented here."""
    day = func.date_trunc("day", func.timezone("UTC", RequestRow.created_at))
    complete, estimated, unknown = _cost_parts(RequestRow.cost_completeness, RequestRow.total_cost)
    return (
        select(
            day.label("day"),
            func.count().label("requests"),
            func.sum(RequestRow.total_cost).filter(complete).label("cost_complete"),
            func.sum(RequestRow.total_cost).filter(estimated).label("cost_estimated"),
            _count(unknown).label("cost_unknown"),
        )
        .where(*request_clauses(filt))
        .group_by(day)
        .order_by(day)
    )


def model_breakdown_statement(filt: DashboardFilter):
    """Attribute attempt cost to the provider and model that ran the attempt."""
    complete, estimated, unknown = _cost_parts(
        AttemptRow.cost_completeness,
        AttemptRow.estimated_cost,
    )
    latency = AttemptRow.provider_latency_ms
    return (
        select(
            AttemptRow.provider.label("provider"),
            AttemptRow.configured_model_id.label("model"),
            func.count().label("attempt_count"),
            func.count(func.distinct(AttemptRow.request_id)).label("request_count"),
            _count(AttemptRow.status == "failed").label("failed_attempts"),
            func.sum(AttemptRow.estimated_cost).filter(complete).label("cost_complete"),
            func.sum(AttemptRow.estimated_cost).filter(estimated).label("cost_estimated"),
            _count(unknown).label("cost_unknown"),
            _latency_p50(latency).label("latency_p50"),
            _count(latency.is_not(None)).label("latency_known"),
            _count(latency.is_(None)).label("latency_missing"),
        )
        .where(AttemptRow.request_id.in_(_filtered_ids(filt)))
        .group_by(AttemptRow.provider, AttemptRow.configured_model_id)
        .order_by(AttemptRow.provider, AttemptRow.configured_model_id)
    )


def judge_breakdown_statement(filt: DashboardFilter):
    """Sum recorded judge costs. These rows are not generation attempts."""
    complete, estimated, unknown = _cost_parts(
        EvaluationRow.judge_cost_completeness,
        EvaluationRow.judge_cost,
    )
    return (
        select(
            EvaluationRow.judge_model.label("judge_model"),
            func.count().label("evaluations"),
            func.sum(EvaluationRow.judge_cost).filter(complete).label("cost_complete"),
            func.sum(EvaluationRow.judge_cost).filter(estimated).label("cost_estimated"),
            _count(unknown).label("cost_unknown"),
        )
        .where(EvaluationRow.request_id.in_(_filtered_ids(filt)))
        .group_by(EvaluationRow.judge_model)
        .order_by(EvaluationRow.judge_model.asc().nulls_last())
    )


def request_page_statement(filt: DashboardFilter):
    """Return one history page. The selected prompt text is capped at 80 characters."""
    attempt_count = (
        select(func.count())
        .select_from(AttemptRow)
        .where(AttemptRow.request_id == RequestRow.id)
        .correlate(RequestRow)
        .scalar_subquery()
    )
    return (
        select(
            RequestRow.id.label("request_id"),
            RequestRow.created_at,
            RequestRow.status,
            RequestRow.provider,
            RequestRow.model_id,
            RequestRow.final_provider,
            RequestRow.final_model_id,
            func.substr(RequestRow.prompt, 1, _PREVIEW).label("prompt_preview"),
            (func.char_length(RequestRow.prompt) > _PREVIEW).label("prompt_truncated"),
            RequestRow.total_cost,
            RequestRow.cost_completeness,
            RequestRow.estimated_savings,
            RequestRow.end_to_end_latency_ms,
            RequestRow.quality_verdict,
            RequestRow.escalated,
            RequestRow.fallback_used,
            attempt_count.label("attempt_count"),
        )
        .where(*request_clauses(filt))
        .order_by(RequestRow.created_at.desc(), RequestRow.id.desc())
        .limit(filt.limit)
        .offset(filt.offset)
    )


def request_total_statement(filt: DashboardFilter):
    """Count filtered requests for pagination."""
    return select(func.count()).select_from(RequestRow).where(*request_clauses(filt))


async def fetch_overview(session: AsyncSession, filt: DashboardFilter) -> OverviewRaw:
    """Run the overview aggregates."""
    totals = (await session.execute(request_totals_statement(filt))).one()
    attempts = (await session.execute(attempt_totals_statement(filt))).one()
    days = (await session.execute(day_statement(filt))).all()
    judges = (await session.execute(_judge_totals(filt))).one()
    return OverviewRaw(
        requests=_int(totals.requests),
        succeeded_requests=_int(totals.succeeded_requests),
        failed_requests=_int(totals.failed_requests),
        pending_requests=_int(totals.pending_requests),
        other_status_requests=_int(totals.other_status_requests),
        escalated_requests=_int(totals.escalated_requests),
        fallback_used_requests=_int(totals.fallback_used_requests),
        fallback_known_requests=_int(totals.fallback_known_requests),
        fallback_unknown_requests=_int(totals.fallback_unknown_requests),
        quality_pass=_int(totals.quality_pass),
        quality_fail=_int(totals.quality_fail),
        quality_unknown=_int(totals.quality_unknown),
        quality_error=_int(totals.quality_error),
        quality_other=_int(totals.quality_other),
        request_latency=LatencySums(
            p50_ms=_decimal(totals.request_latency_p50),
            average_ms=_decimal(totals.request_latency_avg),
            known_count=_int(totals.request_latency_known),
            missing_count=_int(totals.request_latency_missing),
        ),
        attempt_count=_int(attempts.attempt_count),
        routing_attempts=_int(attempts.routing_attempts),
        fallback_attempts=_int(attempts.fallback_attempts),
        escalation_attempts=_int(attempts.escalation_attempts),
        purpose_unknown_attempts=_int(attempts.purpose_unknown_attempts),
        attempt_latency=LatencySums(
            p50_ms=_decimal(attempts.attempt_latency_p50),
            average_ms=_decimal(attempts.attempt_latency_avg),
            known_count=_int(attempts.attempt_latency_known),
            missing_count=_int(attempts.attempt_latency_missing),
        ),
        request_cost=CostSums(
            complete=_decimal(totals.cost_complete),
            estimated=_decimal(totals.cost_estimated),
            unknown_count=_int(totals.cost_unknown),
        ),
        judge_cost=CostSums(
            complete=_decimal(judges.cost_complete),
            estimated=_decimal(judges.cost_estimated),
            unknown_count=_int(judges.cost_unknown),
        ),
        savings_estimate=_decimal(totals.savings_estimate),
        savings_known_requests=_int(totals.savings_known),
        savings_unknown_requests=_int(totals.savings_unknown),
        days=tuple(_day(row) for row in days),
    )


async def fetch_breakdown(
    session: AsyncSession,
    filt: DashboardFilter,
) -> tuple[tuple[BreakdownRaw, ...], tuple[JudgeRaw, ...]]:
    """Run the model and judge breakdowns."""
    models = (await session.execute(model_breakdown_statement(filt))).all()
    judges = (await session.execute(judge_breakdown_statement(filt))).all()
    return tuple(_breakdown(row) for row in models), tuple(_judge(row) for row in judges)


async def fetch_requests(
    session: AsyncSession,
    filt: DashboardFilter,
) -> tuple[tuple[DashboardListItem, ...], int]:
    """Run the history page and its filtered total."""
    total = await session.scalar(request_total_statement(filt))
    rows = (await session.execute(request_page_statement(filt))).all()
    return tuple(_list_item(row) for row in rows), _int(total)


def _filtered_ids(filt: DashboardFilter):
    statement = select(RequestRow.id)
    clauses = request_clauses(filt)
    if clauses:
        statement = statement.where(*clauses)
    return statement


def _judge_totals(filt: DashboardFilter):
    complete, estimated, unknown = _cost_parts(
        EvaluationRow.judge_cost_completeness,
        EvaluationRow.judge_cost,
    )
    return select(
        func.sum(EvaluationRow.judge_cost).filter(complete).label("cost_complete"),
        func.sum(EvaluationRow.judge_cost).filter(estimated).label("cost_estimated"),
        _count(unknown).label("cost_unknown"),
    ).where(EvaluationRow.request_id.in_(_filtered_ids(filt)))


def _cost_parts(completeness, amount):
    complete = and_(completeness == "complete", amount.is_not(None))
    estimated = and_(completeness == "estimated", amount.is_not(None))
    unknown = not_(or_(complete, estimated))
    return complete, estimated, unknown


def _count(clause):
    return func.count().filter(clause)


def _latency_p50(column):
    return func.cast(func.percentile_cont(0.5).within_group(column), _MONEY)


def _latency_avg(column):
    return func.cast(func.avg(column), _MONEY)


def _day(row) -> DaySums:
    return DaySums(
        day=_as_date(row.day),
        requests=_int(row.requests),
        cost_complete=_decimal(row.cost_complete),
        cost_estimated=_decimal(row.cost_estimated),
        cost_unknown_requests=_int(row.cost_unknown),
    )


def _breakdown(row) -> BreakdownRaw:
    return BreakdownRaw(
        provider=row.provider,
        model=row.model,
        attempt_count=_int(row.attempt_count),
        request_count=_int(row.request_count),
        failed_attempts=_int(row.failed_attempts),
        cost_complete=_decimal(row.cost_complete),
        cost_estimated=_decimal(row.cost_estimated),
        cost_unknown_attempts=_int(row.cost_unknown),
        latency_p50_ms=_decimal(row.latency_p50),
        latency_known_attempts=_int(row.latency_known),
        latency_missing_attempts=_int(row.latency_missing),
    )


def _judge(row) -> JudgeRaw:
    return JudgeRaw(
        judge_model=row.judge_model,
        evaluations=_int(row.evaluations),
        cost_complete=_decimal(row.cost_complete),
        cost_estimated=_decimal(row.cost_estimated),
        cost_unknown_evaluations=_int(row.cost_unknown),
    )


def _list_item(row) -> DashboardListItem:
    return DashboardListItem(
        request_id=row.request_id,
        created_at=row.created_at,
        status=row.status,
        provider=row.provider,
        model_id=row.model_id,
        final_provider=row.final_provider,
        final_model_id=row.final_model_id,
        prompt_preview=row.prompt_preview,
        prompt_truncated=bool(row.prompt_truncated),
        total_cost=row.total_cost,
        cost_completeness=row.cost_completeness,
        estimated_savings=row.estimated_savings,
        end_to_end_latency_ms=row.end_to_end_latency_ms,
        quality_verdict=row.quality_verdict,
        escalated=bool(row.escalated),
        fallback_used=row.fallback_used,
        attempt_count=_int(row.attempt_count),
    )


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError("day bucket was not a date")


def _int(value: object) -> int:
    if value is None:
        return 0
    return int(value)


def _decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))
