"""Turn database aggregates into API values.

Empty days inside an explicit range get a request count of zero. Their costs
stay null. A missing sum is not formatted as zero.
"""

from datetime import timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

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
from app.dashboard.schemas import (
    AttemptPurposeCounts,
    BreakdownResponse,
    CostView,
    DashboardHistoryItem,
    DashboardHistoryResponse,
    DayView,
    FilterView,
    JudgeBreakdown,
    LatencyView,
    ModelBreakdown,
    OverviewResponse,
    QualityCounts,
    StatusCounts,
)
from app.pricing.money import money_to_api

_RATE = Decimal("0.000001")
_MILLISECOND = Decimal("0.001")

OVERVIEW_NOTES = (
    "Counts are stored chat requests. Generation attempts are counted separately, so a fallback or escalation does not add a second request.",
    "Provider fallback and quality escalation are separate. The fallback rate ignores rows whose fallback flag was never recorded.",
    "Quality pass, fail, unknown, and error are separate. A missing verdict is unknown, not a pass or a failure.",
    "Request latency is end-to-end. Attempt latency is the provider call. Missing samples are left out of the average and median, not treated as zero.",
    "Complete, estimated, and unknown costs stay separate. Unknown amounts are not added as zero.",
    "Judge costs are the recorded judge component. They are not added again on top of the request total.",
    "Savings use the same-token-volume estimate. The premium model was not called.",
    "Times are UTC. from is inclusive and to is exclusive.",
    "Offline evaluation runs are not included.",
)

BREAKDOWN_NOTES = (
    "Attempt costs belong to the provider and model that ran the attempt.",
    "Filters select requests by the initial routed provider, model, status, and time. Attempts inside those requests can use another provider.",
    "A request with two attempts counts once in the overview and in each model row it touched.",
    "Judge rows are judge calls, not generation attempts. A missing judge model stays unrecorded.",
    "Unknown attempt costs are counted and are not added as zero.",
)


def overview_response(raw: OverviewRaw, filt: DashboardFilter) -> OverviewResponse:
    """Format overview aggregates and fill empty UTC days in a bounded range."""
    return OverviewResponse(
        timezone="UTC",
        filters=_filters(filt),
        request_count=raw.requests,
        attempt_count=raw.attempt_count,
        status_counts=StatusCounts(
            succeeded=raw.succeeded_requests,
            failed=raw.failed_requests,
            pending=raw.pending_requests,
            other=raw.other_status_requests,
        ),
        attempt_purposes=AttemptPurposeCounts(
            routing=raw.routing_attempts,
            fallback=raw.fallback_attempts,
            escalation=raw.escalation_attempts,
            unknown=raw.purpose_unknown_attempts,
        ),
        failed_requests=raw.failed_requests,
        error_rate=_rate(raw.failed_requests, raw.requests),
        escalated_requests=raw.escalated_requests,
        escalation_rate=_rate(raw.escalated_requests, raw.requests),
        fallback_used_requests=raw.fallback_used_requests,
        fallback_known_requests=raw.fallback_known_requests,
        fallback_unknown_requests=raw.fallback_unknown_requests,
        fallback_rate=_rate(raw.fallback_used_requests, raw.fallback_known_requests),
        quality=QualityCounts(
            passed=raw.quality_pass,
            failed=raw.quality_fail,
            unknown=raw.quality_unknown,
            error=raw.quality_error,
            other=raw.quality_other,
        ),
        request_latency_ms=_latency(raw.request_latency),
        attempt_latency_ms=_latency(raw.attempt_latency),
        request_cost=_cost(raw.request_cost),
        judge_cost=_cost(raw.judge_cost),
        savings_estimate=money_to_api(raw.savings_estimate),
        savings_basis="same_token_volume",
        savings_known_requests=raw.savings_known_requests,
        savings_unknown_requests=raw.savings_unknown_requests,
        by_day=[_day(item) for item in fill_days(raw.days, filt)],
        notes=list(OVERVIEW_NOTES),
    )


def breakdown_response(
    models: tuple[BreakdownRaw, ...],
    judges: tuple[JudgeRaw, ...],
    filt: DashboardFilter,
) -> BreakdownResponse:
    """Format provider and judge breakdowns."""
    return BreakdownResponse(
        timezone="UTC",
        filters=_filters(filt),
        models=[_model(item) for item in models],
        judges=[_judge(item) for item in judges],
        notes=list(BREAKDOWN_NOTES),
    )


def history_response(
    items: tuple[DashboardListItem, ...],
    total: int,
    filt: DashboardFilter,
) -> DashboardHistoryResponse:
    """Format one history page."""
    return DashboardHistoryResponse(
        timezone="UTC",
        filters=_filters(filt),
        items=[_item(item) for item in items],
        limit=filt.limit,
        offset=filt.offset,
        total=total,
    )


def fill_days(days: tuple[DaySums, ...], filt: DashboardFilter) -> tuple[DaySums, ...]:
    """Insert empty UTC days between an inclusive start and an exclusive end.

    Days outside that window are left as the database returned them. An empty
    day has no cost. The request count is zero because no row exists.
    """
    if filt.start is None or filt.end is None:
        return days
    start_day = filt.start.astimezone(timezone.utc).date()
    last_day = (filt.end - timedelta(microseconds=1)).astimezone(timezone.utc).date()
    if last_day < start_day:
        return days
    span = (last_day - start_day).days + 1
    if span > 366:
        return days
    observed = {item.day: item for item in days}
    filled: list[DaySums] = []
    cursor = start_day
    while cursor <= last_day:
        filled.append(
            observed.get(
                cursor,
                DaySums(
                    day=cursor,
                    requests=0,
                    cost_complete=None,
                    cost_estimated=None,
                    cost_unknown_requests=0,
                ),
            )
        )
        cursor += timedelta(days=1)
    return tuple(filled)


def _filters(filt: DashboardFilter) -> FilterView:
    return FilterView(
        start=filt.start,
        end=filt.end,
        model=filt.model,
        provider=filt.provider,
        status=filt.status,
    )


def _rate(part: int, whole: int) -> str | None:
    if whole <= 0:
        return None
    quantized = (Decimal(part) / Decimal(whole)).quantize(_RATE, rounding=ROUND_HALF_UP)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _latency(value: LatencySums) -> LatencyView:
    return LatencyView(
        p50=_ms(value.p50_ms),
        average=_ms(value.average_ms),
        known_count=value.known_count,
        missing_count=value.missing_count,
    )


def _ms(value: Decimal | None) -> float | None:
    if value is None:
        return None
    return float(value.quantize(_MILLISECOND, rounding=ROUND_HALF_UP))


def _cost(value: CostSums) -> CostView:
    return CostView(
        complete=money_to_api(value.complete),
        estimated=money_to_api(value.estimated),
        unknown_count=value.unknown_count,
    )


def _day(value: DaySums) -> DayView:
    return DayView(
        day=value.day.isoformat(),
        requests=value.requests,
        complete=money_to_api(value.cost_complete),
        estimated=money_to_api(value.cost_estimated),
        unknown_count=value.cost_unknown_requests,
    )


def _model(value: BreakdownRaw) -> ModelBreakdown:
    return ModelBreakdown(
        provider=value.provider,
        model=value.model,
        attempt_count=value.attempt_count,
        request_count=value.request_count,
        failed_attempts=value.failed_attempts,
        cost_complete=money_to_api(value.cost_complete),
        cost_estimated=money_to_api(value.cost_estimated),
        cost_unknown_attempts=value.cost_unknown_attempts,
        latency_p50_ms=_ms(value.latency_p50_ms),
        latency_known_attempts=value.latency_known_attempts,
        latency_missing_attempts=value.latency_missing_attempts,
    )


def _judge(value: JudgeRaw) -> JudgeBreakdown:
    return JudgeBreakdown(
        judge_model=value.judge_model,
        evaluations=value.evaluations,
        cost_complete=money_to_api(value.cost_complete),
        cost_estimated=money_to_api(value.cost_estimated),
        cost_unknown_evaluations=value.cost_unknown_evaluations,
    )


def _item(value: DashboardListItem) -> DashboardHistoryItem:
    return DashboardHistoryItem(
        request_id=str(value.request_id),
        created_at=value.created_at,
        status=value.status,
        provider=value.provider,
        model=value.model_id,
        final_provider=value.final_provider,
        final_model=value.final_model_id,
        prompt_preview=value.prompt_preview,
        prompt_truncated=value.prompt_truncated,
        total_cost=money_to_api(value.total_cost),
        cost_completeness=value.cost_completeness,
        estimated_savings=money_to_api(value.estimated_savings),
        savings_basis="same_token_volume",
        end_to_end_latency_ms=_ms(value.end_to_end_latency_ms),
        quality_verdict=value.quality_verdict,
        escalated=value.escalated,
        fallback_used=value.fallback_used,
        attempt_count=value.attempt_count,
    )
