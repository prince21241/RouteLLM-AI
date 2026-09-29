"""Read-only dashboard routes.

These endpoints do not call a provider. They read stored chat history.
"""

import logging
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query

from app.api.history import serialize_request
from app.api.history_schemas import RequestDetailResponse
from app.dashboard.filters import DashboardQueryError, parse_dashboard_filter
from app.dashboard.present import breakdown_response, history_response, overview_response
from app.dashboard.schemas import BreakdownResponse, DashboardHistoryResponse, OverviewResponse
from app.db.store import RequestStore

logger = logging.getLogger("app.dashboard")


def register_dashboard_routes(app: FastAPI) -> None:
    """Add the dashboard routes. They read ``app.state.store`` per request."""

    @app.get("/api/v1/dashboard/overview", response_model=OverviewResponse)
    async def overview(
        start: str | None = Query(default=None, alias="from"),
        end: str | None = Query(default=None, alias="to"),
        model: str | None = None,
        provider: str | None = None,
        status: str | None = None,
    ) -> OverviewResponse:
        filt = _filter(start=start, end=end, model=model, provider=provider, status=status)
        store = _store(app)
        try:
            raw = await store.dashboard_overview(filt)
        except Exception as exc:
            _raise_unavailable(exc)
        return overview_response(raw, filt)

    @app.get("/api/v1/dashboard/breakdown", response_model=BreakdownResponse)
    async def breakdown(
        start: str | None = Query(default=None, alias="from"),
        end: str | None = Query(default=None, alias="to"),
        model: str | None = None,
        provider: str | None = None,
        status: str | None = None,
    ) -> BreakdownResponse:
        filt = _filter(start=start, end=end, model=model, provider=provider, status=status)
        store = _store(app)
        try:
            models, judges = await store.dashboard_breakdown(filt)
        except Exception as exc:
            _raise_unavailable(exc)
        return breakdown_response(models, judges, filt)

    @app.get("/api/v1/dashboard/requests", response_model=DashboardHistoryResponse)
    async def requests(
        start: str | None = Query(default=None, alias="from"),
        end: str | None = Query(default=None, alias="to"),
        model: str | None = None,
        provider: str | None = None,
        status: str | None = None,
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> DashboardHistoryResponse:
        filt = _filter(
            start=start,
            end=end,
            model=model,
            provider=provider,
            status=status,
            limit=limit,
            offset=offset,
        )
        store = _store(app)
        try:
            items, total = await store.dashboard_requests(filt)
        except Exception as exc:
            _raise_unavailable(exc)
        return history_response(items, total, filt)

    @app.get("/api/v1/dashboard/requests/{request_id}", response_model=RequestDetailResponse)
    async def request_detail(request_id: str) -> RequestDetailResponse:
        parsed = _parse_id(request_id)
        store = _store(app)
        try:
            row = await store.get_request(parsed)
        except Exception as exc:
            _raise_unavailable(exc)
        if row is None:
            raise HTTPException(status_code=404, detail="Request not found")
        return serialize_request(row)


def _filter(
    *,
    start: str | None,
    end: str | None,
    model: str | None,
    provider: str | None,
    status: str | None,
    limit: int = 20,
    offset: int = 0,
):
    try:
        return parse_dashboard_filter(
            start=start,
            end=end,
            model=model,
            provider=provider,
            status=status,
            limit=limit,
            offset=offset,
        )
    except DashboardQueryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


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
    logger.error("Dashboard query failed (%s)", type(exc).__name__)
    raise HTTPException(status_code=503, detail="Dashboard data is unavailable") from None
