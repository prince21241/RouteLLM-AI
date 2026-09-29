"""Dashboard API tests. Provider calls and PostgreSQL are not used."""

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.api.schemas import Provider
from app.config import Settings
from app.dashboard.filters import parse_dashboard_filter
from app.dashboard.present import fill_days
from app.dashboard.raw import CostSums, DaySums, LatencySums, OverviewRaw
from app.db.analytics import request_page_statement, request_totals_statement
from app.db.records import StoredAttempt, StoredEvaluation, StoredRequest
from app.main import create_app


def test_dashboard_page_is_served(client: TestClient) -> None:
    page = client.get("/dashboard")
    script = client.get("/static/dashboard.js")
    home = client.get("/")

    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert 'id="filters"' in page.text
    assert 'id="dashboard-status"' in page.text
    assert 'id="retry"' in page.text
    assert page.text.count('id="status"') == 1
    assert 'id="empty"' in page.text
    assert 'id="detail"' in page.text
    assert "Local use only" in page.text
    assert script.status_code == 200
    assert "innerHTML" not in script.text
    assert "insertAdjacentHTML" not in script.text
    assert "document.write" not in script.text
    assert 'get("demo")' in script.text
    assert 'id="chat-form"' in home.text
    assert 'href="/dashboard"' in home.text


def test_dashboard_without_a_database_is_unavailable(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/overview")

    assert response.status_code == 503
    assert response.json()["detail"] == "Request history is unavailable"


def test_dashboard_rejects_invalid_filters(client: TestClient) -> None:
    cases = [
        {"from": "2026-09-30T00:00:00Z", "to": "2026-09-29T00:00:00Z"},
        {"from": "2026-09-29T00:00:00"},
        {"provider": "azure"},
        {"status": "cancelled"},
        {"model": "gpt\nnano"},
        {"from": "2026-01-01T00:00:00Z", "to": "2027-02-01T00:00:00Z"},
    ]
    for params in cases:
        response = client.get("/api/v1/dashboard/overview", params=params)
        assert response.status_code == 422, params
        assert isinstance(response.json()["detail"], str)

    rejected = client.get("/api/v1/dashboard/requests", params={"limit": 101})
    assert rejected.status_code == 422


def test_date_bounds_are_utc_and_exclusive_at_the_end() -> None:
    filt = parse_dashboard_filter(
        start="2026-09-29",
        end="2026-09-30T04:00:00-04:00",
        model=None,
        provider=None,
        status=None,
    )

    assert filt.start == datetime(2026, 9, 29, tzinfo=timezone.utc)
    assert filt.end == datetime(2026, 9, 30, 8, tzinfo=timezone.utc)
    assert {item.value for item in Provider} == {"openai", "anthropic", "ollama"}


def test_empty_days_do_not_invent_zero_cost() -> None:
    filled = fill_days(
        (
            DaySums(
                day=date(2026, 9, 29),
                requests=2,
                cost_complete=None,
                cost_estimated=Decimal("0.01"),
                cost_unknown_requests=1,
            ),
        ),
        parse_dashboard_filter(
            start="2026-09-29T00:00:00Z",
            end="2026-10-01T00:00:00Z",
            model=None,
            provider=None,
            status=None,
        ),
    )

    assert [item.day for item in filled] == [date(2026, 9, 29), date(2026, 9, 30)]
    assert filled[0].cost_complete is None
    assert filled[0].cost_estimated == Decimal("0.01")
    assert filled[1].requests == 0
    assert filled[1].cost_complete is None
    assert filled[1].cost_estimated is None


def test_overview_sql_aggregates_requests_in_the_database() -> None:
    statement = request_totals_statement(
        parse_dashboard_filter(
            start=None,
            end=None,
            model=None,
            provider=None,
            status=None,
        )
    )
    page = request_page_statement(
        parse_dashboard_filter(
            start=None,
            end=None,
            model=None,
            provider=None,
            status=None,
            limit=20,
            offset=0,
        )
    )
    sql = str(statement.compile(dialect=postgresql.dialect())).lower()
    page_sql = str(page.compile(dialect=postgresql.dialect())).lower()

    assert "count(" in sql
    assert "filter" in sql
    assert "percentile_cont" in sql
    assert "sum(" in sql
    assert "substr(" in page_sql
    assert "limit" in page_sql
    assert "offset" in page_sql


def test_empty_overview_leaves_rates_and_costs_unknown() -> None:
    response = _client(_Store(overview=_raw())).get("/api/v1/dashboard/overview")
    body = response.json()

    assert response.status_code == 200
    assert body["request_count"] == 0
    assert body["attempt_count"] == 0
    assert body["error_rate"] is None
    assert body["escalation_rate"] is None
    assert body["fallback_rate"] is None
    assert body["request_cost"] == {"complete": None, "estimated": None, "unknown_count": 0}
    assert body["judge_cost"]["complete"] is None
    assert body["savings_estimate"] is None
    assert body["savings_basis"] == "same_token_volume"
    assert body["by_day"] == []
    assert body["timezone"] == "UTC"
    assert "Offline evaluation runs are not included." in body["notes"]


def test_mixed_results_keep_request_totals_separate_from_attempts() -> None:
    raw = _raw(
        requests=2,
        succeeded_requests=1,
        failed_requests=1,
        escalated_requests=1,
        fallback_used_requests=1,
        fallback_known_requests=1,
        fallback_unknown_requests=1,
        quality_pass=0,
        quality_fail=1,
        quality_unknown=1,
        attempt_count=3,
        routing_attempts=1,
        fallback_attempts=1,
        escalation_attempts=1,
        request_latency=LatencySums(Decimal("20"), Decimal("20"), 1, 1),
        attempt_latency=LatencySums(Decimal("10"), Decimal("15"), 2, 1),
        request_cost=CostSums(Decimal("0.03"), None, 1),
        judge_cost=CostSums(Decimal("0.005"), None, 1),
        savings_estimate=None,
        savings_known_requests=0,
        savings_unknown_requests=2,
    )

    body = _client(_Store(overview=raw)).get("/api/v1/dashboard/overview").json()

    assert body["request_count"] == 2
    assert body["attempt_count"] == 3
    assert body["error_rate"] == "0.5"
    assert body["escalation_rate"] == "0.5"
    assert body["fallback_rate"] == "1"
    assert body["fallback_unknown_requests"] == 1
    assert body["quality"] == {"pass": 0, "fail": 1, "unknown": 1, "error": 0, "other": 0}
    assert body["request_cost"]["complete"] == "0.03"
    assert body["request_cost"]["unknown_count"] == 1
    assert body["judge_cost"]["complete"] == "0.005"
    assert body["savings_estimate"] is None
    assert body["request_latency_ms"]["missing_count"] == 1
    assert body["attempt_purposes"] == {
        "routing": 1,
        "fallback": 1,
        "escalation": 1,
        "unknown": 0,
    }


def test_history_page_uses_filters_and_hides_the_full_prompt() -> None:
    store = _Store(items=(_item(),), total=5)
    response = _client(store).get(
        "/api/v1/dashboard/requests",
        params={"limit": 1, "offset": 1, "model": "gpt-5-nano", "status": "failed"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["total"] == 5
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert body["items"][0]["prompt_preview"] == "short preview"
    assert body["items"][0]["prompt_truncated"] is True
    assert "FULL PROMPT THAT MUST NOT LEAK" not in response.text
    assert store.filters[-1].model == "gpt-5-nano"
    assert store.filters[-1].status == "failed"
    assert store.filters[-1].offset == 1


def test_database_errors_do_not_return_the_exception_text() -> None:
    response = _client(_Store(error=RuntimeError("password=secret postgres"))).get(
        "/api/v1/dashboard/breakdown"
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Dashboard data is unavailable"
    assert "secret" not in response.text
    assert "postgres" not in response.text.lower()


def test_request_detail_keeps_attempts_and_unknown_cost() -> None:
    request_id = uuid4()
    store = _Store(detail=_stored(request_id))
    response = _client(store).get(f"/api/v1/dashboard/requests/{request_id}")
    missing = _client(store).get(f"/api/v1/dashboard/requests/{uuid4()}")
    body = response.json()

    assert response.status_code == 200
    assert body["prompt"] == "<script>alert(1)</script>"
    assert body["total_cost"] is None
    assert body["cost_completeness"] == "unknown"
    assert body["fallback_used"] is None
    assert body["fallback_reason"] == "timeout"
    assert [item["attempt_number"] for item in body["attempts"]] == [1, 2]
    assert [item["purpose"] for item in body["attempts"]] == ["routing", "fallback"]
    assert body["attempts"][1]["estimated_cost"] == "0.2"
    assert body["attempts"][0]["estimated_cost"] is None
    assert body["evaluations"][0]["judge_cost"] == "0.005"
    assert body["final_provider"] == "anthropic"
    assert body["model"] == "gpt-5-nano"
    assert missing.status_code == 404
    assert "api_key" not in response.text


class _Store:
    """Returns canned aggregates and does not open a database."""

    def __init__(self, overview=None, items=(), total=0, detail=None, error=None) -> None:
        self.overview = overview if overview is not None else _raw()
        self.items = items
        self.total = total
        self.detail = detail
        self.error = error
        self.filters = []

    async def dashboard_overview(self, filt):
        self.filters.append(filt)
        self._raise()
        return self.overview

    async def dashboard_breakdown(self, filt):
        self.filters.append(filt)
        self._raise()
        return (), ()

    async def dashboard_requests(self, filt):
        self.filters.append(filt)
        self._raise()
        return self.items, self.total

    async def get_request(self, request_id):
        self._raise()
        if self.detail is None or self.detail.request_id != request_id:
            return None
        return self.detail

    def _raise(self) -> None:
        if self.error is not None:
            raise self.error


def _client(store: _Store) -> TestClient:
    return TestClient(create_app(settings=Settings(_env_file=None), store=store))  # type: ignore[arg-type]


def _raw(**overrides) -> OverviewRaw:
    values = dict(
        requests=0,
        succeeded_requests=0,
        failed_requests=0,
        pending_requests=0,
        other_status_requests=0,
        escalated_requests=0,
        fallback_used_requests=0,
        fallback_known_requests=0,
        fallback_unknown_requests=0,
        quality_pass=0,
        quality_fail=0,
        quality_unknown=0,
        quality_error=0,
        quality_other=0,
        request_latency=LatencySums(None, None, 0, 0),
        attempt_count=0,
        routing_attempts=0,
        fallback_attempts=0,
        escalation_attempts=0,
        purpose_unknown_attempts=0,
        attempt_latency=LatencySums(None, None, 0, 0),
        request_cost=CostSums(None, None, 0),
        judge_cost=CostSums(None, None, 0),
        savings_estimate=None,
        savings_known_requests=0,
        savings_unknown_requests=0,
        days=(),
    )
    values.update(overrides)
    return OverviewRaw(**values)


def _item():
    from app.dashboard.raw import DashboardListItem

    return DashboardListItem(
        request_id=uuid4(),
        created_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        status="failed",
        provider="openai",
        model_id="gpt-5-nano",
        final_provider=None,
        final_model_id=None,
        prompt_preview="short preview",
        prompt_truncated=True,
        total_cost=None,
        cost_completeness="unknown",
        estimated_savings=None,
        end_to_end_latency_ms=None,
        quality_verdict=None,
        escalated=False,
        fallback_used=None,
        attempt_count=2,
    )


def _stored(request_id: UUID) -> StoredRequest:
    created = datetime(2026, 9, 29, tzinfo=timezone.utc)
    return StoredRequest(
        request_id=request_id,
        prompt="<script>alert(1)</script>",
        system_prompt=None,
        created_at=created,
        completed_at=created,
        status="succeeded",
        complexity_score=Decimal("0"),
        complexity_tier="low",
        complexity_reasons=[],
        provider="openai",
        model_id="gpt-5-nano",
        selected_model_tier="low",
        selection_reason="test",
        degraded=False,
        escalated=False,
        response_text="answer",
        total_cost=None,
        cost_completeness="unknown",
        end_to_end_latency_ms=Decimal("40"),
        premium_baseline_model="claude-sonnet-4-6",
        premium_baseline_cost=None,
        estimated_savings=None,
        attempts=[
            _attempt(1, "routing", "openai", "gpt-5-nano", "failed", None, "unknown"),
            _attempt(2, "fallback", "anthropic", "claude-sonnet-4-6", "succeeded", Decimal("0.20"), "complete"),
        ],
        evaluations=[
            StoredEvaluation(
                attempt_number=1,
                source="live",
                method="judge",
                verdict="unknown",
                score=None,
                reasons=["not graded"],
                judge_model="gpt-5-nano",
                judge_input_tokens=8,
                judge_output_tokens=4,
                judge_cost=Decimal("0.005"),
                judge_cost_completeness="complete",
                judge_latency_ms=Decimal("3"),
                error_message=None,
                created_at=created,
            )
        ],
        fallback_used=None,
        fallback_reason="timeout",
        fallback_skips=[],
        final_provider="anthropic",
        final_model_id="claude-sonnet-4-6",
    )


def _attempt(number, purpose, provider, model, status, cost, completeness) -> StoredAttempt:
    return StoredAttempt(
        attempt_number=number,
        provider=provider,
        configured_model_id=model,
        reported_model_id=model,
        status=status,
        input_tokens=None,
        output_tokens=None,
        cached_input_tokens=None,
        cache_write_input_tokens=None,
        cache_read_input_tokens=None,
        cache_write_5m_tokens=None,
        cache_write_1h_tokens=None,
        reasoning_tokens=None,
        provider_latency_ms=None,
        estimated_cost=cost,
        cost_completeness=completeness,
        error_message=None,
        pricing_snapshot=None,
        created_at=datetime(2026, 9, 29, tzinfo=timezone.utc),
        purpose=purpose,
        error_category=None,
    )
