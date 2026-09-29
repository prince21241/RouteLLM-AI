"""Dashboard aggregates against PostgreSQL. No provider is called."""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import AttemptRow, EvaluationRow, RequestRow
from app.main import create_app

REQ1 = UUID("11111111-1111-4111-8111-111111111111")
REQ2 = UUID("22222222-2222-4222-8222-222222222222")
REQ3 = UUID("33333333-3333-4333-8333-333333333333")
REQ4 = UUID("44444444-4444-4444-8444-444444444444")
_SCRIPT = "<script>alert(1)</script>"


def test_empty_dashboard_does_not_turn_missing_values_into_zeros(store) -> None:
    body = _overview(store)

    assert body["request_count"] == 0
    assert body["attempt_count"] == 0
    assert body["error_rate"] is None
    assert body["fallback_rate"] is None
    assert body["request_cost"]["complete"] is None
    assert body["request_cost"]["estimated"] is None
    assert body["judge_cost"]["complete"] is None
    assert body["savings_estimate"] is None
    assert body["by_day"] == []


def test_mixed_history_counts_requests_once_and_keeps_unknowns(store) -> None:
    _seed(store)
    client = _client(store)
    overview = client.get("/api/v1/dashboard/overview")
    breakdown = client.get("/api/v1/dashboard/breakdown")
    window = client.get(
        "/api/v1/dashboard/overview",
        params={"from": "2026-09-29T00:00:00Z", "to": "2026-09-30T00:00:00Z"},
    )
    failed = client.get("/api/v1/dashboard/overview", params={"status": "failed"})
    ollama = client.get("/api/v1/dashboard/requests", params={"provider": "ollama", "limit": 20})
    page = client.get("/api/v1/dashboard/requests", params={"limit": 1, "offset": 0})
    detail = client.get(f"/api/v1/dashboard/requests/{REQ1}")
    gap = client.get(
        "/api/v1/dashboard/overview",
        params={"from": "2026-09-26T00:00:00Z", "to": "2026-09-28T00:00:00Z"},
    )

    assert overview.status_code == 200
    body = overview.json()
    assert body["request_count"] == 4
    assert body["attempt_count"] == 6
    assert body["error_rate"] == "0.25"
    assert body["escalation_rate"] == "0.25"
    assert body["fallback_used_requests"] == 1
    assert body["fallback_known_requests"] == 3
    assert body["fallback_unknown_requests"] == 1
    assert body["fallback_rate"] == "0.333333"
    assert body["quality"] == {"pass": 1, "fail": 1, "unknown": 2, "error": 0, "other": 0}
    assert body["attempt_purposes"] == {
        "routing": 3,
        "fallback": 1,
        "escalation": 1,
        "unknown": 1,
    }
    assert body["request_cost"] == {"complete": "0.03", "estimated": "0.01", "unknown_count": 2}
    assert body["judge_cost"]["complete"] == "0.005"
    assert body["judge_cost"]["unknown_count"] == 1
    assert body["savings_estimate"] == "0.02"
    assert body["savings_unknown_requests"] == 3
    assert body["request_latency_ms"]["p50"] == 20
    assert body["request_latency_ms"]["average"] == 25
    assert body["request_latency_ms"]["missing_count"] == 1
    assert body["attempt_latency_ms"]["p50"] == 10
    assert body["attempt_latency_ms"]["average"] == 13
    assert body["attempt_latency_ms"]["missing_count"] == 1

    models = {item["model"]: item for item in breakdown.json()["models"]}
    assert models["claude-sonnet-4-6"]["attempt_count"] == 2
    assert models["claude-sonnet-4-6"]["request_count"] == 2
    assert models["claude-sonnet-4-6"]["cost_complete"] == "0.22"
    assert models["gpt-5-nano"]["attempt_count"] == 3
    assert models["gpt-5-nano"]["request_count"] == 3
    assert models["gpt-5-nano"]["cost_complete"] == "0.01"
    assert models["gpt-5-nano"]["cost_unknown_attempts"] == 2
    assert models["llama3.2"]["cost_complete"] == "0"
    assert models["llama3.2"]["request_count"] == 1
    judges = breakdown.json()["judges"]
    assert judges[0]["judge_model"] == "gpt-5-nano"
    assert judges[0]["cost_complete"] == "0.005"
    assert judges[1]["judge_model"] is None
    assert judges[1]["cost_unknown_evaluations"] == 1

    only = window.json()
    assert only["request_count"] == 1
    assert only["attempt_count"] == 2
    assert failed.json()["request_count"] == 1
    assert failed.json()["fallback_rate"] is None
    assert failed.json()["request_cost"]["complete"] is None
    assert failed.json()["request_latency_ms"]["p50"] is None
    assert ollama.json()["total"] == 1
    assert ollama.json()["items"][0]["model"] == "llama3.2"

    listed = page.json()
    assert listed["total"] == 4
    assert len(listed["items"]) == 1
    assert listed["items"][0]["request_id"] == str(REQ2)
    assert len(listed["items"][0]["prompt_preview"]) == 80
    assert _SCRIPT + ("x" * 90) not in page.text

    stored = detail.json()
    assert [item["attempt_number"] for item in stored["attempts"]] == [1, 2]
    assert stored["attempts"][0]["purpose"] == "routing"
    assert stored["attempts"][1]["purpose"] == "fallback"
    assert stored["final_provider"] == "anthropic"
    assert stored["model"] == "gpt-5-nano"
    assert stored["total_cost"] is None

    days = gap.json()["by_day"]
    assert [item["day"] for item in days] == ["2026-09-26", "2026-09-27"]
    assert days[0]["requests"] == 0
    assert days[0]["complete"] is None
    assert days[1]["requests"] == 1
    assert days[1]["complete"] == "0.03"


def test_dashboard_database_error_is_generic(store) -> None:
    _request_store, engine = store

    async def rename(source: str, target: str) -> None:
        async with engine.begin() as connection:
            await connection.execute(text(f"ALTER TABLE {source} RENAME TO {target}"))

    asyncio.run(rename("requests", "requests_hidden"))
    try:
        response = _client(store).get("/api/v1/dashboard/overview")
        assert response.status_code == 503
        assert response.json()["detail"] == "Dashboard data is unavailable"
        assert "postgres" not in response.text.lower()
    finally:
        asyncio.run(rename("requests_hidden", "requests"))


def _overview(store) -> dict:
    response = _client(store).get("/api/v1/dashboard/overview")
    assert response.status_code == 200
    return response.json()


def _client(store) -> TestClient:
    request_store, _engine = store
    return TestClient(create_app(settings=Settings(_env_file=None), store=request_store))


def _seed(store) -> None:
    _request_store, engine = store

    async def insert() -> None:
        async with AsyncSession(engine, expire_on_commit=False) as session:
            async with session.begin():
                session.add_all(
                    [
                        _request(
                            REQ1,
                            _at(2026, 9, 29),
                            status="succeeded",
                            provider="openai",
                            model_id="gpt-5-nano",
                            end_to_end_latency_ms=Decimal("40"),
                            fallback_used=True,
                            fallback_reason="timeout",
                            final_provider="anthropic",
                            final_model_id="claude-sonnet-4-6",
                        ),
                        _request(
                            REQ2,
                            _at(2026, 9, 30),
                            status="failed",
                            prompt=_SCRIPT + ("x" * 90),
                            response_text=None,
                            fallback_used=None,
                        ),
                        _request(
                            REQ3,
                            _at(2026, 9, 28),
                            provider="ollama",
                            model_id="llama3.2",
                            total_cost=Decimal("0.01"),
                            cost_completeness="estimated",
                            estimated_savings=Decimal("0.02"),
                            end_to_end_latency_ms=Decimal("15"),
                            quality_verdict="pass",
                            fallback_used=False,
                        ),
                        _request(
                            REQ4,
                            _at(2026, 9, 27),
                            total_cost=Decimal("0.03"),
                            cost_completeness="complete",
                            end_to_end_latency_ms=Decimal("20"),
                            quality_verdict="fail",
                            escalated=True,
                            escalation_reason="explicit fail",
                            final_provider="anthropic",
                            final_model_id="claude-sonnet-4-6",
                            fallback_used=False,
                        ),
                    ]
                )
                session.add_all(
                    [
                        _attempt(REQ1, 2, "anthropic", "claude-sonnet-4-6", "succeeded", "fallback", Decimal("30"), Decimal("0.20"), "complete"),
                        _attempt(REQ1, 1, "openai", "gpt-5-nano", "failed", "routing", Decimal("10"), None, "unknown", "upstream timeout"),
                        _attempt(REQ2, 1, "openai", "gpt-5-nano", "failed", None, None, None, "unknown", "upstream timeout"),
                        _attempt(REQ3, 1, "ollama", "llama3.2", "succeeded", "routing", Decimal("5"), Decimal("0"), "complete"),
                        _attempt(REQ4, 1, "openai", "gpt-5-nano", "succeeded", "routing", Decimal("8"), Decimal("0.01"), "complete"),
                        _attempt(REQ4, 2, "anthropic", "claude-sonnet-4-6", "succeeded", "escalation", Decimal("12"), Decimal("0.02"), "complete"),
                        EvaluationRow(
                            id=uuid4(),
                            request_id=REQ1,
                            attempt_number=1,
                            source="live",
                            method="judge",
                            verdict="unknown",
                            reasons=["not graded"],
                            judge_model=None,
                            judge_cost=None,
                            judge_cost_completeness=None,
                            created_at=_at(2026, 9, 29),
                        ),
                        EvaluationRow(
                            id=uuid4(),
                            request_id=REQ3,
                            attempt_number=1,
                            source="live",
                            method="judge",
                            verdict="pass",
                            reasons=["matched"],
                            judge_model="gpt-5-nano",
                            judge_cost=Decimal("0.005"),
                            judge_cost_completeness="complete",
                            judge_latency_ms=Decimal("3"),
                            created_at=_at(2026, 9, 28),
                        ),
                    ]
                )

    asyncio.run(insert())


def _request(request_id: UUID, created_at: datetime, **overrides) -> RequestRow:
    values = dict(
        id=request_id,
        prompt="short prompt",
        system_prompt=None,
        created_at=created_at,
        updated_at=created_at,
        completed_at=created_at,
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
        end_to_end_latency_ms=None,
        premium_baseline_model=None,
        premium_baseline_cost=None,
        estimated_savings=None,
        quality_verdict=None,
        fallback_used=False,
        final_model_id=None,
        final_provider=None,
    )
    values.update(overrides)
    return RequestRow(**values)


def _attempt(
    request_id: UUID,
    number: int,
    provider: str,
    model: str,
    status: str,
    purpose: str | None,
    latency: Decimal | None,
    cost: Decimal | None,
    completeness: str,
    error: str | None = None,
) -> AttemptRow:
    return AttemptRow(
        id=uuid4(),
        request_id=request_id,
        attempt_number=number,
        provider=provider,
        configured_model_id=model,
        reported_model_id=model,
        status=status,
        provider_latency_ms=latency,
        estimated_cost=cost,
        cost_completeness=completeness,
        error_message=error,
        purpose=purpose,
        created_at=_at(2026, 9, 29),
    )


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)
