"""PostgreSQL integration tests.

These tests migrate only a database whose name ends with ``_test``.
They do not run as part of the default offline suite.
"""

import asyncio
from decimal import Decimal
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.main import create_app
from app.providers.errors import ProviderUpstreamError
from app.routing.catalog import _VERIFIED, verified_metadata
from tests.test_chat import PROMPT, RecordingProvider
from tests.test_lifecycle import _app


def test_migrations_create_request_tables(store) -> None:
    _request_store, engine = store

    async def table_names():
        async with engine.connect() as connection:
            rows = await connection.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            )
            names = {row[0] for row in rows}
            verdict = await connection.scalar(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'requests' AND column_name = 'quality_verdict'"
                )
            )
            fallback_used = await connection.scalar(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'requests' AND column_name = 'fallback_used'"
                )
            )
            cost_type = await connection.scalar(
                text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_name = 'requests' AND column_name = 'total_cost'"
                )
            )
        return names, cost_type, verdict, fallback_used

    names, cost_type, verdict, fallback_used = asyncio.run(table_names())
    assert {"requests", "attempts", "evaluations", "alembic_version"} <= names
    assert cost_type == "numeric"
    assert verdict == "quality_verdict"
    assert fallback_used == "fallback_used"


def test_success_is_persisted_with_its_attempt(store) -> None:
    request_store, _engine = store
    provider = RecordingProvider()

    with TestClient(_app(provider, request_store)) as client:
        created = client.post(
            "/api/v1/chat",
            json={"prompt": PROMPT, "system_prompt": "Be brief"},
        )
        assert created.status_code == 200
        body = created.json()
        detail = client.get(f"/api/v1/requests/{body['request_id']}")
        missing = client.get(f"/api/v1/requests/{uuid4()}")

    assert provider.calls == [(PROMPT, "Be brief")]
    assert detail.status_code == 200
    stored = detail.json()
    assert stored["prompt"] == PROMPT
    assert stored["system_prompt"] == "Be brief"
    assert stored["status"] == "succeeded"
    assert stored["total_cost"] == "0.00000415"
    assert stored["response"].startswith("An API is a contract")
    assert len(stored["attempts"]) == 1
    attempt = stored["attempts"][0]
    assert attempt["attempt_number"] == 1
    assert attempt["configured_model_id"] == "gpt-5-nano"
    assert attempt["reported_model_id"] == "fictional-low"
    assert attempt["provider_latency_ms"] == 42.5
    assert attempt["input_tokens"] == 11
    assert attempt["output_tokens"] == 9
    assert attempt["pricing_snapshot"]["input_per_million"] == "0.05"
    assert attempt["pricing_snapshot"]["verified_on"] == "2026-09-28"
    assert missing.status_code == 404


def test_failed_request_keeps_sanitized_error_and_null_cost(store) -> None:
    request_store, _engine = store
    provider = RecordingProvider(
        error=ProviderUpstreamError("OpenAI upstream error (HTTP 500)")
    )

    with TestClient(_app(provider, request_store)) as client:
        response = client.post("/api/v1/chat", json={"prompt": "This request fails"})
        listed = client.get("/api/v1/requests")
        request_id = listed.json()["items"][0]["request_id"]
        detail = client.get(f"/api/v1/requests/{request_id}")

    assert response.status_code == 502
    assert response.json()["detail"] == "OpenAI upstream error (HTTP 500)"
    assert provider.calls == [("This request fails", None)]
    stored = detail.json()
    assert stored["status"] == "failed"
    assert stored["total_cost"] is None
    assert stored["cost_completeness"] == "unknown"
    assert stored["attempts"][0]["error_message"] == "OpenAI upstream error (HTTP 500)"
    assert stored["attempts"][0]["input_tokens"] is None
    assert stored["attempts"][0]["estimated_cost"] is None


def test_pricing_snapshot_survives_a_later_price_change(store, monkeypatch: pytest.MonkeyPatch) -> None:
    request_store, _engine = store
    provider = RecordingProvider()

    with TestClient(_app(provider, request_store)) as client:
        created = client.post("/api/v1/chat", json={"prompt": PROMPT})
        original = verified_metadata("gpt-5-nano")
        assert original is not None
        changed = original.__class__(
            **{
                **original.__dict__,
                "input_cost_per_million_tokens": Decimal("9"),
            }
        )
        monkeypatch.setitem(_VERIFIED, "gpt-5-nano", changed)
        detail = client.get(f"/api/v1/requests/{created.json()['request_id']}")

    assert detail.json()["attempts"][0]["pricing_snapshot"]["input_per_million"] == "0.05"
    assert provider.calls == [(PROMPT, None)]


def test_history_is_paginated_without_prompts(store) -> None:
    request_store, _engine = store
    prompts = [
        "history marker alpha unique",
        "history marker beta unique",
        "history marker gamma unique",
    ]
    provider = RecordingProvider()

    with TestClient(_app(provider, request_store)) as client:
        for prompt in prompts:
            assert client.post("/api/v1/chat", json={"prompt": prompt}).status_code == 200
        page = client.get("/api/v1/requests", params={"limit": 2, "offset": 0})
        rest = client.get("/api/v1/requests", params={"limit": 2, "offset": 2})
        rejected = client.get("/api/v1/requests", params={"limit": 101})
        missing = client.get(f"/api/v1/requests/{uuid4()}")

    body = page.json()
    assert page.status_code == 200
    assert body["total"] == 3
    assert len(body["items"]) == 2
    assert len(rest.json()["items"]) == 1
    assert rejected.status_code == 422
    assert missing.status_code == 404
    rendered = page.text + rest.text
    for prompt in prompts:
        assert prompt not in rendered
    created = [item["created_at"] for item in body["items"]]
    assert created == sorted(created, reverse=True)
    ids = [item["request_id"] for item in body["items"] + rest.json()["items"]]
    assert len(ids) == len(set(ids)) == 3


def test_database_failure_before_generation_skips_the_provider(store) -> None:
    request_store, engine = store
    provider = RecordingProvider()

    async def rename(source: str, target: str) -> None:
        async with engine.begin() as connection:
            await connection.execute(text(f"ALTER TABLE {source} RENAME TO {target}"))

    asyncio.run(rename("requests", "requests_hidden"))
    try:
        with TestClient(_app(provider, request_store)) as client:
            response = client.post("/api/v1/chat", json={"prompt": PROMPT})
        assert response.status_code == 503
        assert response.json()["detail"] == "Request storage is unavailable"
        assert "postgresql" not in response.text.lower()
        assert provider.calls == []
    finally:
        asyncio.run(rename("requests_hidden", "requests"))


def test_persistence_failure_after_generation_does_not_repeat_the_call(store) -> None:
    request_store, engine = store
    provider = RecordingProvider()

    async def rename(source: str, target: str) -> None:
        async with engine.begin() as connection:
            await connection.execute(text(f"ALTER TABLE {source} RENAME TO {target}"))

    asyncio.run(rename("attempts", "attempts_hidden"))
    try:
        with TestClient(_app(provider, request_store)) as client:
            response = client.post("/api/v1/chat", json={"prompt": PROMPT})
        body = response.json()
        assert response.status_code == 200
        assert body["response"].startswith("An API is a contract")
        assert body["request_id"]
        assert body["persistence_status"] == "failed"
        assert body["persistence_warning"] == "The response was generated but could not be saved."
        assert provider.calls == [(PROMPT, None)]
    finally:
        asyncio.run(rename("attempts_hidden", "attempts"))
