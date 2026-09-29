"""Chat lifecycle with an in-memory store. Provider calls stay mocked."""

from decimal import Decimal

from fastapi.testclient import TestClient

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.config import Settings
from app.main import create_app
from app.routing.model_registry import ModelRegistry
from tests.test_chat import PROMPT, RecordingProvider, fictional


class MemoryStore:
    """Records lifecycle calls without opening a database."""

    def __init__(self, *, fail_pending: bool = False, fail_final: bool = False) -> None:
        self.fail_pending = fail_pending
        self.fail_final = fail_final
        self.pending: list[object] = []
        self.outcomes: list[object] = []

    async def create_pending(self, pending: object) -> None:
        if self.fail_pending:
            raise RuntimeError("database unavailable")
        self.pending.append(pending)

    async def record_success(self, outcome: object) -> None:
        if self.fail_final:
            raise RuntimeError("database unavailable")
        self.outcomes.append(outcome)

    async def record_failure(self, outcome: object) -> None:
        if self.fail_final:
            raise RuntimeError("database unavailable")
        self.outcomes.append(outcome)

    async def ping(self) -> None:
        return None


def _app(provider: RecordingProvider, store: MemoryStore, settings: Settings | None = None):
    registry = ModelRegistry(
        [
            ModelConfig(
                provider=Provider.OPENAI,
                model_name="GPT-5 nano",
                model_id="gpt-5-nano",
                quality_tier=QualityTier.LOW,
                input_cost_per_million_tokens=Decimal("0"),
                output_cost_per_million_tokens=Decimal("0"),
                context_window=400_000,
                enabled=True,
                local=False,
            )
        ]
    )
    return create_app(
        settings=settings or Settings(_env_file=None),
        registry=registry,
        provider_factory=lambda _model: provider,
        store=store,  # type: ignore[arg-type]
    )


def test_database_failure_before_generation_does_not_call_the_provider() -> None:
    provider = RecordingProvider()
    store = MemoryStore(fail_pending=True)

    with TestClient(_app(provider, store)) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    assert response.status_code == 503
    assert response.json()["detail"] == "Request storage is unavailable"
    assert "database unavailable" not in response.text
    assert provider.calls == []
    assert store.pending == []


def test_persistence_failure_after_generation_returns_the_answer_once() -> None:
    provider = RecordingProvider()
    store = MemoryStore(fail_final=True)

    with TestClient(_app(provider, store)) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    body = response.json()
    assert response.status_code == 200
    assert body["response"].startswith("An API is a contract")
    assert body["request_id"]
    assert body["persistence_status"] == "failed"
    assert body["persistence_warning"] == "The response was generated but could not be saved."
    assert provider.calls == [(PROMPT, None)]
    assert len(store.pending) == 1
    assert store.outcomes == []


def test_priced_success_uses_the_verified_book_not_registry_zeros() -> None:
    provider = RecordingProvider()
    store = MemoryStore()

    with TestClient(_app(provider, store)) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    body = response.json()
    assert response.status_code == 200
    assert body["metrics"]["cost"] == "0.00000415"
    assert body["metrics"]["cost_completeness"] == "estimated"
    assert body["metrics"]["quality_score"] is None
    assert body["routing"]["escalated"] is False
    assert body["baseline_model"] == "claude-sonnet-4-6"
    assert body["baseline_cost"] == "0.000168"
    assert body["estimated_savings"] == "0.00016385"
    assert body["savings_basis"] == "same_token_volume"
    assert body["persistence_status"] == "stored"
    assert body["metrics"]["latency_ms"] == 42.5
    assert body["end_to_end_latency_ms"] >= 0
    assert len(store.outcomes) == 1
    assert store.outcomes[0].provider_latency_ms == Decimal("42.500")
    assert store.outcomes[0].final_model_id == "gpt-5-nano"
    assert store.outcomes[0].final_provider == "openai"
    assert body["final_provider"] == "openai"
    assert body["returned_model"] == "gpt-5-nano"
    assert provider.calls == [(PROMPT, None)]


def test_zero_baseline_can_produce_negative_savings() -> None:
    provider = RecordingProvider()
    settings = Settings(_env_file=None, premium_baseline_model="llama3.2")

    with TestClient(_app(provider, MemoryStore(), settings)) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    body = response.json()
    assert body["baseline_model"] == "llama3.2"
    assert body["baseline_cost"] == "0"
    assert body["estimated_savings"] == "-0.00000415"


def test_unknown_model_cost_stays_null() -> None:
    provider = RecordingProvider()
    registry = ModelRegistry([fictional("fictional-low", QualityTier.LOW)])
    app = create_app(
        settings=Settings(_env_file=None),
        registry=registry,
        provider_factory=lambda _model: provider,
        store=MemoryStore(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    body = response.json()
    assert body["metrics"]["cost"] is None
    assert body["metrics"]["cost_completeness"] == "unknown"
    assert body["estimated_savings"] is None
    assert body["baseline_cost"] == "0.000168"


def test_history_and_readiness_without_a_database() -> None:
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        history = client.get("/api/v1/requests")
        missing = client.get(
            "/api/v1/requests/00000000-0000-0000-0000-000000000000"
        )
        invalid_page = client.get("/api/v1/requests", params={"limit": 0})

    assert ready.status_code == 503
    assert ready.json()["detail"] == "Database is not configured"
    assert history.status_code == 503
    assert history.json()["detail"] == "Request history is unavailable"
    assert missing.status_code == 503
    assert invalid_page.status_code == 422
