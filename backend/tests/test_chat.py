"""Offline chat endpoint tests. Generation is mocked."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import LLMResponse, ModelConfig, Provider, QualityTier
from app.config import Settings
from app.main import create_app
from app.providers.base import LLMProvider
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUpstreamError,
)
from app.routing.model_registry import ModelRegistry

UPSTREAM_MARKER = "raw-upstream-body-marker"
PROMPT = "Explain what an API is in two sentences"


def fictional(
    model_id: str,
    tier: QualityTier,
    *,
    provider: Provider = Provider.OPENAI,
) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_name=model_id,
        model_id=model_id,
        quality_tier=tier,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        context_window=1024,
        enabled=True,
        local=False,
    )


class RecordingProvider(LLMProvider):
    """In-memory provider. It records one call and can close."""

    def __init__(
        self,
        *,
        content: str = "An API is a contract between programs. It defines how they exchange data.",
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.closed = False
        self._content = content
        self._error = error

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        if self._error is not None:
            raise self._error
        return LLMResponse(
            provider=Provider.OPENAI,
            model="fictional-low",
            content=self._content,
            input_tokens=11,
            output_tokens=9,
            latency_ms=42.5,
            estimated_cost=None,
        )

    async def aclose(self) -> None:
        self.closed = True


def app_with(registry: ModelRegistry, provider: RecordingProvider, settings: Settings | None = None):
    resolved = settings or Settings(_env_file=None)
    return create_app(
        settings=resolved,
        registry=registry,
        provider_factory=lambda _model: provider,
    )


def test_chat_returns_routing_and_provider_metrics() -> None:
    provider = RecordingProvider()
    registry = ModelRegistry(
        [
            fictional("fictional-low", QualityTier.LOW),
            fictional("fictional-high", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ]
    )

    with TestClient(app_with(registry, provider)) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    assert response.status_code == 200
    body = response.json()
    assert body["response"].startswith("An API is a contract")
    assert body["request_id"]
    routing = body["routing"]
    assert routing["complexity_tier"] == "low"
    assert routing["complexity_score"] <= 0.30
    assert routing["provider"] == "openai"
    assert routing["model"] == "fictional-low"
    assert routing["selected_model_tier"] == "low"
    assert routing["degraded"] is False
    assert routing["escalated"] is False
    assert "short response requested" in routing["complexity_reasons"]
    metrics = body["metrics"]
    assert metrics["input_tokens"] == 11
    assert metrics["output_tokens"] == 9
    assert metrics["latency_ms"] == 42.5
    assert metrics["cost"] is None
    assert metrics["quality_score"] is None
    assert provider.calls == [(PROMPT, None)]
    assert provider.closed is True


def test_system_prompt_is_classified_and_sent_separately() -> None:
    provider = RecordingProvider(content="I looked at the function.")
    registry = ModelRegistry([fictional("fictional-low", QualityTier.LOW)])
    system_prompt = "Debug this Python function and return JSON."

    with TestClient(app_with(registry, provider)) as client:
        response = client.post(
            "/api/v1/chat",
            json={"prompt": "Hello", "system_prompt": system_prompt},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["complexity_tier"] != "low"
    assert "debugging" in body["routing"]["complexity_reasons"]
    assert body["routing"]["selected_model_tier"] == "low"
    assert body["routing"]["degraded"] is True
    assert "requested" not in body["routing"]["selection_reason"]
    assert provider.calls == [("Hello", system_prompt)]
    assert provider.closed is True


def test_provider_failure_is_sanitized_and_does_not_retry() -> None:
    provider = RecordingProvider(
        error=ProviderAuthenticationError("OpenAI authentication failed (HTTP 401)")
    )
    registry = ModelRegistry(
        [
            fictional("fictional-low", QualityTier.LOW),
            fictional("fictional-high", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ]
    )

    with TestClient(app_with(registry, provider)) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})

    assert response.status_code == 401
    detail = response.json()["detail"]
    assert detail == "OpenAI authentication failed (HTTP 401)"
    assert UPSTREAM_MARKER not in detail
    assert "Bearer" not in detail
    assert len(provider.calls) == 1
    assert provider.closed is True


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ProviderRateLimitError("OpenAI rate limit exceeded (HTTP 429)"), 429),
        (ProviderTimeoutError("OpenAI request timed out"), 504),
        (ProviderUpstreamError("OpenAI upstream error (HTTP 500)"), 502),
    ],
)
def test_provider_errors_map_to_http_status(error: Exception, status: int) -> None:
    provider = RecordingProvider(error=error)
    registry = ModelRegistry([fictional("fictional-low", QualityTier.LOW)])

    with TestClient(app_with(registry, provider)) as client:
        response = client.post("/api/v1/chat", json={"prompt": "Hi"})

    assert response.status_code == status
    assert UPSTREAM_MARKER not in response.text
    assert len(provider.calls) == 1


def test_blank_prompt_is_rejected_before_generation() -> None:
    provider = RecordingProvider()
    registry = ModelRegistry([fictional("fictional-low", QualityTier.LOW)])

    with TestClient(app_with(registry, provider)) as client:
        empty = client.post("/api/v1/chat", json={"prompt": "   "})
        missing = client.post("/api/v1/chat", json={})

    assert empty.status_code == 422
    assert empty.json()["detail"] == "Prompt must not be blank"
    assert missing.status_code == 422
    assert provider.calls == []


def test_input_size_limit_is_an_application_limit() -> None:
    provider = RecordingProvider()
    registry = ModelRegistry([fictional("fictional-low", QualityTier.LOW)])
    settings = Settings(_env_file=None, max_input_characters=20)

    with TestClient(app_with(registry, provider, settings)) as client:
        response = client.post(
            "/api/v1/chat",
            json={"prompt": "Hello", "system_prompt": "x" * 20},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Input exceeds the application limit of 20 characters"
    assert provider.calls == []


def test_no_enabled_models_returns_availability_error() -> None:
    provider = RecordingProvider()

    with TestClient(app_with(ModelRegistry(), provider)) as client:
        health = client.get("/health")
        chat = client.post("/api/v1/chat", json={"prompt": PROMPT})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert chat.status_code == 503
    assert chat.json()["detail"] == "No routing models are available"
    assert provider.calls == []


def test_health_without_credentials_does_not_enable_models(settings: Settings) -> None:
    app = create_app(settings=settings)

    assert app.state.registry.list_enabled() == []
    assert app.state.registry.get("gpt-5-nano").enabled is False

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
