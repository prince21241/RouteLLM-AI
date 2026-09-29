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
    enabled: bool = True,
) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_name=model_id,
        model_id=model_id,
        quality_tier=tier,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        context_window=1024,
        enabled=enabled,
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


def test_chat_options_list_catalog_models_without_credentials(client: TestClient) -> None:
    response = client.get("/api/v1/chat/options")

    assert response.status_code == 200
    body = response.json()
    assert body["requests_are_independent"] is True
    assert body["model_selection_supported"] is False
    assert body["provider_restriction_supported"] is True
    assert body["max_input_characters"] == 8000
    assert "not given this conversation" in body["context_note"]
    assert body["quality_evaluation"] == {"enabled": False, "per_request_override": False}
    assert body["escalation"]["enabled"] is False
    assert body["fallback"]["per_request_override"] is False
    assert [model["model_id"] for model in body["models"]] == [
        "gpt-5-nano",
        "claude-sonnet-4-6",
        "llama3.2",
    ]
    assert all(model["enabled"] is False for model in body["models"])
    assert _keys(body).isdisjoint({"api_key", "openai_api_key", "anthropic_api_key", "ollama_api_key"})


def test_chat_options_reflect_enabled_models_and_server_features() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="fictional-openai-key",
        quality_evaluation_enabled=True,
        escalation_enabled=True,
        fallback_enabled=True,
    )

    with TestClient(create_app(settings=settings)) as client:
        response = client.get("/api/v1/chat/options")

    body = response.json()
    enabled = {model["model_id"]: model["enabled"] for model in body["models"]}
    assert enabled["gpt-5-nano"] is True
    assert enabled["claude-sonnet-4-6"] is False
    assert enabled["llama3.2"] is False
    assert body["quality_evaluation"]["enabled"] is True
    assert body["escalation"]["enabled"] is True
    assert body["fallback"]["enabled"] is True
    assert body["model_selection_supported"] is False
    assert "fictional-openai-key" not in response.text


def test_chat_page_keeps_untrusted_output_out_of_html_strings() -> None:
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        page = client.get("/")
        script = client.get("/static/app.js")

    assert page.status_code == 200
    assert 'id="chat-form"' in page.text
    assert 'id="use-prompt"' in page.text
    assert 'href="/dashboard"' in page.text
    assert "innerHTML" not in script.text
    assert "insertAdjacentHTML" not in script.text
    assert "document.write" not in script.text


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found.update(_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_keys(item))
    return found
