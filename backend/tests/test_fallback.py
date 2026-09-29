"""Bounded provider fallback. Providers are in-memory."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import LLMResponse, Provider, QualityTier
from app.chat.fallback import select_fallback_candidates
from app.chat.service import ChatService
from app.config import Settings
from app.evaluation.schema import EvaluationResult
from app.main import create_app
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderSafetyError,
    ProviderTimeoutError,
    ProviderUpstreamError,
)
from app.providers.failure import ErrorCategory, classify_provider_error
from app.routing.complexity import ComplexityClassifier
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter
from tests.test_chat import PROMPT, fictional
from tests.test_escalation import CaptureStore, StubEvaluator


class Scripted:
    """One fake provider. ``action`` is text, an exception, or ``\"sleep\"``."""

    def __init__(self, owner: "Script", model_id: str) -> None:
        self.owner = owner
        self.model_id = model_id

    async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        del system_prompt
        self.owner.calls.append(self.model_id)
        self.owner.prompts.append(prompt)
        if self.owner.hang is not None and self.model_id == self.owner.hang:
            self.owner.started.set()
            await asyncio.Future()
        action = self.owner.actions[self.model_id]
        if action == "sleep":
            await asyncio.sleep(5)
        if isinstance(action, Exception):
            raise action
        return LLMResponse(
            provider=Provider.OPENAI,
            model=self.model_id,
            content=str(action),
            input_tokens=11,
            output_tokens=9,
            latency_ms=42.5,
        )

    async def aclose(self) -> None:
        self.owner.closed += 1


class Script:
    def __init__(self, actions: dict[str, object], *, hang: str | None = None) -> None:
        self.actions = actions
        self.hang = hang
        self.calls: list[str] = []
        self.prompts: list[str] = []
        self.closed = 0
        self.started = asyncio.Event()

    def factory(self, model: object):
        return Scripted(self, model.model_id)  # type: ignore[attr-defined]


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "fallback_enabled": True,
        "fallback_models": "claude-sonnet-4-6",
        "max_fallback_attempts": 1,
        "max_model_attempts": 3,
        "request_deadline_seconds": 30,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _models() -> list[object]:
    return [
        fictional("gpt-5-nano", QualityTier.LOW),
        fictional("claude-sonnet-4-6", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        fictional("llama3.2", QualityTier.LOW, provider=Provider.OLLAMA),
    ]


def _post(script: Script, settings: Settings, *, body: dict[str, object] | None = None, store: CaptureStore | None = None, evaluator: StubEvaluator | None = None):
    app = create_app(
        settings=settings,
        registry=ModelRegistry(_models()),  # type: ignore[arg-type]
        provider_factory=script.factory,
        evaluator=evaluator,
        store=store,  # type: ignore[arg-type]
    )
    payload = {"prompt": PROMPT} if body is None else body
    with TestClient(app) as client:
        return client.post("/api/v1/chat", json=payload)


def test_classify_fallback_eligibility() -> None:
    eligible = (
        ProviderTimeoutError("OpenAI request timed out"),
        ProviderTimeoutError("OpenAI connection failed"),
        ProviderRateLimitError("OpenAI rate limit exceeded (HTTP 429)"),
        ProviderUpstreamError("OpenAI upstream error (HTTP 500)"),
    )
    blocked = (
        ProviderResponseError("OpenAI returned an unexpected HTTP status (HTTP 400)"),
        ProviderAuthenticationError("OpenAI authentication failed (HTTP 401)"),
        ProviderAuthenticationError("OpenAI authentication failed (HTTP 403)"),
        ProviderSafetyError("OpenAI safety refusal"),
        ProviderResponseError("OpenAI response was incomplete (content_filter)"),
    )
    assert [classify_provider_error(exc).category for exc in eligible] == [
        ErrorCategory.TIMEOUT,
        ErrorCategory.CONNECTION,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.TRANSIENT_SERVER,
    ]
    assert all(classify_provider_error(exc).fallback_eligible for exc in eligible)
    assert [classify_provider_error(exc).category for exc in blocked] == [
        ErrorCategory.INVALID_REQUEST,
        ErrorCategory.AUTHENTICATION,
        ErrorCategory.PERMISSION,
        ErrorCategory.SAFETY_REFUSAL,
        ErrorCategory.SAFETY_REFUSAL,
    ]
    assert not any(classify_provider_error(exc).fallback_eligible for exc in blocked)


def test_fallback_disabled_preserves_single_attempt() -> None:
    script = Script(
        {
            "gpt-5-nano": ProviderTimeoutError("OpenAI request timed out"),
            "claude-sonnet-4-6": "backup",
        }
    )
    response = _post(script, Settings(_env_file=None, fallback_models="claude-sonnet-4-6"))
    assert response.status_code == 504
    assert script.calls == ["gpt-5-nano"]


def test_initial_success_does_not_fall_back() -> None:
    script = Script({"gpt-5-nano": "primary", "claude-sonnet-4-6": "backup"})
    response = _post(script, _settings())
    body = response.json()
    assert response.status_code == 200
    assert body["response"] == "primary"
    assert body["fallback_used"] is False
    assert script.calls == ["gpt-5-nano"]


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (ProviderTimeoutError("OpenAI request timed out"), "timeout"),
        (ProviderTimeoutError("OpenAI connection failed"), "connection"),
        (ProviderRateLimitError("OpenAI rate limit exceeded (HTTP 429)"), "rate_limit"),
        (ProviderUpstreamError("OpenAI upstream error (HTTP 500)"), "transient_server"),
    ],
)
def test_eligible_failures_fall_back_once(error: Exception, category: str) -> None:
    script = Script({"gpt-5-nano": error, "claude-sonnet-4-6": "backup"})
    store = CaptureStore()
    response = _post(script, _settings(), store=store)
    body = response.json()
    assert response.status_code == 200
    assert body["response"] == "backup"
    assert body["fallback_used"] is True
    assert body["fallback_reason"] == category
    assert body["final_provider"] == "anthropic"
    assert body["routing"]["model"] == "gpt-5-nano"
    assert script.calls == ["gpt-5-nano", "claude-sonnet-4-6"]
    assert [attempt.purpose for attempt in store.attempts] == ["routing", "fallback"]  # type: ignore[attr-defined]
    assert store.attempts[0].error_category == category  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ProviderResponseError("OpenAI returned an unexpected HTTP status (HTTP 400)"), 502),
        (ProviderAuthenticationError("OpenAI authentication failed (HTTP 401)"), 401),
        (ProviderAuthenticationError("OpenAI authentication failed (HTTP 403)"), 403),
        (ProviderSafetyError("OpenAI safety refusal"), 502),
    ],
)
def test_ineligible_failures_do_not_fall_back(error: Exception, status: int) -> None:
    script = Script({"gpt-5-nano": error, "claude-sonnet-4-6": "backup"})
    response = _post(script, _settings())
    assert response.status_code == status
    assert response.json()["detail"] == str(error)
    assert script.calls == ["gpt-5-nano"]


def test_provider_restriction_blocks_other_providers() -> None:
    script = Script(
        {
            "gpt-5-nano": ProviderTimeoutError("OpenAI request timed out"),
            "claude-sonnet-4-6": "backup",
        }
    )
    store = CaptureStore()
    response = _post(
        script,
        _settings(),
        body={"prompt": PROMPT, "provider": "openai"},
        store=store,
    )
    assert response.status_code == 504
    assert script.calls == ["gpt-5-nano"]
    assert store.final.fallback_skips == (  # type: ignore[attr-defined]
        ("claude-sonnet-4-6", "The request restricts the provider."),
    )


def test_unavailable_candidates_are_skipped() -> None:
    models = [
        fictional("gpt-5-nano", QualityTier.LOW),
        fictional("claude-sonnet-4-6", QualityTier.HIGH, provider=Provider.ANTHROPIC, enabled=False),
        fictional("llama3.2", QualityTier.LOW, provider=Provider.OLLAMA),
    ]
    script = Script(
        {
            "gpt-5-nano": ProviderUpstreamError("OpenAI upstream error (HTTP 500)"),
            "claude-sonnet-4-6": "nope",
            "llama3.2": "local",
        }
    )
    store = CaptureStore()
    app = create_app(
        settings=_settings(fallback_models="claude-sonnet-4-6,llama3.2"),
        registry=ModelRegistry(models),  # type: ignore[arg-type]
        provider_factory=script.factory,
        store=store,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})
    assert response.status_code == 200
    assert response.json()["response"] == "local"
    assert script.calls == ["gpt-5-nano", "llama3.2"]
    assert store.final.fallback_skips[0] == ("claude-sonnet-4-6", "The model is not enabled.")  # type: ignore[attr-defined]


def test_all_candidates_fail() -> None:
    script = Script(
        {
            "gpt-5-nano": ProviderUpstreamError("OpenAI upstream error (HTTP 500)"),
            "claude-sonnet-4-6": ProviderUpstreamError("Anthropic upstream error (HTTP 500)"),
        }
    )
    store = CaptureStore()
    response = _post(script, _settings(), store=store)
    assert response.status_code == 502
    assert response.json()["detail"] == "All configured providers failed"
    assert script.calls == ["gpt-5-nano", "claude-sonnet-4-6"]
    assert len(store.attempts) == 2
    assert store.final.status == "failed"  # type: ignore[attr-defined]


def test_deadline_stops_further_attempts() -> None:
    script = Script({"gpt-5-nano": "sleep", "claude-sonnet-4-6": "backup"})
    response = _post(
        script,
        _settings(request_deadline_seconds=0.05, openai_timeout_seconds=30),
    )
    assert response.status_code == 504
    assert script.calls == ["gpt-5-nano"]


def test_cancellation_does_not_start_fallback() -> None:
    script = Script(
        {"gpt-5-nano": "unused", "claude-sonnet-4-6": "backup"},
        hang="gpt-5-nano",
    )
    service = ChatService(
        settings=_settings(),
        registry=ModelRegistry(_models()),  # type: ignore[arg-type]
        classifier=ComplexityClassifier(),
        router=ModelRouter(),
        provider_factory=script.factory,
    )

    async def scenario() -> None:
        task = asyncio.create_task(service.complete(PROMPT))
        await script.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert script.calls == ["gpt-5-nano"]


def test_duplicate_pairs_are_not_called() -> None:
    registry = ModelRegistry(_models())  # type: ignore[arg-type]
    chosen, skipped = select_fallback_candidates(
        ("gpt-5-nano", "claude-sonnet-4-6"),
        registry,
        used={("openai", "gpt-5-nano")},
        failed_provider=Provider.OPENAI,
        restricted_provider=None,
        limit=1,
    )
    assert [model.model_id for model in chosen] == ["claude-sonnet-4-6"]
    assert skipped[0].reason == "This provider and model were already called."


def test_shared_attempt_budget_blocks_escalation_after_fallback() -> None:
    script = Script(
        {
            "gpt-5-nano": ProviderTimeoutError("OpenAI request timed out"),
            "claude-sonnet-4-6": "backup",
            "llama3.2": "should-not-run",
        }
    )
    evaluator = StubEvaluator(EvaluationResult(verdict="fail", score=0, reasons=("fail",), method="live"))
    response = _post(
        script,
        _settings(
            max_model_attempts=2,
            quality_evaluation_enabled=True,
            escalation_enabled=True,
            fallback_models="claude-sonnet-4-6,llama3.2",
        ),
        evaluator=evaluator,
    )
    assert response.status_code == 200
    assert response.json()["response"] == "backup"
    assert response.json()["routing"]["escalated"] is False
    assert script.calls == ["gpt-5-nano", "claude-sonnet-4-6"]


def test_failed_escalation_keeps_the_earlier_answer() -> None:
    script = Script(
        {
            "gpt-5-nano": "primary",
            "claude-sonnet-4-6": ProviderUpstreamError("Anthropic upstream error (HTTP 500)"),
            "llama3.2": ProviderUpstreamError("Ollama upstream error (HTTP 500)"),
        }
    )
    evaluator = StubEvaluator(EvaluationResult(verdict="fail", score=0, reasons=("fail",), method="live"))
    response = _post(
        script,
        _settings(
            quality_evaluation_enabled=True,
            escalation_enabled=True,
            fallback_models="llama3.2",
        ),
        evaluator=evaluator,
    )
    body = response.json()
    assert body["response"] == "primary"
    assert body["routing"]["escalated"] is False
    assert body["fallback_used"] is True
    assert script.calls == ["gpt-5-nano", "claude-sonnet-4-6", "llama3.2"]


def test_persistence_failure_does_not_fall_back() -> None:
    script = Script({"gpt-5-nano": "primary", "claude-sonnet-4-6": "backup"})

    class BoomStore(CaptureStore):
        async def save_attempt(self, outcome: object) -> None:
            raise RuntimeError("database unavailable")

    response = _post(script, _settings(), store=BoomStore())
    body = response.json()
    assert response.status_code == 200
    assert body["response"] == "primary"
    assert body["persistence_status"] == "failed"
    assert script.calls == ["gpt-5-nano"]


def test_unknown_timeout_cost_stays_unknown() -> None:
    script = Script(
        {
            "gpt-5-nano": ProviderTimeoutError("OpenAI request timed out"),
            "claude-sonnet-4-6": "backup",
        }
    )
    response = _post(script, _settings())
    body = response.json()
    assert body["metrics"]["cost"] is None
    assert body["metrics"]["cost_completeness"] == "unknown"
    assert body["estimated_savings"] is None
    assert body["savings_basis"] == "same_token_volume"
    assert body["metrics"]["input_tokens"] == 11


def test_same_provider_candidate_is_skipped() -> None:
    chosen, skipped = select_fallback_candidates(
        ("gpt-5-nano",),
        ModelRegistry([fictional("gpt-5-nano", QualityTier.LOW)]),
        used=set(),
        failed_provider=Provider.OPENAI,
        restricted_provider=None,
        limit=1,
    )
    assert chosen == []
    assert skipped[0].reason == "Fallback requires a different provider."
