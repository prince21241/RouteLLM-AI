"""Bounded escalation. Generation and judging are mocked."""

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import LLMResponse, Provider, QualityTier, ReportedUsage
from app.config import Settings
from app.evaluation.schema import EvaluationResult
from app.main import create_app
from app.pricing.service import CostCompleteness, CostEstimate, combine_costs, estimate_cost, lookup_prices
from app.providers.errors import ProviderUpstreamError
from app.routing.model_registry import ModelRegistry
from tests.test_chat import PROMPT, fictional


class StubEvaluator:
    def __init__(self, result: EvaluationResult) -> None:
        self.result = result
        self.calls = 0

    async def evaluate(self, prompt: str, answer: str) -> EvaluationResult:
        self.calls += 1
        return self.result


class SequenceProvider:
    def __init__(self, contents: list[str], *, fail_at: int | None = None) -> None:
        self.contents = contents
        self.fail_at = fail_at
        self.calls: list[tuple[str, str | None]] = []
        self.closed = False

    async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        self.calls.append((prompt, system_prompt))
        if self.fail_at == len(self.calls):
            raise ProviderUpstreamError("OpenAI upstream error (HTTP 500)")
        content = self.contents[len(self.calls) - 1]
        return LLMResponse(
            provider=Provider.OPENAI,
            model="returned",
            content=content,
            input_tokens=11,
            output_tokens=9,
            latency_ms=42.5,
        )

    async def aclose(self) -> None:
        self.closed = True


class CaptureStore:
    def __init__(self) -> None:
        self.pending: list[object] = []
        self.attempts: list[object] = []
        self.final: object | None = None

    async def create_pending(self, pending: object) -> None:
        self.pending.append(pending)

    async def save_attempt(self, outcome: object) -> None:
        self.attempts.append(outcome)

    async def finalize_request(self, outcome: object) -> None:
        self.final = outcome

    async def record_success(self, outcome: object) -> None:
        self.final = outcome

    async def record_failure(self, outcome: object) -> None:
        self.final = outcome


def _result(verdict: str, *, score: float | None = None, judge: CostEstimate | None = None) -> EvaluationResult:
    return EvaluationResult(
        verdict=verdict,  # type: ignore[arg-type]
        score=score,
        reasons=(verdict,),
        method="live",
        judge_model="gpt-5-nano" if judge is not None else None,
        judge_cost=judge,
        judge_latency_ms=3.0 if judge is not None else None,
    )


def _post(
    provider: SequenceProvider,
    evaluator: StubEvaluator,
    *,
    settings: Settings,
    models: list[object],
    store: CaptureStore | None = None,
):
    app = create_app(
        settings=settings,
        registry=ModelRegistry(models),  # type: ignore[arg-type]
        provider_factory=lambda _model: provider,
        evaluator=evaluator,
        store=store,  # type: ignore[arg-type]
    )
    with TestClient(app) as client:
        return client.post("/api/v1/chat", json={"prompt": PROMPT})


def test_disabled_quality_does_not_escalate() -> None:
    provider = SequenceProvider(["first answer", "second answer"])
    evaluator = StubEvaluator(_result("fail", score=0))

    response = _post(
        provider,
        evaluator,
        settings=Settings(_env_file=None),
        models=[fictional("low-model", QualityTier.LOW), fictional("high-model", QualityTier.HIGH)],
    )

    body = response.json()
    assert response.status_code == 200
    assert evaluator.calls == 0
    assert len(provider.calls) == 1
    assert body["routing"]["escalated"] is False
    assert body["metrics"]["quality_score"] is None
    assert body["quality_verdict"] is None
    assert body["response"] == "first answer"


def test_quality_pass_does_not_escalate() -> None:
    provider = SequenceProvider(["good answer", "should not run"])
    evaluator = StubEvaluator(_result("pass", score=1))
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
    )

    response = _post(
        provider,
        evaluator,
        settings=settings,
        models=[fictional("low-model", QualityTier.LOW), fictional("high-model", QualityTier.HIGH)],
    )

    body = response.json()
    assert body["routing"]["escalated"] is False
    assert body["quality_verdict"] == "pass"
    assert body["metrics"]["quality_score"] == 1
    assert len(provider.calls) == 1
    assert body["response"] == "good answer"


def test_quality_failure_escalates_once() -> None:
    provider = SequenceProvider(["weak answer", "strong answer"])
    evaluator = StubEvaluator(_result("fail", score=0))
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
    )

    response = _post(
        provider,
        evaluator,
        settings=settings,
        models=[
            fictional("gpt-5-nano", QualityTier.LOW),
            fictional("claude-sonnet-4-6", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ],
    )

    body = response.json()
    assert len(provider.calls) == 2
    assert body["routing"]["escalated"] is True
    assert body["response"] == "strong answer"
    assert body["returned_attempt"] == 2
    assert body["routing"]["model"] == "gpt-5-nano"
    assert body["returned_model"] == "claude-sonnet-4-6"
    assert body["escalation_error"] is None
    assert "failed quality" in body["escalation_reason"]


def test_unknown_and_evaluator_error_do_not_escalate() -> None:
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
    )
    models = [fictional("low-model", QualityTier.LOW), fictional("high-model", QualityTier.HIGH)]
    for verdict in ("unknown", "error"):
        provider = SequenceProvider(["kept", "not used"])
        response = _post(
            provider,
            StubEvaluator(_result(verdict)),
            settings=settings,
            models=models,
        )
        assert len(provider.calls) == 1
        assert response.json()["routing"]["escalated"] is False
        assert response.json()["quality_verdict"] == verdict
        assert response.json()["response"] == "kept"


def test_missing_stronger_model_keeps_the_initial_answer() -> None:
    provider = SequenceProvider(["only answer", "not used"])
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
    )

    response = _post(
        provider,
        StubEvaluator(_result("fail", score=0)),
        settings=settings,
        models=[fictional("gpt-5-nano", QualityTier.LOW)],
    )

    body = response.json()
    assert len(provider.calls) == 1
    assert body["response"] == "only answer"
    assert body["routing"]["escalated"] is False
    assert body["escalation_error"] == "No stronger eligible model is available."


def test_same_model_is_not_used_for_escalation() -> None:
    provider = SequenceProvider(["only answer", "not used"])
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
        escalation_model="gpt-5-nano",
    )

    response = _post(
        provider,
        StubEvaluator(_result("fail", score=0)),
        settings=settings,
        models=[fictional("gpt-5-nano", QualityTier.LOW)],
    )

    assert len(provider.calls) == 1
    assert response.json()["escalation_error"] == "The escalation model is the same as the selected model."


def test_failed_escalation_returns_the_initial_answer() -> None:
    provider = SequenceProvider(["kept answer", "unused"], fail_at=2)
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
    )

    response = _post(
        provider,
        StubEvaluator(_result("fail", score=0)),
        settings=settings,
        models=[fictional("low-model", QualityTier.LOW), fictional("high-model", QualityTier.HIGH)],
    )

    body = response.json()
    assert response.status_code == 200
    assert len(provider.calls) == 2
    assert body["response"] == "kept answer"
    assert body["routing"]["escalated"] is False
    assert body["escalation_error"] == "OpenAI upstream error (HTTP 500)"


def test_cost_includes_both_attempts_and_the_judge() -> None:
    judge = CostEstimate(total=Decimal("0.0000005"), completeness=CostCompleteness.ESTIMATED)
    provider = SequenceProvider(["weak", "strong"])
    settings = Settings(
        _env_file=None,
        quality_evaluation_enabled=True,
        escalation_enabled=True,
    )
    store = CaptureStore()
    response = _post(
        provider,
        StubEvaluator(_result("fail", score=0, judge=judge)),
        settings=settings,
        models=[
            fictional("gpt-5-nano", QualityTier.LOW),
            fictional("claude-sonnet-4-6", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ],
        store=store,
    )
    reported = ReportedUsage(input_tokens=11, output_tokens=9)
    first = estimate_cost(reported, lookup_prices("gpt-5-nano"))
    second = estimate_cost(reported, lookup_prices("claude-sonnet-4-6"))
    expected = combine_costs([first, second, judge])

    body = response.json()
    assert body["metrics"]["cost"] == "0.00017265" or body["metrics"]["cost"] == _money(expected.total)
    assert body["metrics"]["cost_completeness"] == "estimated"
    assert body["savings_basis"] == "same_token_volume"
    assert len(store.attempts) == 2
    assert [attempt.attempt_number for attempt in store.attempts] == [1, 2]
    assert store.final is not None
    assert store.final.total_cost == expected.total
    assert store.final.evaluations[0].judge_cost == judge.total
    assert store.final.escalated is True


def _money(value: Decimal | None) -> str | None:
    from app.pricing.money import money_to_api

    return money_to_api(value)
