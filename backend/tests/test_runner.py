"""Baseline runner aggregation. Providers are in-memory."""

import asyncio
from decimal import Decimal

import pytest

from app.api.schemas import LLMResponse, ModelConfig, Provider, QualityTier
from app.config import Settings
from app.evaluation.dataset import load_dataset, select_cases
from app.evaluation.runner import main, reference_answer, run_evaluation, _generate, _mock_judge_generate
from app.providers.errors import ProviderUpstreamError
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter


def _model(model_id: str, tier: QualityTier) -> ModelConfig:
    return ModelConfig(
        provider=Provider.OPENAI,
        model_name=model_id,
        model_id=model_id,
        quality_tier=tier,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        context_window=1024,
        enabled=True,
        local=False,
    )


def _factory(wrong_prompt: str | None = None, fail: bool = False):
    def factory(model: ModelConfig):
        class _Provider:
            async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
                if fail:
                    raise ProviderUpstreamError("OpenAI upstream error (HTTP 500)")
                content = "0" if prompt == wrong_prompt else _answer_for(prompt)
                return LLMResponse(
                    provider=model.provider,
                    model=model.model_id,
                    content=content,
                    input_tokens=8,
                    output_tokens=4,
                    latency_ms=5,
                )

            async def aclose(self) -> None:
                return None

        return _Provider()

    return factory


def _answer_for(prompt: str) -> str:
    dataset = load_dataset()
    case = next(item for item in dataset.cases if item.prompt == prompt)
    return reference_answer(case)


def test_failures_stay_in_the_report() -> None:
    dataset = load_dataset()
    cases = select_cases(dataset, smoke=True)
    wrong = next(case.prompt for case in cases if case.method == "numeric")
    registry = ModelRegistry([_model("gpt-5-nano", QualityTier.LOW)])

    report = asyncio.run(run_evaluation(
        dataset,
        mode="mock",
        cases=cases,
        registry=registry,
        router=ModelRouter(),
        provider_factory=_factory(wrong_prompt=wrong),
        baseline_model_id="gpt-5-nano",
        settings=Settings(_env_file=None),
    ))

    assert len(report.cases) == len(cases)
    assert report.summary["cases"] == len(cases)
    assert report.summary["router_failures"]
    assert report.summary["router_pass_rate"] == report.summary["router_passes"] / len(cases)
    failed = next(case for case in report.cases if case.router.verdict == "fail")
    assert failed.case_id in report.summary["router_failures"]
    assert failed.router.answer == "0"


def test_provider_errors_are_not_dropped() -> None:
    dataset = load_dataset()
    cases = select_cases(dataset, smoke=True)
    registry = ModelRegistry([_model("gpt-5-nano", QualityTier.LOW)])

    report = asyncio.run(run_evaluation(
        dataset,
        mode="mock",
        cases=cases,
        registry=registry,
        router=ModelRouter(),
        provider_factory=_factory(fail=True),
        baseline_model_id="gpt-5-nano",
        settings=Settings(_env_file=None),
    ))

    assert len(report.cases) == len(cases)
    assert all(case.router.error for case in report.cases)
    assert report.summary["router_passes"] == 0
    assert set(report.summary["router_failures"]) == {case.id for case in cases}


def test_unknown_costs_stay_unknown() -> None:
    dataset = load_dataset()
    cases = select_cases(dataset, smoke=True)
    registry = ModelRegistry([_model("fictional-low", QualityTier.LOW)])

    report = asyncio.run(run_evaluation(
        dataset,
        mode="mock",
        cases=cases,
        registry=registry,
        router=ModelRouter(),
        provider_factory=_factory(),
        baseline_model_id="fictional-low",
        settings=Settings(_env_file=None),
    ))

    assert report.summary["router_cost"] is None
    assert report.summary["baseline_cost"] is None
    assert report.summary["measured_cost_difference"] is None
    assert report.to_dict()["savings_basis"] == "measured_actual_cost"


def test_mock_rubric_cases_fail_instead_of_staying_unjudged() -> None:
    dataset = load_dataset()
    case = next(item for item in dataset.cases if item.method == "rubric")
    registry = ModelRegistry([_model("gpt-5-nano", QualityTier.LOW)])

    report = asyncio.run(
        run_evaluation(
            dataset,
            mode="mock",
            cases=[case],
            registry=registry,
            router=ModelRouter(),
            provider_factory=_factory(),
            baseline_model_id="gpt-5-nano",
            settings=Settings(_env_file=None),
            judge=_mock_judge_generate,
            judge_model_id="mock-judge",
        )
    )

    record = report.cases[0]
    assert record.router.verdict == "fail"
    assert record.baseline.verdict == "fail"
    assert record.router.cost == "0.000002"
    assert report.summary["router_unknowns"] == []
    assert case.id in report.summary["router_failures"]


def test_paid_execution_must_be_explicit() -> None:
    assert main([]) == 2
    assert main(["--mock", "--execute", "--smoke"]) == 2


def test_console_entry_reports_cancellation(monkeypatch, capsys) -> None:
    def boom(argv: list[str] | None = None) -> int:
        del argv
        raise KeyboardInterrupt

    monkeypatch.setattr("app.evaluation.__main__.main", boom)
    from app.evaluation.__main__ import console_main

    assert console_main(["--mock", "--smoke"]) == 130
    captured = capsys.readouterr()
    assert captured.err == "Evaluation cancelled.\n"
    assert captured.out == ""


def test_keyboard_interrupt_exits_without_a_traceback(monkeypatch, capsys) -> None:
    def raise_interrupt(coro, *_args, **_kwargs):
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr("app.evaluation.runner.asyncio.run", raise_interrupt)

    assert main(["--mock", "--smoke"]) == 130
    captured = capsys.readouterr()
    assert captured.err == "Evaluation cancelled.\n"
    assert "Traceback" not in captured.out


def test_cancelled_generation_does_not_close_the_provider() -> None:
    dataset = load_dataset()
    case = select_cases(dataset, smoke=True)[0]
    model = _model("gpt-5-nano", QualityTier.LOW)
    started = asyncio.Event()

    class _Provider:
        def __init__(self) -> None:
            self.closed = False

        async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
            del prompt, system_prompt
            started.set()
            await asyncio.Future()
            raise AssertionError("cancelled generation returned")

        async def aclose(self) -> None:
            self.closed = True

    provider = _Provider()

    async def scenario() -> bool:
        task = asyncio.create_task(_generate(lambda _model_config: provider, model, case))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return provider.closed

    assert asyncio.run(scenario()) is False
