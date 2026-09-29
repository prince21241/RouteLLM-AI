"""Compare the router with a baseline model on the same prompts.

Runs stay out of chat history. Paid provider calls happen only when the
caller passes ``mode="execute"``. Mock mode never constructs a provider
client from credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Literal

from app.api.schemas import LLMResponse, ModelConfig, Provider
from app.config import Settings, get_settings
from app.evaluation.dataset import EvalCase, EvaluationDataset, load_dataset, select_cases
from app.evaluation.escalation import select_escalation_target
from app.evaluation.grading import grade_case
from app.evaluation.judge import run_judge
from app.evaluation.live import apply_score_threshold
from app.evaluation.schema import EvaluationResult
from app.pricing.money import money_to_api, quantize_money
from app.pricing.service import (
    CostEstimate,
    combine_costs,
    estimate_cost,
    lookup_prices,
    usage_from_response,
)
from app.providers.base import LLMProvider
from app.providers.errors import ProviderError
from app.providers.factory import build_provider
from app.routing.catalog import build_catalog
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter, NoModelsAvailableError, parse_preference

Mode = Literal["mock", "execute"]
ProviderFactory = Callable[[ModelConfig], LLMProvider]


@dataclass
class SideRecord:
    """One arm of a case. Errors stay on the record."""

    model_id: str | None
    answer: str | None
    verdict: str | None
    score: float | None
    reasons: list[str]
    input_tokens: int | None
    output_tokens: int | None
    cost: str | None
    cost_completeness: str
    latency_ms: float | None
    error: str | None
    escalated: bool = False
    escalation_reason: str | None = None
    escalation_error: str | None = None
    initial_verdict: str | None = None


@dataclass
class CaseRecord:
    """Router and baseline results for one dataset case."""

    case_id: str
    split: str
    category: str
    method: str
    router: SideRecord
    baseline: SideRecord


@dataclass
class EvaluationReport:
    """Machine-readable run plus the numbers used in the text summary."""

    version: str
    mode: Mode
    baseline_model: str
    cases: list[CaseRecord] = field(default_factory=list)
    summary: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "mode": self.mode,
            "baseline_model": self.baseline_model,
            "savings_basis": "measured_actual_cost",
            "cases": [asdict(case) for case in self.cases],
            "summary": self.summary,
        }


async def run_evaluation(
    dataset: EvaluationDataset,
    *,
    mode: Mode,
    registry: ModelRegistry,
    router: ModelRouter,
    provider_factory: ProviderFactory,
    baseline_model_id: str,
    cases: list[EvalCase] | None = None,
    settings: Settings | None = None,
    escalate: bool = False,
    judge: Callable[..., Awaitable[LLMResponse]] | None = None,
    judge_model_id: str | None = None,
) -> EvaluationReport:
    """Run every selected case. A failed case remains in the report."""
    if mode not in ("mock", "execute"):
        raise ValueError("mode must be mock or execute")
    selected = list(dataset.cases if cases is None else cases)
    resolved_settings = settings
    records: list[CaseRecord] = []
    total = len(selected)
    for index, case in enumerate(selected, start=1):
        print(f"[{index}/{total}] {case.id}", file=sys.stderr, flush=True)
        records.append(
            await _run_case(
                case,
                registry=registry,
                router=router,
                provider_factory=provider_factory,
                baseline_model_id=baseline_model_id,
                settings=resolved_settings,
                escalate=escalate,
                judge=judge,
                judge_model_id=judge_model_id,
                minimum_score=0.75 if resolved_settings is None else resolved_settings.min_quality_score,
            )
        )
    report = EvaluationReport(
        version=dataset.version,
        mode=mode,
        baseline_model=baseline_model_id,
        cases=records,
    )
    report.summary = summarize(report)
    return report


def summarize(report: EvaluationReport) -> dict[str, object]:
    """Aggregate quality, cost, latency, and escalation without dropping cases."""
    total = len(report.cases)
    verdicts = [case.router.verdict for case in report.cases]
    passes = verdicts.count("pass")
    failures = [
        case.case_id
        for case in report.cases
        if case.router.error is not None or case.router.verdict in {"fail", "error"}
    ]
    unknowns = [case.case_id for case in report.cases if case.router.verdict == "unknown"]
    escalated = sum(1 for case in report.cases if case.router.escalated)
    router_cost, router_cost_known = _total_cost(case.router.cost for case in report.cases)
    baseline_cost, baseline_cost_known = _total_cost(case.baseline.cost for case in report.cases)
    difference: str | None
    if router_cost_known and baseline_cost_known and router_cost is not None and baseline_cost is not None:
        difference = money_to_api(quantize_money(Decimal(baseline_cost) - Decimal(router_cost)))
    else:
        difference = None
    router_latencies = [case.router.latency_ms for case in report.cases if case.router.latency_ms is not None]
    return {
        "cases": total,
        "router_passes": passes,
        "router_pass_rate": None if total == 0 else passes / total,
        "router_failures": failures,
        "router_unknowns": unknowns,
        "escalations": escalated,
        "escalation_rate": None if total == 0 else escalated / total,
        "router_cost": router_cost,
        "baseline_cost": baseline_cost,
        "measured_cost_difference": difference,
        "measured_cost_difference_basis": "baseline_actual_cost_minus_router_actual_cost",
        "router_mean_latency_ms": (
            None if not router_latencies else sum(router_latencies) / len(router_latencies)
        ),
        "router_latency_samples": len(router_latencies),
        "router_latency_missing": total - len(router_latencies),
    }


def render_summary(report: EvaluationReport) -> str:
    """Readable summary. Failure ids are listed, not omitted."""
    summary = report.summary
    rate = summary["router_pass_rate"]
    rate_text = "n/a" if rate is None else f"{rate:.3f}"
    escalation = summary["escalation_rate"]
    escalation_text = "n/a" if escalation is None else f"{escalation:.3f}"
    difference = summary["measured_cost_difference"]
    difference_text = "unknown" if difference is None else str(difference)
    failures = summary["router_failures"]
    failure_text = "(none)" if not failures else ", ".join(str(item) for item in failures)
    unknowns = summary["router_unknowns"]
    unknown_text = "(none)" if not unknowns else ", ".join(str(item) for item in unknowns)
    mean_latency = summary["router_mean_latency_ms"]
    latency_text = "n/a" if mean_latency is None else f"{mean_latency:.3f}"
    return "\n".join(
        [
            f"Dataset {report.version} mode={report.mode} baseline={report.baseline_model}",
            f"Cases: {summary['cases']}",
            f"Router pass rate: {rate_text} ({summary['router_passes']}/{summary['cases']})",
            f"Router failures: {failure_text}",
            f"Router unknowns: {unknown_text}",
            f"Escalation rate: {escalation_text}",
            f"Router cost: {summary['router_cost'] if summary['router_cost'] is not None else 'unknown'}",
            f"Baseline cost: {summary['baseline_cost'] if summary['baseline_cost'] is not None else 'unknown'}",
            f"Measured cost difference (baseline minus router): {difference_text}",
            (
                "Router mean latency ms: "
                f"{latency_text} ({summary['router_latency_samples']} samples, "
                f"{summary['router_latency_missing']} missing)"
            ),
        ]
    )


def reference_answer(case: EvalCase) -> str:
    """Render a deterministic mock answer from the reference, not from a model."""
    if case.method == "json_fields" and isinstance(case.expected, dict):
        return json.dumps(case.expected)
    if case.method == "rubric":
        return "Mock response for an open-ended rubric case."
    return str(case.expected)


async def _run_case(
    case: EvalCase,
    *,
    registry: ModelRegistry,
    router: ModelRouter,
    provider_factory: ProviderFactory,
    baseline_model_id: str,
    settings: Settings | None,
    escalate: bool,
    judge: Callable[..., Awaitable[LLMResponse]] | None,
    judge_model_id: str | None,
    minimum_score: float,
) -> CaseRecord:
    router_side = await _router_side(
        case,
        registry=registry,
        router=router,
        provider_factory=provider_factory,
        settings=settings,
        escalate=escalate,
        judge=judge,
        judge_model_id=judge_model_id,
        minimum_score=minimum_score,
    )
    baseline_side = await _baseline_side(
        case,
        registry=registry,
        provider_factory=provider_factory,
        baseline_model_id=baseline_model_id,
        judge=judge,
        judge_model_id=judge_model_id,
        minimum_score=minimum_score,
    )
    return CaseRecord(
        case_id=case.id,
        split=case.split,
        category=case.category,
        method=case.method,
        router=router_side,
        baseline=baseline_side,
    )


async def _router_side(
    case: EvalCase,
    *,
    registry: ModelRegistry,
    router: ModelRouter,
    provider_factory: ProviderFactory,
    settings: Settings | None,
    escalate: bool,
    judge: Callable[..., Awaitable[LLMResponse]] | None,
    judge_model_id: str | None,
    minimum_score: float,
) -> SideRecord:
    try:
        from app.routing.complexity import ComplexityClassifier

        classifier = ComplexityClassifier()
        thresholds = settings
        low = 0.30 if thresholds is None else thresholds.low_complexity_threshold
        high = 0.70 if thresholds is None else thresholds.high_complexity_threshold
        assessment = classifier.classify(case.prompt, case.system_prompt, low=low, high=high)
        decision = router.select(assessment.tier, registry)
    except NoModelsAvailableError:
        return _empty_side(error="No routing models are available")
    generated, error = await _generate(provider_factory, decision.model, case)
    if generated is None:
        return _empty_side(model_id=decision.model.model_id, error=error)
    quality, judge_estimate, judge_latency = await _quality(
        case,
        generated.content,
        judge=judge,
        judge_model_id=judge_model_id,
        minimum_score=minimum_score,
    )
    parts = [_price(generated, decision.model.model_id)]
    latency = generated.latency_ms
    answer = generated.content
    model_id = decision.model.model_id
    tokens_in = generated.input_tokens
    tokens_out = generated.output_tokens
    escalated = False
    escalation_reason = None
    escalation_error = None
    initial_verdict = quality.verdict
    if escalate and quality.verdict == "fail" and settings is not None:
        target = select_escalation_target(decision.model, registry, settings)
        escalation_reason = f"initial answer failed quality; {target.reason}"
        if target.model is None:
            escalation_error = target.reason
        else:
            second, second_error = await _generate(provider_factory, target.model, case)
            if second is None:
                escalation_error = second_error
            else:
                escalated = True
                answer = second.content
                model_id = target.model.model_id
                tokens_in += second.input_tokens
                tokens_out += second.output_tokens
                latency += second.latency_ms
                parts.append(_price(second, target.model.model_id))
                quality, second_judge, second_latency = await _quality(
                    case,
                    second.content,
                    judge=judge,
                    judge_model_id=judge_model_id,
                    minimum_score=minimum_score,
                )
                if second_judge is not None:
                    judge_estimate = _merge_optional(judge_estimate, second_judge)
                if second_latency is not None:
                    judge_latency = (judge_latency or 0) + second_latency
    if judge_estimate is not None:
        parts.append(judge_estimate)
    if judge_latency is not None:
        latency += judge_latency
    priced = combine_costs(parts)
    return SideRecord(
        model_id=model_id,
        answer=answer,
        verdict=quality.verdict,
        score=quality.score,
        reasons=list(quality.reasons),
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        cost=money_to_api(priced.total),
        cost_completeness=priced.completeness.value,
        latency_ms=latency,
        error=None,
        escalated=escalated,
        escalation_reason=escalation_reason,
        escalation_error=escalation_error,
        initial_verdict=initial_verdict,
    )


async def _baseline_side(
    case: EvalCase,
    *,
    registry: ModelRegistry,
    provider_factory: ProviderFactory,
    baseline_model_id: str,
    judge: Callable[..., Awaitable[LLMResponse]] | None,
    judge_model_id: str | None,
    minimum_score: float,
) -> SideRecord:
    from app.routing.model_registry import UnknownModelError

    try:
        model = registry.get(baseline_model_id)
    except UnknownModelError:
        return _empty_side(model_id=baseline_model_id, error="The baseline model is not in the registry")
    generated, error = await _generate(provider_factory, model, case)
    if generated is None:
        return _empty_side(model_id=model.model_id, error=error)
    quality, judge_estimate, judge_latency = await _quality(
        case,
        generated.content,
        judge=judge,
        judge_model_id=judge_model_id,
        minimum_score=minimum_score,
    )
    parts = [_price(generated, model.model_id)]
    latency = generated.latency_ms
    if judge_estimate is not None:
        parts.append(judge_estimate)
    if judge_latency is not None:
        latency += judge_latency
    priced = combine_costs(parts)
    return SideRecord(
        model_id=model.model_id,
        answer=generated.content,
        verdict=quality.verdict,
        score=quality.score,
        reasons=list(quality.reasons),
        input_tokens=generated.input_tokens,
        output_tokens=generated.output_tokens,
        cost=money_to_api(priced.total),
        cost_completeness=priced.completeness.value,
        latency_ms=latency,
        error=None,
        initial_verdict=quality.verdict,
    )


async def _quality(
    case: EvalCase,
    answer: str,
    *,
    judge: Callable[..., Awaitable[LLMResponse]] | None,
    judge_model_id: str | None,
    minimum_score: float,
) -> tuple[EvaluationResult, CostEstimate | None, float | None]:
    if case.method != "rubric" or judge is None or judge_model_id is None:
        return grade_case(case, answer), None, None
    result = await run_judge(
        judge,
        model_id=judge_model_id,
        prompt=case.prompt,
        answer=answer,
        rubric=case.criteria or "",
    )
    return apply_score_threshold(result, minimum_score), result.judge_cost, result.judge_latency_ms


async def _generate(
    provider_factory: ProviderFactory,
    model: ModelConfig,
    case: EvalCase,
) -> tuple[LLMResponse | None, str | None]:
    provider: LLMProvider | None = None
    try:
        provider = provider_factory(model)
        generated = await provider.generate(case.prompt, case.system_prompt)
    except ProviderError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, type(exc).__name__
    else:
        return generated, None
    finally:
        await _close_provider(provider)


async def _close_provider(provider: LLMProvider | None) -> None:
    """Close a provider client.

    A task that is already cancelling must not await close. That await is
    cancelled immediately and replaces the original interruption with a
    second traceback.
    """
    if provider is None:
        return
    task = asyncio.current_task()
    if task is not None and task.cancelling():
        return
    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()


def _price(generated: LLMResponse, model_id: str) -> CostEstimate:
    return estimate_cost(usage_from_response(generated), lookup_prices(model_id))


def _merge_optional(first: CostEstimate | None, second: CostEstimate | None) -> CostEstimate | None:
    parts = [item for item in (first, second) if item is not None]
    if not parts:
        return None
    return combine_costs(parts)


def _empty_side(*, model_id: str | None = None, error: str) -> SideRecord:
    return SideRecord(
        model_id=model_id,
        answer=None,
        verdict=None,
        score=None,
        reasons=[],
        input_tokens=None,
        output_tokens=None,
        cost=None,
        cost_completeness="unknown",
        latency_ms=None,
        error=error,
    )


def _total_cost(values: object) -> tuple[str | None, bool]:
    amounts: list[Decimal] = []
    for value in values:
        if value is None:
            return None, False
        amounts.append(Decimal(str(value)))
    if not amounts:
        return None, False
    return money_to_api(quantize_money(sum(amounts, Decimal("0")))), True


def _mock_judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
    """Offline rubric verdict. This is not a measured quality score."""
    del prompt, system_prompt
    return LLMResponse(
        provider=Provider.OPENAI,
        model="mock-judge",
        content=json.dumps(
            {
                "verdict": "fail",
                "score": 0,
                "reasons": [
                    "Mock mode does not call a model judge. The placeholder answer was not accepted."
                ],
            }
        ),
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
    )


async def _mock_judge_generate(prompt: str, system_prompt: str | None = None) -> LLMResponse:
    return _mock_judge(prompt, system_prompt)


def _mock_factory(cases: list[EvalCase]) -> ProviderFactory:
    by_prompt = {case.prompt: case for case in cases}

    def factory(model: ModelConfig) -> LLMProvider:
        class _MockProvider:
            async def generate(
                self,
                prompt: str,
                system_prompt: str | None = None,
            ) -> LLMResponse:
                case = by_prompt.get(prompt)
                content = reference_answer(case) if case is not None else ""
                return LLMResponse(
                    provider=model.provider,
                    model=model.model_id,
                    content=content,
                    input_tokens=8,
                    output_tokens=4,
                    latency_ms=1.0,
                )

            async def aclose(self) -> None:
                return None

        return _MockProvider()

    return factory


def main(argv: list[str] | None = None) -> int:
    """CLI entry. ``--execute`` is the only path that calls paid APIs."""
    parser = argparse.ArgumentParser(
        description="Compare the router with a baseline model. Paid calls require --execute."
    )
    parser.add_argument("--mock", action="store_true", help="Grade reference answers. No provider calls.")
    parser.add_argument("--execute", action="store_true", help="Call configured providers. This can cost money.")
    parser.add_argument("--smoke", action="store_true", help="Run the small smoke subset.")
    parser.add_argument("--split", choices=["calibration", "held_out"])
    parser.add_argument("--escalate", action="store_true", help="Allow one escalation after a quality failure.")
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--output", default=None, help="Write the JSON report to this path.")
    args = parser.parse_args(argv)
    if args.mock == args.execute:
        print(
            "Choose exactly one of --mock or --execute. --execute sends prompts to paid provider APIs.",
            file=sys.stderr,
        )
        return 2
    settings = get_settings()
    dataset = load_dataset()
    cases = select_cases(dataset, split=args.split, smoke=args.smoke)
    registry = build_catalog(settings)
    router = ModelRouter(parse_preference(settings.routing_preference))
    baseline_model = args.baseline_model or settings.evaluation_baseline_model
    if args.mock:
        factory: ProviderFactory = _mock_factory(cases)
        mode: Mode = "mock"
        judge = _mock_judge_generate
        judge_model_id = "mock-judge"
    else:
        factory = lambda model: build_provider(model, settings)
        mode = "execute"
        judge = None
        judge_model_id = None
    try:
        report = asyncio.run(
            run_evaluation(
                dataset,
                mode=mode,
                cases=cases,
                registry=registry,
                router=router,
                provider_factory=factory,
                baseline_model_id=baseline_model,
                settings=settings,
                escalate=args.escalate,
                judge=judge,
                judge_model_id=judge_model_id,
            )
        )
    except KeyboardInterrupt:
        print("Evaluation cancelled.", file=sys.stderr)
        return 130
    print(render_summary(report))
    payload = json.dumps(report.to_dict(), indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
