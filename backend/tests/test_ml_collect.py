"""Collection mechanics. Providers are in-memory and are not training evidence."""

import asyncio
from decimal import Decimal

import pytest

from app.api.schemas import LLMResponse, ModelConfig, Provider, QualityTier, ReportedUsage
from app.config import Settings
from app.evaluation.dataset import FAMILIES_PATH, EvalCase, load_dataset
from app.ml.collect import (
    collection_audit_path,
    collect_measured,
    default_collection_directory,
    estimate_collection,
    load_collection_audit,
    render_estimate,
    repository_root,
    run_collect,
)
from app.ml.schema import load_training_dataset
from app.pricing.service import estimate_cost, lookup_prices, usage_from_response
from app.providers.errors import ProviderResponseError, ProviderTimeoutError, ProviderUpstreamError


def _case() -> EvalCase:
    return EvalCase(
        id="fact-water",
        split="calibration",
        category="factual",
        method="exact",
        prompt="What is the chemical symbol for water? Reply with only the symbol.",
        expected="H2O",
        group_id="water-family",
    )


def _models() -> list[ModelConfig]:
    return [
        ModelConfig(
            provider=Provider.OPENAI,
            model_name="GPT-5 nano",
            model_id="gpt-5-nano",
            quality_tier=QualityTier.LOW,
            input_cost_per_million_tokens=Decimal("0.05"),
            output_cost_per_million_tokens=Decimal("0.40"),
            context_window=1024,
            enabled=True,
            local=False,
        ),
        ModelConfig(
            provider=Provider.ANTHROPIC,
            model_name="Claude Sonnet 4.6",
            model_id="claude-sonnet-4-6",
            quality_tier=QualityTier.HIGH,
            input_cost_per_million_tokens=Decimal("3"),
            output_cost_per_million_tokens=Decimal("15"),
            context_window=1024,
            enabled=True,
            local=False,
        ),
    ]


def _factory(calls: list[str], *, fail: str | None = None):
    def factory(model: ModelConfig):
        class _Provider:
            async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
                calls.append(model.model_id)
                if model.model_id == fail:
                    raise ProviderUpstreamError("upstream")
                content = "H2O" if model.model_id == "gpt-5-nano" else "water"
                return LLMResponse(
                    provider=model.provider,
                    model=model.model_id,
                    content=content,
                    input_tokens=11,
                    output_tokens=9,
                    latency_ms=12.5,
                )

            async def aclose(self) -> None:
                return None

        return _Provider()

    return factory


def test_estimate_does_not_call_providers() -> None:
    calls: list[str] = []
    settings = Settings(_env_file=None)
    estimate = estimate_collection([_case()], _models(), settings, judge=False)
    text = render_estimate(estimate)
    assert calls == []
    assert estimate.prompts == 1
    assert estimate.generation_ceiling > 0
    assert estimate.judge_ceiling == 0
    assert "approximate" in text.casefold()
    assert "character-based spending estimates are approximate" in text.casefold()
    assert run_collect(["--estimate", "--smoke"]) == 0


def test_families_estimate_covers_forty_eight_groups() -> None:
    dataset = load_dataset(FAMILIES_PATH)
    estimate = estimate_collection(
        list(dataset.cases),
        _models(),
        Settings(_env_file=None),
        judge=True,
    )
    assert estimate.prompts == 96
    assert estimate.families == 48
    assert estimate.rubric_cases == 2
    assert estimate.exploratory_even_if_fully_labeled is False
    assert estimate.judge_ceiling is not None and estimate.judge_ceiling > 0


def test_recorded_outcome_uses_provider_usage_and_family(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"
    result = asyncio.run(
        collect_measured(
            [_case()],
            _models(),
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.1",
        )
    )
    dataset = load_training_dataset(output)
    assert result.provider_calls == 2
    assert dataset.provenance == "live"
    assert dataset.evidence == "measured"
    prompt = dataset.prompts[0]
    assert prompt.group_id == "water-family"
    by_model = {item.model_id: item for item in prompt.outcomes}
    assert by_model["gpt-5-nano"].quality_verdict == "pass"
    assert by_model["claude-sonnet-4-6"].quality_verdict == "fail"
    assert by_model["gpt-5-nano"].cost == "0.00000415"
    assert by_model["gpt-5-nano"].latency_ms == 12.5
    assert by_model["gpt-5-nano"].status == "succeeded"


def test_spending_cap_saves_a_partial_file_and_resume_skips_it(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"
    settings = Settings(_env_file=None)
    first = asyncio.run(
        collect_measured(
            [_case()],
            _models(),
            provider_factory=_factory(calls),
            settings=settings,
            output=output,
            max_spend=Decimal("0.001"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.1",
        )
    )
    partial = load_training_dataset(output)
    assert first.stopped_reason == "spending cap reached before the next generation"
    assert [item.model_id for item in partial.prompts[0].outcomes] == ["gpt-5-nano"]
    second = asyncio.run(
        collect_measured(
            [_case()],
            _models(),
            provider_factory=_factory(calls),
            settings=settings,
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.1",
        )
    )
    finished = load_training_dataset(output)
    assert second.skipped_completed == 1
    assert calls == ["gpt-5-nano", "claude-sonnet-4-6"]
    assert {item.model_id for item in finished.prompts[0].outcomes} == {
        "gpt-5-nano",
        "claude-sonnet-4-6",
    }


def test_provider_failure_is_not_a_quality_grade(tmp_path) -> None:
    output = tmp_path / "training.json"
    asyncio.run(
        collect_measured(
            [_case()],
            _models()[:1],
            provider_factory=_factory([], fail="gpt-5-nano"),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.1",
        )
    )
    outcome = load_training_dataset(output).prompts[0].outcomes[0]
    assert outcome.status == "failed"
    assert outcome.quality_verdict is None
    assert outcome.cost is None
    assert outcome.error_category == "transient_server"


def test_smoke_dataset_still_loads_without_group_ids() -> None:
    dataset = load_dataset()
    assert dataset.version == "2026-09-29.1"
    assert all(case.group_id is None for case in dataset.cases)
    assert len(dataset.cases) == 30


def _rubric() -> EvalCase:
    return EvalCase(
        id="reason-ice-1",
        split="held_out",
        category="reasoning",
        method="rubric",
        prompt="Why does ice float on liquid water? Answer in two sentences.",
        criteria="Pass only if the answer states that ice is less dense than liquid water.",
        group_id="reason-ice",
    )


def _judge_response() -> LLMResponse:
    return LLMResponse(
        provider=Provider.OPENAI,
        model="gpt-5-nano",
        content='{"verdict": "pass", "score": 0.91, "reasons": ["density"]}',
        input_tokens=40,
        output_tokens=20,
        latency_ms=8.0,
    )


def test_judge_usage_is_audited_separately_and_counted_on_resume(tmp_path) -> None:
    calls: list[str] = []
    judge_calls: list[str] = []
    output = tmp_path / "training.json"

    async def judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        judge_calls.append("judge")
        return _judge_response()

    result = asyncio.run(
        collect_measured(
            [_rubric()],
            _models()[:1],
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    dataset = load_training_dataset(output)
    outcome = dataset.prompts[0].outcomes[0]
    audit = load_collection_audit(collection_audit_path(output))
    generation = estimate_cost(
        usage_from_response(
            LLMResponse(
                provider=Provider.OPENAI,
                model="gpt-5-nano",
                content="H2O",
                input_tokens=11,
                output_tokens=9,
                latency_ms=12.5,
            )
        ),
        lookup_prices("gpt-5-nano"),
    )
    judged = estimate_cost(usage_from_response(_judge_response()), lookup_prices("gpt-5-nano"))
    assert generation.total is not None and judged.total is not None
    assert outcome.cost is not None and audit.judge_calls[0].cost is not None
    assert Decimal(outcome.cost) == generation.total
    assert outcome.quality_verdict == "pass"
    assert audit.judge_calls[0].input_tokens == 40
    assert audit.judge_calls[0].output_tokens == 20
    assert Decimal(audit.judge_calls[0].cost) == judged.total
    assert Decimal(outcome.cost) + Decimal(audit.judge_calls[0].cost) == result.spent
    assert judge_calls == ["judge"]
    resumed = asyncio.run(
        collect_measured(
            [_rubric()],
            _models()[:1],
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    assert resumed.provider_calls == 0
    assert resumed.judge_calls == 0
    assert resumed.skipped_completed == 1
    assert resumed.spent == result.spent
    assert judge_calls == ["judge"]


def test_unknown_incurred_cost_stops_without_a_zero(tmp_path) -> None:
    calls: list[str] = []

    def factory(model: ModelConfig):
        class _Provider:
            async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
                calls.append(model.model_id)
                return LLMResponse(
                    provider=model.provider,
                    model=model.model_id,
                    content="dense",
                    input_tokens=11,
                    output_tokens=9,
                    cached_input_tokens=50,
                    latency_ms=3.0,
                )

            async def aclose(self) -> None:
                return None

        return _Provider()

    output = tmp_path / "training.json"
    result = asyncio.run(
        collect_measured(
            [_case()],
            _models(),
            provider_factory=factory,
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.1",
        )
    )
    outcome = load_training_dataset(output).prompts[0].outcomes[0]
    assert result.stopped_reason == "incurred generation cost is unknown"
    assert result.spent == 0
    assert calls == ["gpt-5-nano"]
    assert outcome.cost is None
    assert outcome.cost_completeness == "unknown"
    resumed_calls: list[str] = []
    resumed = asyncio.run(
        collect_measured(
            [_case()],
            _models(),
            provider_factory=_factory(resumed_calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.1",
        )
    )
    assert resumed_calls == []
    assert resumed.stopped_reason == "incurred generation cost is unknown"


def test_resume_rejects_a_different_dataset_candidates_or_judge(tmp_path) -> None:
    output = tmp_path / "training.json"
    kwargs = dict(
        provider_factory=_factory([]),
        settings=Settings(_env_file=None),
        output=output,
        max_spend=Decimal("10"),
        stronger_model_id="claude-sonnet-4-6",
        quality_threshold=0.75,
        dataset_version="2026-09-29.1",
    )
    asyncio.run(collect_measured([_case()], _models(), **kwargs))
    with pytest.raises(ValueError, match="dataset"):
        asyncio.run(
            collect_measured(
                [_case()],
                _models(),
                **{**kwargs, "dataset_version": "other"},
            )
        )
    with pytest.raises(ValueError, match="candidate"):
        asyncio.run(
            collect_measured(
                [_case()],
                _models()[:1],
                provider_factory=_factory([]),
                settings=Settings(_env_file=None),
                output=output,
                max_spend=Decimal("10"),
                stronger_model_id="gpt-5-nano",
                quality_threshold=0.75,
                dataset_version="2026-09-29.1",
            )
        )
    with pytest.raises(ValueError, match="judge"):
        asyncio.run(
            collect_measured(
                [_case()],
                _models(),
                provider_factory=_factory([]),
                settings=Settings(_env_file=None),
                output=output,
                max_spend=Decimal("10"),
                stronger_model_id="claude-sonnet-4-6",
                quality_threshold=0.75,
                dataset_version="2026-09-29.1",
                judge_generate=lambda prompt, system_prompt=None: None,
                judge_model_id="gpt-5-nano",
            )
        )


def test_resume_judges_a_retained_answer_without_repeating_generation(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"

    def factory(model: ModelConfig):
        class _Provider:
            async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
                calls.append(model.model_id)
                return LLMResponse(
                    provider=model.provider,
                    model=model.model_id,
                    content="Ice is less dense than water.",
                    input_tokens=400_000,
                    output_tokens=9,
                    latency_ms=4.0,
                )

            async def aclose(self) -> None:
                return None

        return _Provider()

    async def judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        calls.append("judge")
        return _judge_response()

    first = asyncio.run(
        collect_measured(
            [_rubric()],
            _models()[:1],
            provider_factory=factory,
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("0.01"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    assert first.stopped_reason == "spending cap reached before the judge call"
    assert calls == ["gpt-5-nano"]
    generation_cost = load_training_dataset(output).prompts[0].outcomes[0].cost
    assert load_training_dataset(output).prompts[0].outcomes[0].quality_verdict == "unknown"
    assert load_collection_audit(collection_audit_path(output)).pending_answers
    second = asyncio.run(
        collect_measured(
            [_rubric()],
            _models()[:1],
            provider_factory=factory,
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    finished = load_training_dataset(output)
    audit = load_collection_audit(collection_audit_path(output))
    assert second.provider_calls == 0
    assert second.judge_calls == 1
    assert calls == ["gpt-5-nano", "judge"]
    assert finished.prompts[0].outcomes[0].quality_verdict == "pass"
    assert finished.prompts[0].outcomes[0].cost == generation_cost
    assert audit.pending_answers == []
    assert audit.judge_calls[0].cost is not None
    assert generation_cost is not None
    assert Decimal(generation_cost) < second.spent


def test_judge_failure_without_usage_stays_unknown_on_resume(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"

    async def judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        raise ProviderTimeoutError("OpenAI request timed out")

    first = asyncio.run(
        collect_measured(
            [_rubric()],
            _models()[:1],
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    before = load_training_dataset(output)
    audit = load_collection_audit(collection_audit_path(output))
    assert first.stopped_reason == "incurred judge cost is unknown"
    assert audit.judge_calls[0].cost is None
    assert audit.judge_calls[0].input_tokens is None
    assert audit.judge_calls[0].error_category == "timeout"
    assert before.prompts[0].outcomes[0].cost is not None
    assert first.spent == Decimal(before.prompts[0].outcomes[0].cost)
    resumed_calls: list[str] = []
    second = asyncio.run(
        collect_measured(
            [_rubric()],
            _models()[:1],
            provider_factory=_factory(resumed_calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    after = load_training_dataset(output)
    audit_after = load_collection_audit(collection_audit_path(output))
    assert resumed_calls == []
    assert second.provider_calls == 0
    assert second.judge_calls == 0
    assert second.spent == first.spent
    assert after.model_dump() == before.model_dump()
    assert audit_after.judge_calls[0].cost is None


def test_reservation_is_separate_and_skips_completed_work(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"

    async def judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        raise ProviderTimeoutError("OpenAI request timed out")

    first = asyncio.run(
        collect_measured(
            [_rubric()],
            _models(),
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    assert first.stopped_reason == "incurred judge cost is unknown"
    assert calls == ["gpt-5-nano"]
    resumed_calls: list[str] = []
    second = asyncio.run(
        collect_measured(
            [_rubric()],
            _models(),
            provider_factory=_factory(resumed_calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=first.spent + Decimal("0.0208192"),
            stronger_model_id="claude-sonnet-4-6",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
            reserve_unknown_judge=True,
        )
    )
    dataset = load_training_dataset(output)
    audit = load_collection_audit(collection_audit_path(output))
    outcome = dataset.prompts[0].outcomes[0]
    assert resumed_calls == []
    assert second.provider_calls == 0
    assert second.judge_calls == 0
    assert second.stopped_reason == "spending cap reached before the next generation"
    assert outcome.quality_verdict == "error"
    assert outcome.cost is not None and Decimal(outcome.cost) == first.spent
    assert audit.judge_calls[0].cost is None
    assert audit.judge_calls[0].input_tokens is None
    assert audit.judge_calls[0].cost_completeness == "unknown"
    assert audit.budget_reservations[0].measured is False
    assert audit.budget_reservations[0].amount == "0.0208192"
    assert audit.budget_reservations[0].context_window_tokens == 400_000
    assert audit.budget_reservations[0].output_token_cap == 2048
    assert second.reserved == Decimal("0.0208192")
    assert second.spent == first.spent


def test_reserved_judge_failure_continues_and_records_the_error_type(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"

    async def judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        raise ProviderTimeoutError("OpenAI request timed out")

    result = asyncio.run(
        collect_measured(
            [_rubric(), _case()],
            _models()[:1],
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
            reserve_unknown_judge=True,
        )
    )
    audit = load_collection_audit(collection_audit_path(output))
    dataset = load_training_dataset(output)
    assert result.stopped_reason is None
    assert calls == ["gpt-5-nano", "gpt-5-nano"]
    assert [item.prompt_id for item in dataset.prompts] == ["reason-ice-1", "fact-water"]
    assert audit.judge_calls[0].cost is None
    assert audit.judge_calls[0].error_type == "ProviderTimeoutError"
    assert audit.budget_reservations[0].amount == "0.0208192"
    assert audit.budget_reservations[0].output_token_cap == 2048
    assert result.reserved == Decimal("0.0208192")


def test_priced_judge_failure_does_not_stop_the_collection(tmp_path) -> None:
    calls: list[str] = []
    output = tmp_path / "training.json"
    usage = ReportedUsage(input_tokens=120, output_tokens=2048)

    async def judge(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        raise ProviderResponseError(
            "OpenAI response was incomplete (max_output_tokens)",
            usage=usage,
        )

    result = asyncio.run(
        collect_measured(
            [_rubric(), _case()],
            _models()[:1],
            provider_factory=_factory(calls),
            settings=Settings(_env_file=None),
            output=output,
            max_spend=Decimal("10"),
            stronger_model_id="gpt-5-nano",
            quality_threshold=0.75,
            dataset_version="2026-09-29.families",
            judge_generate=judge,
            judge_model_id="gpt-5-nano",
        )
    )
    dataset = load_training_dataset(output)
    audit = load_collection_audit(collection_audit_path(output))
    priced = estimate_cost(usage, lookup_prices("gpt-5-nano"))
    assert result.stopped_reason is None
    assert [prompt.prompt_id for prompt in dataset.prompts] == ["reason-ice-1", "fact-water"]
    assert calls == ["gpt-5-nano", "gpt-5-nano"]
    assert dataset.prompts[0].outcomes[0].quality_verdict == "error"
    assert audit.judge_calls[0].input_tokens == 120
    assert audit.judge_calls[0].output_tokens == 2048
    assert audit.judge_calls[0].error_category == "invalid_request"
    assert priced.total is not None and audit.judge_calls[0].cost is not None
    assert Decimal(audit.judge_calls[0].cost) == priced.total
    assert result.spent > Decimal(dataset.prompts[0].outcomes[0].cost or "0")


def test_collection_output_stays_outside_the_repository(tmp_path) -> None:
    directory = default_collection_directory()
    root = repository_root()
    assert directory != root and root not in directory.parents
    with pytest.raises(ValueError, match="outside the repository"):
        asyncio.run(
            collect_measured(
                [_case()],
                _models()[:1],
                provider_factory=_factory([]),
                settings=Settings(_env_file=None),
                output=root / "training.json",
                max_spend=Decimal("10"),
                stronger_model_id="gpt-5-nano",
                quality_threshold=0.75,
                dataset_version="2026-09-29.1",
            )
        )
