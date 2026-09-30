"""Collect comparable live outcomes for ML training.

The evaluation runner compares the router with one baseline. This collector
calls each candidate model on the same prompt and writes the training file
``app.ml validate`` accepts. Generation cost stays on the candidate outcome.
Judge token usage and cost are written to a sibling audit file and count
toward the spending cap. They are not added to the candidate's training cost.

A character-length ceiling is an approximate allowance, not a measured
invoice. An incurred cost that cannot be priced stops collection. It is not
replaced with zero.

Mock answers are not accepted. This module does not invent token counts,
verdicts, or prices.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.api.schemas import ModelConfig, Provider
from app.config import Settings
from app.evaluation.dataset import EvalCase
from app.evaluation.grading import grade_case
from app.evaluation.judge import run_judge
from app.evaluation.live import apply_score_threshold
from app.evaluation.runner import ProviderFactory, _price
from app.ml.schema import (
    CandidateModel,
    GenerationSettings,
    ModelOutcome,
    PromptRecord,
    TrainingDataset,
    load_training_dataset,
)
from app.ml.split import EXPLORATORY_LABELED_COUNT
from app.pricing.money import line_cost, money_to_api, quantize_money
from app.pricing.service import CostCompleteness, CostEstimate, lookup_prices
from app.providers.errors import ProviderError
from app.providers.failure import classify_provider_error

PROPOSED_FAMILY_COUNT = 48
_JUDGE_INPUT_ALLOWANCE = 800
_AUDIT_FORMAT = "routellm-collection-audit-1"
_UNKNOWN_GENERATION = "incurred generation cost is unknown"
_UNKNOWN_JUDGE = "incurred judge cost is unknown"


class JudgeCallRecord(BaseModel):
    """One judge call. This is audit data, not a training feature."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    candidate_model_id: str = Field(min_length=1)
    judge_model_id: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_write_5m_tokens: int | None = Field(default=None, ge=0)
    cache_write_1h_tokens: int | None = Field(default=None, ge=0)
    cost: str | None = None
    cost_completeness: Literal["complete", "estimated", "unknown"]
    latency_ms: float | None = Field(default=None, ge=0)
    error_category: str | None = None
    error_type: str | None = None


class BudgetReservation(BaseModel):
    """A budget hold for an unknown incurred cost. This is not measured spend."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    candidate_model_id: str = Field(min_length=1)
    judge_model_id: str = Field(min_length=1)
    amount: str = Field(min_length=1)
    basis: Literal["context_window_and_output_cap"]
    context_window_tokens: int = Field(ge=1)
    output_token_cap: int = Field(ge=1)
    input_price_per_million: str = Field(min_length=1)
    output_price_per_million: str = Field(min_length=1)
    measured: Literal[False] = False


class PendingJudgeAnswer(BaseModel):
    """A candidate answer retained so a later resume can judge it once."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    candidate_model_id: str = Field(min_length=1)
    answer: str


class CollectionAudit(BaseModel):
    """Resume identity and judge charges for one collection file."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["routellm-collection-audit-1"] = _AUDIT_FORMAT
    dataset_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    candidate_ids: list[str]
    candidate_providers: list[str]
    candidate_tiers: list[str]
    judge_model_id: str | None = None
    quality_threshold: float = Field(ge=0, le=1)
    stronger_model_id: str = Field(min_length=1)
    judge_calls: list[JudgeCallRecord] = Field(default_factory=list)
    pending_answers: list[PendingJudgeAnswer] = Field(default_factory=list)
    budget_reservations: list[BudgetReservation] = Field(default_factory=list)


@dataclass
class CollectionEstimate:
    """An approximate price-book ceiling. It is not a measured invoice."""

    prompts: int
    families: int
    models: tuple[str, ...]
    rubric_cases: int
    generation_ceiling: Decimal
    judge_ceiling: Decimal | None
    recommended_max_spend: Decimal | None
    exploratory_even_if_fully_labeled: bool
    proposed_family_count: int
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "prompts": self.prompts,
            "families": self.families,
            "models": list(self.models),
            "rubric_cases": self.rubric_cases,
            "estimate_basis": "approximate character-length allowance",
            "approximate_generation_ceiling_usd": money_to_api(self.generation_ceiling),
            "approximate_judge_ceiling_usd": (
                None if self.judge_ceiling is None else money_to_api(self.judge_ceiling)
            ),
            "approximate_recommended_max_spend_usd": (
                None
                if self.recommended_max_spend is None
                else money_to_api(self.recommended_max_spend)
            ),
            "exploratory_even_if_fully_labeled": self.exploratory_even_if_fully_labeled,
            "proposed_family_count": self.proposed_family_count,
            "notes": list(self.notes),
        }


@dataclass
class CollectionResult:
    """What the collector finished before it stopped."""

    output: Path | None
    prompts_written: int
    provider_calls: int
    judge_calls: int
    spent: Decimal
    reserved: Decimal
    stopped_reason: str | None
    skipped_completed: int = 0
    notes: list[str] = field(default_factory=list)


def repository_root() -> Path:
    """Return the repository root that contains ``backend``."""
    return Path(__file__).resolve().parents[3]


def default_collection_directory() -> Path:
    """Return a durable collection directory outside the repository."""
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else Path.home() / ".local" / "share"
    return base / "RouteLLM-AI" / "collections"


def collection_audit_path(output: Path) -> Path:
    """Return the sibling audit file for a training output."""
    return output.with_name(output.name + ".audit.json")


def assert_collection_output_outside_repository(path: Path) -> None:
    """Reject an output path inside the git repository."""
    resolved = path.resolve()
    root = repository_root()
    if resolved == root or root in resolved.parents:
        raise ValueError("Collection output must be stored outside the repository.")


def prompt_family(case: EvalCase) -> str:
    """Return the family id. A case without one is its own family."""
    return case.group_id or case.id


def dataset_fingerprint(cases: list[EvalCase]) -> str:
    """Hash the case list so a resume cannot continue on a different dataset."""
    payload = [case.model_dump(mode="json") for case in cases]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_collection(
    cases: list[EvalCase],
    models: list[ModelConfig],
    settings: Settings,
    *,
    judge: bool,
) -> CollectionEstimate:
    """Bound the spend from the price book and the configured output caps.

    The bound is approximate. It uses a character-length input allowance and
    each model's configured maximum output, not measured tokens.
    """
    generation = Decimal("0")
    judge_total = Decimal("0")
    judge_ceiling: Decimal | None = Decimal("0")
    priced_models: list[str] = []
    notes: list[str] = [
        "Character-based spending estimates are approximate. They use a "
        "character-length input allowance and each model's configured maximum "
        "output. They are not measured token counts or invoices."
    ]
    notes.append(
        "Recorded outcome cost is generation only. Judge usage and cost are "
        "stored in the sibling audit file, count toward the spending cap, and "
        "are not added to a candidate's training cost."
    )
    for model in models:
        if lookup_prices(model.model_id) is None:
            notes.append(
                f"{model.model_id} has no published token price and is omitted from the ceiling."
            )
            continue
        priced_models.append(model.model_id)
        for case in cases:
            generation += _generation_ceiling(model, case, settings)
    rubric_cases = sum(1 for case in cases if case.method == "rubric")
    if judge and rubric_cases:
        one_judge = _judge_ceiling(settings)
        if one_judge is None:
            judge_ceiling = None
            notes.append(
                "The approximate judge ceiling is unavailable because the judge "
                "model has no published token price. It is not treated as zero."
            )
        else:
            judge_total = one_judge * rubric_cases * len(priced_models)
            judge_ceiling = quantize_money(judge_total)
    families = len({prompt_family(case) for case in cases})
    recommended: Decimal | None
    if judge_ceiling is None:
        recommended = None
    else:
        recommended = quantize_money(generation + judge_ceiling)
    exploratory = len(cases) < EXPLORATORY_LABELED_COUNT
    if exploratory:
        notes.append(
            f"{len(cases)} prompts stay exploratory even if every one receives a label. "
            f"A non-exploratory file needs at least {EXPLORATORY_LABELED_COUNT} labeled "
            f"prompts. The proposed collection is {PROPOSED_FAMILY_COUNT} families, "
            "eight in each of the six categories, with paraphrases sharing a group_id."
        )
    if not judge and rubric_cases:
        notes.append(
            f"{rubric_cases} rubric cases stay verdict unknown unless collection runs with --judge."
        )
    return CollectionEstimate(
        prompts=len(cases),
        families=families,
        models=tuple(priced_models),
        rubric_cases=rubric_cases,
        generation_ceiling=quantize_money(generation),
        judge_ceiling=judge_ceiling,
        recommended_max_spend=recommended,
        exploratory_even_if_fully_labeled=exploratory,
        proposed_family_count=PROPOSED_FAMILY_COUNT,
        notes=tuple(notes),
    )


async def collect_measured(
    cases: list[EvalCase],
    models: list[ModelConfig],
    *,
    provider_factory: ProviderFactory,
    settings: Settings,
    output: Path,
    max_spend: Decimal,
    stronger_model_id: str,
    quality_threshold: float,
    dataset_version: str,
    judge_generate: object | None = None,
    judge_model_id: str | None = None,
    reserve_unknown_judge: bool = False,
) -> CollectionResult:
    """Call candidates until the cap, an unknown incurred cost, or the case list ends."""
    if max_spend < 0:
        raise ValueError("max_spend must be zero or positive")
    assert_collection_output_outside_repository(output)
    fingerprint = dataset_fingerprint(cases)
    existing = _load_partial(output)
    audit = _load_audit(output) if existing is not None else None
    if existing is not None:
        _require_resume_compatible(
            existing,
            audit,
            cases,
            models,
            stronger_model_id,
            quality_threshold,
            dataset_version,
            judge_model_id,
            fingerprint,
        )
        assert audit is not None
        if reserve_unknown_judge:
            _apply_budget_reservations(audit, settings)
            _save(output, existing, audit)
        _validate_stored_reservations(audit, settings)
        reason = _unknown_incurred_reason(existing, audit)
        if reason is not None:
            return CollectionResult(
                output=output,
                prompts_written=len(existing.prompts),
                provider_calls=0,
                judge_calls=0,
                spent=_known_spend(existing, audit),
                reserved=_reservation_total(audit),
                stopped_reason=reason,
                notes=["Spent omits calls whose incurred cost is unknown."],
            )
    else:
        audit = _blank_audit(
            dataset_version,
            fingerprint,
            models,
            judge_model_id,
            quality_threshold,
            stronger_model_id,
        )
    spent = _budget_used(existing, audit)
    calls = 0
    judge_calls = 0
    skipped = 0
    stopped: str | None = None
    records: dict[str, PromptRecord] = {}
    if existing is not None:
        for prompt in existing.prompts:
            records[prompt.prompt_id] = prompt
    judge_index = {
        (item.prompt_id, item.candidate_model_id): item for item in audit.judge_calls
    }
    for case in cases:
        current = records.get(case.id)
        outcomes = [] if current is None else list(current.outcomes)
        known = {item.model_id: item for item in outcomes}
        for model in models:
            previous = known.get(model.model_id)
            if previous is not None and previous.status in {"succeeded", "failed"}:
                skipped += 1
                if previous.status == "succeeded" and _judge_still_required(
                    case, model.model_id, judge_generate, judge_model_id, judge_index
                ):
                    judge_stop, used = await _judge_pending(
                        case,
                        model,
                        cases,
                        models,
                        outcomes,
                        records,
                        audit,
                        judge_index,
                        judge_generate,
                        judge_model_id,
                        settings,
                        spent,
                        max_spend,
                        quality_threshold,
                        output,
                        stronger_model_id,
                        dataset_version,
                    )
                    judge_calls += used
                    if judge_stop == _UNKNOWN_JUDGE and reserve_unknown_judge:
                        _apply_budget_reservations(audit, settings)
                        _persist(
                            output,
                            records,
                            cases,
                            models,
                            audit,
                            stronger_model_id,
                            quality_threshold,
                            dataset_version,
                            judge_model_id,
                        )
                        judge_stop = None
                    if judge_stop is not None:
                        stopped = judge_stop
                        break
                    spent = _budget_used(
                        _assembled(
                            [item.id for item in cases if item.id in records],
                            records,
                            models,
                            stronger_model_id,
                            quality_threshold,
                            dataset_version,
                            judge_model_id,
                        ),
                        audit,
                    )
                continue
            if lookup_prices(model.model_id) is None:
                stopped = f"{model.model_id} has no published price; it was not called"
                break
            ceiling = _generation_ceiling(model, case, settings)
            hold = ceiling
            judge_ceiling = None
            if _wants_judge(case, judge_generate, judge_model_id):
                judge_ceiling = _judge_ceiling(settings)
                if judge_ceiling is None:
                    stopped = "the judge model has no published price; it was not called"
                    break
                hold += judge_ceiling
            if spent + hold > max_spend:
                stopped = "spending cap reached before the next generation"
                break
            generated, failure = await _call_model(provider_factory, model, case)
            calls += 1
            if generated is None:
                outcomes = _replace(outcomes, _failed_outcome(model, case, settings, failure))
                records[case.id] = _prompt_record(case, outcomes)
                _persist(
                    output,
                    records,
                    cases,
                    models,
                    audit,
                    stronger_model_id,
                    quality_threshold,
                    dataset_version,
                    judge_model_id,
                )
                continue
            priced = _price(generated, model.model_id)
            if _estimate_unknown(priced):
                outcomes = _replace(
                    outcomes,
                    _succeeded_outcome(
                        model,
                        case,
                        settings,
                        priced,
                        grade_case(case, generated.content),
                        generated.latency_ms,
                    ),
                )
                records[case.id] = _prompt_record(case, outcomes)
                _persist(
                    output,
                    records,
                    cases,
                    models,
                    audit,
                    stronger_model_id,
                    quality_threshold,
                    dataset_version,
                    judge_model_id,
                )
                stopped = _UNKNOWN_GENERATION
                break
            assert priced.total is not None
            quality = grade_case(case, generated.content)
            if _wants_judge(case, judge_generate, judge_model_id):
                assert judge_ceiling is not None
                if spent + priced.total + judge_ceiling > max_spend:
                    outcomes = _replace(
                        outcomes,
                        _succeeded_outcome(
                            model, case, settings, priced, quality, generated.latency_ms
                        ),
                    )
                    records[case.id] = _prompt_record(case, outcomes)
                    _retain_answer(audit, case.id, model.model_id, generated.content)
                    _persist(
                        output,
                        records,
                        cases,
                        models,
                        audit,
                        stronger_model_id,
                        quality_threshold,
                        dataset_version,
                        judge_model_id,
                    )
                    stopped = "spending cap reached before the judge call"
                    break
                judged = await run_judge(
                    judge_generate,  # type: ignore[arg-type]
                    model_id=judge_model_id or "",
                    prompt=case.prompt,
                    answer=generated.content,
                    rubric=case.criteria or "",
                )
                judge_calls += 1
                quality = apply_score_threshold(judged, quality_threshold)
                record = _judge_record(case.id, model.model_id, judge_model_id or "", judged)
                _store_judge_call(audit, judge_index, record)
                outcomes = _replace(
                    outcomes,
                    _succeeded_outcome(
                        model, case, settings, priced, quality, generated.latency_ms
                    ),
                )
                records[case.id] = _prompt_record(case, outcomes)
                _persist(
                    output,
                    records,
                    cases,
                    models,
                    audit,
                    stronger_model_id,
                    quality_threshold,
                    dataset_version,
                    judge_model_id,
                )
                if _judge_record_unknown(record):
                    if reserve_unknown_judge:
                        _apply_budget_reservations(audit, settings)
                        _persist(
                            output,
                            records,
                            cases,
                            models,
                            audit,
                            stronger_model_id,
                            quality_threshold,
                            dataset_version,
                            judge_model_id,
                        )
                    else:
                        stopped = _UNKNOWN_JUDGE
                        break
            else:
                outcomes = _replace(
                    outcomes,
                    _succeeded_outcome(
                        model, case, settings, priced, quality, generated.latency_ms
                    ),
                )
                records[case.id] = _prompt_record(case, outcomes)
                _persist(
                    output,
                    records,
                    cases,
                    models,
                    audit,
                    stronger_model_id,
                    quality_threshold,
                    dataset_version,
                    judge_model_id,
                )
            spent = _budget_used(
                _assembled(
                    [item.id for item in cases if item.id in records],
                    records,
                    models,
                    stronger_model_id,
                    quality_threshold,
                    dataset_version,
                    judge_model_id,
                ),
                audit,
            )
            if stopped is not None:
                break
            if spent > max_spend:
                stopped = "spending cap reached after a recorded call"
                break
        if stopped is not None:
            break
    written = _assembled(
        [case.id for case in cases if case.id in records],
        records,
        models,
        stronger_model_id,
        quality_threshold,
        dataset_version,
        judge_model_id,
    )
    path = None
    if written.prompts:
        _save(output, written, audit)
        path = output
    notes: list[str] = []
    if stopped in {_UNKNOWN_GENERATION, _UNKNOWN_JUDGE}:
        notes.append("Spent omits calls whose incurred cost is unknown.")
    return CollectionResult(
        output=path,
        prompts_written=len(written.prompts),
        provider_calls=calls,
        judge_calls=judge_calls,
        spent=_known_spend(written, audit),
        reserved=_reservation_total(audit),
        stopped_reason=stopped,
        skipped_completed=skipped,
        notes=notes,
    )


def _generation_ceiling(model: ModelConfig, case: EvalCase, settings: Settings) -> Decimal:
    metadata = lookup_prices(model.model_id)
    if metadata is None:
        return Decimal("0")
    text = case.prompt if case.system_prompt is None else f"{case.prompt}\n{case.system_prompt}"
    input_tokens = max(1, (len(text) + 3) // 4)
    output_tokens = _output_cap(model, settings)
    return quantize_money(
        line_cost(input_tokens, metadata.input_cost_per_million_tokens)
        + line_cost(output_tokens, metadata.output_cost_per_million_tokens)
    )


def _judge_ceiling(settings: Settings) -> Decimal | None:
    metadata = lookup_prices(settings.quality_judge_model)
    if metadata is None:
        return None
    return quantize_money(
        line_cost(_JUDGE_INPUT_ALLOWANCE, metadata.input_cost_per_million_tokens)
        + line_cost(settings.openai_max_output_tokens, metadata.output_cost_per_million_tokens)
    )


def _output_cap(model: ModelConfig, settings: Settings) -> int:
    if model.provider is Provider.OPENAI:
        return settings.openai_max_output_tokens
    if model.provider is Provider.ANTHROPIC:
        return settings.anthropic_max_tokens
    return 1024


def _estimate_unknown(priced: CostEstimate | None) -> bool:
    if priced is None:
        return True
    return priced.total is None or priced.completeness is CostCompleteness.UNKNOWN


def _provider(model: ModelConfig) -> str:
    return model.provider.value


def _settings_for(model: ModelConfig, case: EvalCase, settings: Settings) -> GenerationSettings:
    return GenerationSettings(
        temperature=None,
        max_output_tokens=_output_cap(model, settings),
        system_prompt_applied=case.system_prompt is not None,
    )


def _failed_outcome(
    model: ModelConfig,
    case: EvalCase,
    settings: Settings,
    failure: ProviderError | None,
) -> ModelOutcome:
    category = "unknown"
    if isinstance(failure, ProviderError):
        category = classify_provider_error(failure).category.value
    return ModelOutcome(
        model_id=model.model_id,
        provider=_provider(model),  # type: ignore[arg-type]
        status="failed",
        quality_verdict=None,
        quality_score=None,
        cost=None,
        cost_completeness="unknown",
        latency_ms=None,
        generation_settings=_settings_for(model, case, settings),
        error_category=category,
    )


def _succeeded_outcome(
    model: ModelConfig,
    case: EvalCase,
    settings: Settings,
    priced: CostEstimate,
    quality: object,
    latency_ms: float,
) -> ModelOutcome:
    verdict = getattr(quality, "verdict", None)
    score = getattr(quality, "score", None)
    return ModelOutcome(
        model_id=model.model_id,
        provider=_provider(model),  # type: ignore[arg-type]
        status="succeeded",
        quality_verdict=verdict,
        quality_score=score,
        cost=None if priced.total is None else money_to_api(priced.total),
        cost_completeness=priced.completeness.value,  # type: ignore[arg-type]
        latency_ms=latency_ms,
        generation_settings=_settings_for(model, case, settings),
        error_category=None,
    )


def _replace(outcomes: list[ModelOutcome], outcome: ModelOutcome) -> list[ModelOutcome]:
    kept = [item for item in outcomes if item.model_id != outcome.model_id]
    kept.append(outcome)
    return kept


def _prompt_record(case: EvalCase, outcomes: list[ModelOutcome]) -> PromptRecord:
    return PromptRecord(
        prompt_id=case.id,
        text=case.prompt,
        category=case.category,
        group_id=prompt_family(case),
        outcomes=outcomes,
    )


def _rubric_version(judge_model_id: str | None, quality_threshold: float) -> str:
    judge_name = judge_model_id or "none"
    return f"dataset-graders; judge={judge_name}; threshold={quality_threshold}"


def _assembled(
    order: list[str],
    records: dict[str, PromptRecord],
    models: list[ModelConfig],
    stronger_model_id: str,
    quality_threshold: float,
    dataset_version: str,
    judge_model_id: str | None,
) -> TrainingDataset:
    prompts = [
        records[prompt_id]
        for prompt_id in order
        if prompt_id in records and records[prompt_id].outcomes
    ]
    return TrainingDataset(
        version=dataset_version,
        provenance="live",
        evidence="measured",
        rubric_version=_rubric_version(judge_model_id, quality_threshold),
        quality_threshold=quality_threshold,
        stronger_model_id=stronger_model_id,
        generation_settings=GenerationSettings(),
        candidates=[
            CandidateModel(
                model_id=model.model_id,
                provider=_provider(model),  # type: ignore[arg-type]
                quality_tier=model.quality_tier.value,  # type: ignore[arg-type]
            )
            for model in models
        ],
        prompts=prompts,
    )


def _blank_audit(
    dataset_version: str,
    fingerprint: str,
    models: list[ModelConfig],
    judge_model_id: str | None,
    quality_threshold: float,
    stronger_model_id: str,
) -> CollectionAudit:
    return CollectionAudit(
        dataset_version=dataset_version,
        dataset_fingerprint=fingerprint,
        candidate_ids=[model.model_id for model in models],
        candidate_providers=[_provider(model) for model in models],
        candidate_tiers=[model.quality_tier.value for model in models],
        judge_model_id=judge_model_id,
        quality_threshold=quality_threshold,
        stronger_model_id=stronger_model_id,
    )


def _require_resume_compatible(
    existing: TrainingDataset,
    audit: CollectionAudit | None,
    cases: list[EvalCase],
    models: list[ModelConfig],
    stronger_model_id: str,
    quality_threshold: float,
    dataset_version: str,
    judge_model_id: str | None,
    fingerprint: str,
) -> None:
    if existing.provenance != "live" or existing.evidence != "measured":
        raise ValueError("Refusing to resume a file that is not live measured evidence.")
    if audit is None:
        raise ValueError("Refusing to resume without the collection audit.")
    by_id = {case.id: case for case in cases}
    dataset_matches = (
        existing.version == dataset_version
        and audit.dataset_version == dataset_version
        and audit.dataset_fingerprint == fingerprint
    )
    if dataset_matches:
        for prompt in existing.prompts:
            case = by_id.get(prompt.prompt_id)
            if (
                case is None
                or prompt.text != case.prompt
                or prompt.category != case.category
                or prompt.group_id != prompt_family(case)
            ):
                dataset_matches = False
                break
    if not dataset_matches:
        raise ValueError("Refusing to resume because the dataset does not match.")
    stored = [(item.model_id, item.provider, item.quality_tier) for item in existing.candidates]
    current = [(model.model_id, _provider(model), model.quality_tier.value) for model in models]
    candidates_match = (
        stored == current
        and audit.candidate_ids == [model.model_id for model in models]
        and audit.candidate_providers == [_provider(model) for model in models]
        and audit.candidate_tiers == [model.quality_tier.value for model in models]
        and existing.stronger_model_id == stronger_model_id
        and audit.stronger_model_id == stronger_model_id
    )
    if not candidates_match:
        raise ValueError("Refusing to resume because the candidate models do not match.")
    judge_matches = (
        audit.judge_model_id == judge_model_id
        and audit.quality_threshold == quality_threshold
        and existing.quality_threshold == quality_threshold
        and existing.rubric_version == _rubric_version(judge_model_id, quality_threshold)
    )
    if not judge_matches:
        raise ValueError("Refusing to resume because the judge configuration does not match.")


def _outcome_cost_unknown(outcome: ModelOutcome) -> bool:
    if outcome.status != "succeeded":
        return False
    return outcome.cost is None or outcome.cost_completeness == "unknown"


def _judge_record_unknown(record: JudgeCallRecord) -> bool:
    return record.cost is None or record.cost_completeness == "unknown"


def _unknown_incurred_reason(
    dataset: TrainingDataset | None,
    audit: CollectionAudit | None,
) -> str | None:
    if dataset is not None:
        for prompt in dataset.prompts:
            if any(_outcome_cost_unknown(item) for item in prompt.outcomes):
                return _UNKNOWN_GENERATION
    if audit is not None and any(
        _judge_record_unknown(item) and _reservation_for(audit, item) is None
        for item in audit.judge_calls
    ):
        return _UNKNOWN_JUDGE
    return None


def _reservation_amount(metadata: object, output_cap: int) -> Decimal:
    """Hold the context-window input at the uncached rate plus the output cap."""
    return quantize_money(
        line_cost(metadata.context_window, metadata.input_cost_per_million_tokens)
        + line_cost(output_cap, metadata.output_cost_per_million_tokens)
    )


def _supported_reservation(
    record: JudgeCallRecord,
    settings: Settings,
) -> tuple[Decimal, int, object]:
    """Price the judge model's context window and the judge output cap.

    The candidate model's output cap is a different request. The judge client
    sends ``openai_max_output_tokens``.
    """
    metadata = lookup_prices(record.judge_model_id)
    if metadata is None:
        raise ValueError(
            f"{record.judge_model_id} has no published price; the reservation is not supported."
        )
    output_cap = settings.openai_max_output_tokens
    return _reservation_amount(metadata, output_cap), output_cap, metadata


def _reservation_for(audit: CollectionAudit, record: JudgeCallRecord) -> BudgetReservation | None:
    matches = [
        item
        for item in audit.budget_reservations
        if item.prompt_id == record.prompt_id
        and item.candidate_model_id == record.candidate_model_id
        and item.judge_model_id == record.judge_model_id
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError("An unknown judge call has more than one budget reservation.")
    return matches[0]


def _apply_budget_reservations(
    audit: CollectionAudit,
    settings: Settings,
) -> None:
    """Record a separate hold for each unknown judge call. Costs stay unknown."""
    for record in audit.judge_calls:
        if not _judge_record_unknown(record):
            continue
        amount, output_cap, metadata = _supported_reservation(record, settings)
        existing = _reservation_for(audit, record)
        if existing is not None:
            continue
        audit.budget_reservations.append(
            BudgetReservation(
                prompt_id=record.prompt_id,
                candidate_model_id=record.candidate_model_id,
                judge_model_id=record.judge_model_id,
                amount=money_to_api(amount) or "0",
                basis="context_window_and_output_cap",
                context_window_tokens=metadata.context_window,
                output_token_cap=output_cap,
                input_price_per_million=money_to_api(metadata.input_cost_per_million_tokens) or "0",
                output_price_per_million=money_to_api(metadata.output_cost_per_million_tokens) or "0",
                measured=False,
            )
        )


def _validate_stored_reservations(
    audit: CollectionAudit,
    settings: Settings,
) -> None:
    for item in audit.budget_reservations:
        if item.measured:
            raise ValueError("A budget reservation cannot be marked measured.")
        record = next(
            (
                call
                for call in audit.judge_calls
                if call.prompt_id == item.prompt_id
                and call.candidate_model_id == item.candidate_model_id
                and call.judge_model_id == item.judge_model_id
            ),
            None,
        )
        if record is None or not _judge_record_unknown(record):
            raise ValueError("A budget reservation does not match an unknown judge call.")
        amount, output_cap, metadata = _supported_reservation(record, settings)
        if (
            Decimal(item.amount) != amount
            or item.output_token_cap != output_cap
            or item.context_window_tokens != metadata.context_window
            or Decimal(item.input_price_per_million) != metadata.input_cost_per_million_tokens
            or Decimal(item.output_price_per_million) != metadata.output_cost_per_million_tokens
        ):
            raise ValueError(
                f"Stored reservation {item.amount} does not match the supported reservation {money_to_api(amount)}."
            )


def _reservation_total(audit: CollectionAudit | None) -> Decimal:
    if audit is None:
        return Decimal("0")
    total = sum((Decimal(item.amount) for item in audit.budget_reservations), Decimal("0"))
    return quantize_money(total)


def _budget_used(dataset: TrainingDataset | None, audit: CollectionAudit | None) -> Decimal:
    """Measured spend plus separate reservations. Reservations are not measured."""
    return quantize_money(_known_spend(dataset, audit) + _reservation_total(audit))


def _known_spend(dataset: TrainingDataset | None, audit: CollectionAudit | None) -> Decimal:
    total = Decimal("0")
    if dataset is not None:
        for prompt in dataset.prompts:
            for outcome in prompt.outcomes:
                if outcome.cost is None or _outcome_cost_unknown(outcome):
                    continue
                total += Decimal(outcome.cost)
    if audit is not None:
        for record in audit.judge_calls:
            if record.cost is None or _judge_record_unknown(record):
                continue
            total += Decimal(record.cost)
    return quantize_money(total)


def _wants_judge(
    case: EvalCase,
    judge_generate: object | None,
    judge_model_id: str | None,
) -> bool:
    return case.method == "rubric" and judge_generate is not None and judge_model_id is not None


def _judge_still_required(
    case: EvalCase,
    model_id: str,
    judge_generate: object | None,
    judge_model_id: str | None,
    judge_index: dict[tuple[str, str], JudgeCallRecord],
) -> bool:
    if not _wants_judge(case, judge_generate, judge_model_id):
        return False
    return (case.id, model_id) not in judge_index


def _retain_answer(audit: CollectionAudit, prompt_id: str, model_id: str, answer: str) -> None:
    kept = [
        item
        for item in audit.pending_answers
        if (item.prompt_id, item.candidate_model_id) != (prompt_id, model_id)
    ]
    kept.append(
        PendingJudgeAnswer(prompt_id=prompt_id, candidate_model_id=model_id, answer=answer)
    )
    audit.pending_answers = kept


def _drop_pending(audit: CollectionAudit, prompt_id: str, model_id: str) -> None:
    audit.pending_answers = [
        item
        for item in audit.pending_answers
        if (item.prompt_id, item.candidate_model_id) != (prompt_id, model_id)
    ]


def _store_judge_call(
    audit: CollectionAudit,
    judge_index: dict[tuple[str, str], JudgeCallRecord],
    record: JudgeCallRecord,
) -> None:
    audit.judge_calls = [
        item
        for item in audit.judge_calls
        if (item.prompt_id, item.candidate_model_id)
        != (record.prompt_id, record.candidate_model_id)
    ]
    audit.judge_calls.append(record)
    judge_index[(record.prompt_id, record.candidate_model_id)] = record
    _drop_pending(audit, record.prompt_id, record.candidate_model_id)


def _judge_record(
    prompt_id: str,
    candidate_model_id: str,
    judge_model_id: str,
    judged: object,
) -> JudgeCallRecord:
    usage = getattr(judged, "judge_usage", None)
    priced = getattr(judged, "judge_cost", None)
    if _estimate_unknown(priced):
        completeness = "unknown"
        cost = None
    else:
        completeness = priced.completeness.value
        cost = money_to_api(priced.total)
    return JudgeCallRecord(
        prompt_id=prompt_id,
        candidate_model_id=candidate_model_id,
        judge_model_id=judge_model_id,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        cached_input_tokens=None if usage is None else usage.cached_input_tokens,
        cache_write_input_tokens=None if usage is None else usage.cache_write_input_tokens,
        cache_read_input_tokens=None if usage is None else usage.cache_read_input_tokens,
        cache_write_5m_tokens=None if usage is None else usage.cache_write_5m_tokens,
        cache_write_1h_tokens=None if usage is None else usage.cache_write_1h_tokens,
        cost=cost,
        cost_completeness=completeness,
        latency_ms=getattr(judged, "judge_latency_ms", None),
        error_category=getattr(judged, "error_category", None),
        error_type=getattr(judged, "error_message", None),
    )


async def _judge_pending(
    case: EvalCase,
    model: ModelConfig,
    cases: list[EvalCase],
    models: list[ModelConfig],
    outcomes: list[ModelOutcome],
    records: dict[str, PromptRecord],
    audit: CollectionAudit,
    judge_index: dict[tuple[str, str], JudgeCallRecord],
    judge_generate: object | None,
    judge_model_id: str | None,
    settings: Settings,
    spent: Decimal,
    max_spend: Decimal,
    quality_threshold: float,
    output: Path,
    stronger_model_id: str,
    dataset_version: str,
) -> tuple[str | None, int]:
    pending = next(
        (
            item
            for item in audit.pending_answers
            if item.prompt_id == case.id and item.candidate_model_id == model.model_id
        ),
        None,
    )
    if pending is None:
        return (
            "a rubric outcome has no judge record and no retained answer; "
            "refusing to assume the judge cost is zero",
            0,
        )
    judge_ceiling = _judge_ceiling(settings)
    if judge_ceiling is None:
        return ("the judge model has no published price; it was not called", 0)
    if spent + judge_ceiling > max_spend:
        return ("spending cap reached before the judge call", 0)
    assert judge_generate is not None and judge_model_id is not None
    judged = await run_judge(
        judge_generate,  # type: ignore[arg-type]
        model_id=judge_model_id,
        prompt=case.prompt,
        answer=pending.answer,
        rubric=case.criteria or "",
    )
    quality = apply_score_threshold(judged, quality_threshold)
    previous = next(item for item in outcomes if item.model_id == model.model_id)
    updated = previous.model_copy(
        update={"quality_verdict": quality.verdict, "quality_score": quality.score}
    )
    replaced = _replace(outcomes, updated)
    records[case.id] = _prompt_record(case, replaced)
    outcomes[:] = replaced
    record = _judge_record(case.id, model.model_id, judge_model_id, judged)
    _store_judge_call(audit, judge_index, record)
    _persist(
        output,
        records,
        cases,
        models,
        audit,
        stronger_model_id,
        quality_threshold,
        dataset_version,
        judge_model_id,
    )
    if _judge_record_unknown(record):
        return (_UNKNOWN_JUDGE, 1)
    return (None, 1)


def _persist(
    output: Path,
    records: dict[str, PromptRecord],
    cases: list[EvalCase],
    models: list[ModelConfig],
    audit: CollectionAudit,
    stronger_model_id: str,
    quality_threshold: float,
    dataset_version: str,
    judge_model_id: str | None,
) -> None:
    order = [case.id for case in cases if case.id in records]
    # Keep prompts that were loaded from an earlier prefix but are absent from
    # the cases argument used by a pending-judge save.
    for prompt_id in records:
        if prompt_id not in order:
            order.append(prompt_id)
    written = _assembled(
        order,
        records,
        models,
        stronger_model_id,
        quality_threshold,
        dataset_version,
        judge_model_id,
    )
    if not written.prompts:
        return
    _save(output, written, audit)


async def _call_model(
    provider_factory: ProviderFactory,
    model: ModelConfig,
    case: EvalCase,
):
    """Return the provider response, or the exception when the call fails."""
    provider = None
    try:
        provider = provider_factory(model)
        generated = await provider.generate(case.prompt, case.system_prompt)
        return generated, None
    except ProviderError as exc:
        return None, exc
    finally:
        if provider is not None:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()


def _save(path: Path, dataset: TrainingDataset, audit: CollectionAudit) -> None:
    if not dataset.prompts:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = collection_audit_path(path)
    audit_temporary = audit_path.with_suffix(audit_path.suffix + ".partial")
    training_temporary = path.with_suffix(path.suffix + ".partial")
    audit_temporary.write_text(audit.model_dump_json(indent=2) + "\n", encoding="utf-8")
    training_temporary.write_text(dataset.model_dump_json(indent=2) + "\n", encoding="utf-8")
    audit_temporary.replace(audit_path)
    training_temporary.replace(path)


def _load_partial(path: Path) -> TrainingDataset | None:
    if not path.is_file():
        return None
    return load_training_dataset(path)


def load_collection_audit(path: Path) -> CollectionAudit:
    """Load the sibling audit for a training file."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return CollectionAudit.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Collection audit is invalid ({type(exc).__name__})") from None


def _load_audit(output: Path) -> CollectionAudit | None:
    path = collection_audit_path(output)
    if not path.is_file():
        return None
    return load_collection_audit(path)


def render_estimate(estimate: CollectionEstimate) -> str:
    """Text report for the estimate command. Ceilings are labeled approximate."""
    judge_text = (
        "unavailable"
        if estimate.judge_ceiling is None
        else money_to_api(estimate.judge_ceiling)
    )
    spend_text = (
        "unavailable"
        if estimate.recommended_max_spend is None
        else money_to_api(estimate.recommended_max_spend)
    )
    lines = [
        f"Prompts: {estimate.prompts}",
        f"Families: {estimate.families}",
        f"Models: {', '.join(estimate.models) if estimate.models else '(none)'}",
        f"Rubric cases: {estimate.rubric_cases}",
        f"Approximate generation ceiling USD: {money_to_api(estimate.generation_ceiling)}",
        f"Approximate judge ceiling USD: {judge_text}",
        f"Approximate recommended max spend USD: {spend_text}",
        (
            "Exploratory even if every prompt is labeled: "
            f"{'yes' if estimate.exploratory_even_if_fully_labeled else 'no'}"
        ),
        f"Proposed family count: {estimate.proposed_family_count}",
        "Estimate basis: approximate character-length allowance",
    ]
    lines.extend(estimate.notes)
    return "\n".join(lines)


def run_collect(argv: list[str] | None = None) -> int:
    """CLI. ``--estimate`` does not call a provider. ``--execute`` does."""
    import argparse
    import sys

    from app.config import get_settings
    from app.evaluation.dataset import load_dataset, select_cases
    from app.providers.factory import build_provider
    from app.routing.catalog import build_catalog

    parser = argparse.ArgumentParser(
        description=(
            "Collect live per-model outcomes for ML training. "
            "Estimate prints an approximate character-based ceiling and does not spend money."
        )
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print an approximate character-based price-book ceiling. No provider calls.",
    )
    parser.add_argument("--execute", action="store_true", help="Call providers. Requires --output and --max-spend.")
    parser.add_argument("--dataset", default=None, help="Evaluation dataset JSON. Defaults to the smoke file.")
    parser.add_argument("--models", default="gpt-5-nano,claude-sonnet-4-6")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Training JSON path outside the repository. "
            f"Durable directory: {default_collection_directory()}"
        ),
    )
    parser.add_argument(
        "--max-spend",
        default=None,
        help="USD cap for generation plus separately recorded judge calls.",
    )
    parser.add_argument("--judge", action="store_true", help="Grade rubric cases with the configured judge model.")
    parser.add_argument(
        "--reserve-unknown-judge",
        action="store_true",
        help=(
            "Hold a context-window and output-cap reservation for an unknown judge call. "
            "The hold is not measured spend."
        ),
    )
    parser.add_argument("--split", choices=["calibration", "held_out"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.estimate == args.execute:
        print("Choose exactly one of --estimate or --execute.", file=sys.stderr)
        return 2
    settings = get_settings()
    dataset = load_dataset(None if args.dataset is None else Path(args.dataset))
    cases = select_cases(dataset, split=args.split, smoke=args.smoke)
    registry = build_catalog(settings)
    model_ids = [part.strip() for part in args.models.split(",") if part.strip()]
    models = []
    for model_id in model_ids:
        try:
            models.append(registry.get(model_id))
        except KeyError:
            print(f"Unknown model {model_id}", file=sys.stderr)
            return 2
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            return 2
    if args.estimate:
        estimate = estimate_collection(cases, models, settings, judge=args.judge)
        print(render_estimate(estimate))
        return 0
    if args.output is None or args.max_spend is None:
        print("--execute requires --output and --max-spend.", file=sys.stderr)
        return 2
    stronger = settings.premium_baseline_model
    if stronger not in model_ids:
        stronger = model_ids[-1]
    judge_generate = None
    judge_model_id = None
    if args.judge:
        judge_model = registry.get(settings.quality_judge_model)
        judge_generate = _bound_judge(judge_model, settings)
        judge_model_id = judge_model.model_id
    try:
        result = asyncio.run(
            collect_measured(
                cases,
                models,
                provider_factory=lambda model: build_provider(model, settings),
                settings=settings,
                output=Path(args.output),
                max_spend=Decimal(args.max_spend),
                stronger_model_id=stronger,
                quality_threshold=settings.min_quality_score,
                dataset_version=dataset.version,
                judge_generate=judge_generate,
                judge_model_id=judge_model_id,
                reserve_unknown_judge=args.reserve_unknown_judge,
            )
        )
    except KeyboardInterrupt:
        print("Collection cancelled. Partial results stay in the output file.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"Prompts written: {result.prompts_written}; provider calls: {result.provider_calls}; "
        f"judge calls: {result.judge_calls}; known spent USD: {money_to_api(result.spent)}; "
        f"reserved USD: {money_to_api(result.reserved)}"
    )
    if result.stopped_reason:
        print(result.stopped_reason)
    for note in result.notes:
        print(note)
    if result.output is not None:
        print(f"Wrote {result.output}")
        print(f"Wrote {collection_audit_path(result.output)}")
    return 0


def _bound_judge(model: ModelConfig, settings: Settings):
    async def generate(prompt: str, system_prompt: str | None = None):
        provider = build_provider(model, settings)
        try:
            return await provider.generate(prompt, system_prompt)
        finally:
            close = getattr(provider, "aclose", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    # Closing the client must not hide the provider response.
                    pass

    return generate
