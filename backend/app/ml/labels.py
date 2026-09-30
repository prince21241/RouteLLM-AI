"""Supervised labels from comparable per-model outcomes.

A label is the cheapest model that passed the quality threshold with a
usable cost, or an explicit no-acceptable-model outcome. Unknown grades,
provider failures, missing rows, and incomplete costs stay identifiable
and are not treated as quality failures or as wins for another model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

from app.ml.schema import NO_ACCEPTABLE_MODEL, ModelOutcome, PromptRecord, TrainingDataset

USABLE_COST = frozenset({"complete", "estimated"})
EXCLUSION_PRIORITY = (
    "missing_evaluation",
    "provider_failure",
    "unknown_grade",
    "incomplete_cost",
)


@dataclass(frozen=True)
class LabeledPrompt:
    """One prompt after labeling. ``label`` is empty when the case is excluded."""

    prompt_id: str
    group_id: str
    text: str
    category: str
    label: str | None
    exclusion_reason: str | None


@dataclass(frozen=True)
class LabelReport:
    """Included labels plus exclusion counts. Excluded rows stay in ``prompts``."""

    prompts: tuple[LabeledPrompt, ...]
    included: int
    excluded: int
    exclusion_counts: dict[str, int]
    label_counts: dict[str, int]
    exploratory: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "included": self.included,
            "excluded": self.excluded,
            "exclusion_counts": self.exclusion_counts,
            "label_counts": self.label_counts,
            "exploratory": self.exploratory,
            "prompts": [
                {
                    "prompt_id": item.prompt_id,
                    "group_id": item.group_id,
                    "category": item.category,
                    "label": item.label,
                    "exclusion_reason": item.exclusion_reason,
                }
                for item in self.prompts
            ],
        }


def label_dataset(dataset: TrainingDataset) -> LabelReport:
    """Label every prompt. Mock rows and fixture rows are not measured evidence."""
    if dataset.provenance == "mock":
        raise ValueError(
            "Mock answers are not training evidence. Label live outcomes only."
        )
    prompts = tuple(_label_prompt(prompt, dataset) for prompt in dataset.prompts)
    included_rows = [item for item in prompts if item.label is not None]
    exclusion_counts = Counter(
        item.exclusion_reason for item in prompts if item.exclusion_reason is not None
    )
    label_counts = Counter(item.label for item in included_rows)
    return LabelReport(
        prompts=prompts,
        included=len(included_rows),
        excluded=len(prompts) - len(included_rows),
        exclusion_counts=dict(exclusion_counts),
        label_counts=dict(label_counts),
        exploratory=len(included_rows) < 40,
    )


def _label_prompt(prompt: PromptRecord, dataset: TrainingDataset) -> LabeledPrompt:
    by_id = {outcome.model_id: outcome for outcome in prompt.outcomes}
    blockers: list[str] = []
    passing: list[tuple[Decimal, str]] = []
    for candidate in dataset.candidates:
        outcome = by_id.get(candidate.model_id)
        if outcome is None or outcome.status == "missing":
            blockers.append("missing_evaluation")
            continue
        if outcome.status == "failed":
            blockers.append("provider_failure")
            continue
        verdict = outcome.quality_verdict
        if verdict in {None, "unknown", "error"}:
            blockers.append("unknown_grade")
            continue
        if verdict == "fail":
            continue
        if not _passes_threshold(outcome, dataset.quality_threshold):
            continue
        cost = _usable_cost(outcome)
        if cost is None:
            blockers.append("incomplete_cost")
            continue
        passing.append((cost, candidate.model_id))
    if blockers:
        reason = min(blockers, key=EXCLUSION_PRIORITY.index)
        return _row(prompt, label=None, reason=reason)
    if not passing:
        return _row(prompt, label=NO_ACCEPTABLE_MODEL, reason=None)
    passing.sort(key=lambda item: (item[0], item[1]))
    return _row(prompt, label=passing[0][1], reason=None)


def _passes_threshold(outcome: ModelOutcome, threshold: float) -> bool:
    if outcome.quality_verdict != "pass":
        return False
    if outcome.quality_score is None:
        return True
    return outcome.quality_score >= threshold


def _usable_cost(outcome: ModelOutcome) -> Decimal | None:
    if outcome.cost is None or outcome.cost_completeness not in USABLE_COST:
        return None
    return Decimal(outcome.cost)


def _row(prompt: PromptRecord, *, label: str | None, reason: str | None) -> LabeledPrompt:
    return LabeledPrompt(
        prompt_id=prompt.prompt_id,
        group_id=prompt.group_id,
        text=prompt.text,
        category=prompt.category,
        label=label,
        exclusion_reason=reason,
    )
