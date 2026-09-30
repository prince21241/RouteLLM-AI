"""Group-aware splits.

Prompts that share a ``group_id`` stay in one split, including duplicates
and near-duplicate variants. The test split is reserved for the final
comparison and is not used to choose the confidence threshold.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.ml.labels import LabeledPrompt

EXPLORATORY_LABELED_COUNT = 40


class SplitError(ValueError):
    """Raised when the labeled data cannot support an honest three-way split."""


@dataclass(frozen=True)
class SplitAssignment:
    """Recorded split for every labeled prompt."""

    seed: int
    dataset_hash: str
    exploratory: bool
    exploratory_reason: str | None
    group_splits: dict[str, str]
    prompts: tuple[LabeledPrompt, ...]

    def split_for(self, prompt_id: str) -> str:
        for prompt in self.prompts:
            if prompt.prompt_id == prompt_id:
                if prompt.label is None:
                    raise SplitError(f"{prompt_id} was excluded from supervised labels")
                return self.group_splits[prompt.group_id]
        raise SplitError(f"Unknown prompt {prompt_id}")

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "dataset_hash": self.dataset_hash,
            "exploratory": self.exploratory,
            "exploratory_reason": self.exploratory_reason,
            "group_splits": self.group_splits,
            "prompt_splits": {
                prompt.prompt_id: self.group_splits[prompt.group_id]
                for prompt in self.prompts
                if prompt.label is not None
            },
        }


def assign_splits(
    prompts: tuple[LabeledPrompt, ...] | list[LabeledPrompt],
    *,
    seed: int,
    dataset_hash: str,
) -> SplitAssignment:
    """Put each prompt family entirely into train, validation, or test."""
    included = [prompt for prompt in prompts if prompt.label is not None]
    if len(included) < 9:
        raise SplitError(
            "Need at least 9 labeled prompts to reserve train, validation, and "
            f"test partitions; found {len(included)}. Collect more comparable "
            "evaluations before fitting a router."
        )
    groups: dict[str, list[LabeledPrompt]] = defaultdict(list)
    for prompt in included:
        groups[prompt.group_id].append(prompt)
    by_class: dict[str, list[str]] = defaultdict(list)
    for group_id, members in groups.items():
        counts = Counter(member.label for member in members)
        label = sorted(counts, key=lambda name: (-counts[name], name))[0]
        by_class[label].append(group_id)
    if len(by_class) < 2:
        found = ", ".join(sorted(by_class)) or "(none)"
        raise SplitError(
            "Supervised training needs at least two label classes so the "
            f"classifier is not a constant. Found only: {found}."
        )
    shortages = []
    for label, group_ids in sorted(by_class.items()):
        if len(group_ids) < 3:
            shortages.append(f"{label} has {len(group_ids)} group(s)")
    if shortages:
        raise SplitError(
            "Each label class needs at least 3 prompt groups so train, "
            "validation, and the held-out test each contain that class. "
            + "; ".join(shortages)
            + "."
        )
    rng = random.Random(seed)
    group_splits: dict[str, str] = {}
    for label in sorted(by_class):
        ordered = list(by_class[label])
        rng.shuffle(ordered)
        group_splits[ordered[0]] = "train"
        group_splits[ordered[1]] = "validation"
        group_splits[ordered[2]] = "test"
        for group_id in ordered[3:]:
            group_splits[group_id] = "train"
    exploratory = len(included) < EXPLORATORY_LABELED_COUNT
    reason = None
    if exploratory:
        reason = (
            f"{len(included)} labeled prompts is below {EXPLORATORY_LABELED_COUNT}. "
            "Treat metrics as exploratory. Dataset sufficiency depends on "
            "coverage, class balance, and held-out results, not on a fixed count."
        )
    return SplitAssignment(
        seed=seed,
        dataset_hash=dataset_hash,
        exploratory=exploratory,
        exploratory_reason=reason,
        group_splits=group_splits,
        prompts=tuple(included),
    )
