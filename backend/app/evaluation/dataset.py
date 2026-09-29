"""Versioned evaluation cases.

Calibration cases are for checking the grader. Held-out cases are the
benchmark split. Neither split is removed when a run fails.
"""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "v1.json"
CATEGORIES = frozenset({"factual", "reasoning", "math", "coding", "structured", "instruction"})
SPLITS = frozenset({"calibration", "held_out"})
METHODS = frozenset({"exact", "numeric", "json_fields", "rubric"})
REFERENCE_METHODS = frozenset({"exact", "numeric", "json_fields"})


class DatasetError(ValueError):
    """Raised when the dataset file does not match the versioned contract."""


class EvalCase(BaseModel):
    """One stable prompt and the check that grades it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    split: Literal["calibration", "held_out"]
    category: Literal["factual", "reasoning", "math", "coding", "structured", "instruction"]
    prompt: str = Field(min_length=1)
    method: Literal["exact", "numeric", "json_fields", "rubric"]
    expected: Any = None
    criteria: str | None = None
    smoke: bool = False
    system_prompt: str | None = None


class EvaluationDataset(BaseModel):
    """A versioned list of cases."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    cases: list[EvalCase]


def load_dataset(path: Path | None = None) -> EvaluationDataset:
    """Load and validate the dataset. Invalid files raise ``DatasetError``."""
    source = DATASET_PATH if path is None else path
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        dataset = EvaluationDataset.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise DatasetError(f"Evaluation dataset is invalid ({type(exc).__name__})") from None
    _validate_contract(dataset)
    return dataset


def select_cases(
    dataset: EvaluationDataset,
    *,
    split: str | None = None,
    smoke: bool = False,
) -> list[EvalCase]:
    """Return cases in file order. ``smoke`` keeps the marked smoke subset."""
    chosen = dataset.cases
    if split is not None:
        if split not in SPLITS:
            raise DatasetError(f"Unknown split {split!r}")
        chosen = [case for case in chosen if case.split == split]
    if smoke:
        chosen = [case for case in chosen if case.smoke]
    return list(chosen)


def _validate_contract(dataset: EvaluationDataset) -> None:
    if len(dataset.cases) < 25:
        raise DatasetError("Evaluation dataset must contain at least 25 cases")
    ids = [case.id for case in dataset.cases]
    if len(ids) != len(set(ids)):
        raise DatasetError("Evaluation case ids must be unique")
    splits = {case.split for case in dataset.cases}
    if splits != SPLITS:
        raise DatasetError("Evaluation dataset must include calibration and held-out cases")
    categories = {case.category for case in dataset.cases}
    if categories != CATEGORIES:
        raise DatasetError("Evaluation dataset must cover every required category")
    if not any(case.smoke for case in dataset.cases):
        raise DatasetError("Evaluation dataset must mark a smoke subset")
    for case in dataset.cases:
        if case.method in REFERENCE_METHODS and case.expected is None:
            raise DatasetError(f"{case.id} requires an expected value")
        if case.method == "rubric" and (case.criteria is None or case.criteria.strip() == ""):
            raise DatasetError(f"{case.id} requires grading criteria")
        if case.method == "numeric":
            try:
                from decimal import Decimal

                Decimal(str(case.expected))
            except Exception:
                raise DatasetError(f"{case.id} has a non-numeric expected value") from None
        if case.method == "json_fields" and not isinstance(case.expected, dict):
            raise DatasetError(f"{case.id} expected value must be a JSON object")
        if case.method == "exact" and not isinstance(case.expected, str):
            raise DatasetError(f"{case.id} expected value must be a string")
