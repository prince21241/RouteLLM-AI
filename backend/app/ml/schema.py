"""Versioned training records for the optional ML router.

Prompt text is the only routing feature. Reference answers, judge prompts,
and outcome fields are labels or audit data, not model inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TRAINING_FORMAT = "routellm-training-1"
NO_ACCEPTABLE_MODEL = "__no_acceptable_model__"


class TrainingDataError(ValueError):
    """Raised when a training file does not match the versioned contract."""


class GenerationSettings(BaseModel):
    """How an answer was requested. These values are not routing features."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    system_prompt_applied: bool | None = None


class ModelOutcome(BaseModel):
    """One model's recorded result for one prompt."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    provider: Literal["openai", "anthropic", "ollama"]
    status: Literal["succeeded", "failed", "missing"]
    quality_verdict: Literal["pass", "fail", "unknown", "error"] | None = None
    quality_score: float | None = Field(default=None, ge=0, le=1)
    cost: str | None = None
    cost_completeness: Literal["complete", "estimated", "unknown"] | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    generation_settings: GenerationSettings | None = None
    error_category: str | None = None


class CandidateModel(BaseModel):
    """A model the offline policies may select. Tiers match the rule router."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    provider: Literal["openai", "anthropic", "ollama"]
    quality_tier: Literal["low", "medium", "high"]
    enabled: bool = True


class PromptRecord(BaseModel):
    """One prompt and every candidate model's recorded outcome."""

    model_config = ConfigDict(extra="forbid")

    prompt_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    category: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    outcomes: list[ModelOutcome]


class TrainingDataset(BaseModel):
    """Comparable evaluations of the same prompts.

    ``provenance`` records whether the outcomes came from a live provider
    call or a mock. ``evidence`` separates measured runs from synthetic
    fixtures used to test the pipeline.
    """

    model_config = ConfigDict(extra="forbid")

    format: Literal["routellm-training-1"] = TRAINING_FORMAT
    version: str = Field(min_length=1)
    provenance: Literal["live", "mock"]
    evidence: Literal["measured", "fixture"]
    rubric_version: str = Field(min_length=1)
    quality_threshold: float = Field(ge=0, le=1)
    stronger_model_id: str = Field(min_length=1)
    generation_settings: GenerationSettings = Field(default_factory=GenerationSettings)
    candidates: list[CandidateModel]
    prompts: list[PromptRecord]


def dataset_hash(dataset: TrainingDataset) -> str:
    """Hash the canonical dataset. Split assignments record this value."""
    payload = dataset.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_training_dataset(path: Path) -> TrainingDataset:
    """Load a training file. Evaluation smoke files are rejected."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingDataError(f"Training dataset is invalid ({type(exc).__name__})") from None
    if isinstance(payload, dict) and "cases" in payload and "prompts" not in payload:
        raise TrainingDataError(
            "This file is an evaluation dataset, not training data. "
            "The smoke evaluation set is not ML training evidence."
        )
    try:
        dataset = TrainingDataset.model_validate(payload)
    except ValidationError as exc:
        raise TrainingDataError(f"Training dataset is invalid ({exc.error_count()} errors)") from None
    _validate_contract(dataset)
    return dataset


def _validate_contract(dataset: TrainingDataset) -> None:
    candidate_ids = [item.model_id for item in dataset.candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise TrainingDataError("Candidate model ids must be unique")
    if not candidate_ids:
        raise TrainingDataError("Training dataset must name at least one candidate model")
    if dataset.stronger_model_id not in set(candidate_ids):
        raise TrainingDataError("stronger_model_id must be one of the candidate models")
    prompt_ids = [item.prompt_id for item in dataset.prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise TrainingDataError("Prompt ids must be unique")
    if not dataset.prompts:
        raise TrainingDataError("Training dataset must contain at least one prompt")
    allowed = set(candidate_ids)
    for prompt in dataset.prompts:
        seen: set[str] = set()
        for outcome in prompt.outcomes:
            if outcome.model_id not in allowed:
                raise TrainingDataError(
                    f"{prompt.prompt_id} has an outcome for unknown model {outcome.model_id}"
                )
            if outcome.model_id in seen:
                raise TrainingDataError(
                    f"{prompt.prompt_id} repeats an outcome for {outcome.model_id}"
                )
            seen.add(outcome.model_id)
            if outcome.cost is not None:
                try:
                    from decimal import Decimal

                    if Decimal(outcome.cost) < 0:
                        raise TrainingDataError(f"{prompt.prompt_id} has a negative cost")
                except TrainingDataError:
                    raise
                except Exception:
                    raise TrainingDataError(
                        f"{prompt.prompt_id} has a non-decimal cost"
                    ) from None
