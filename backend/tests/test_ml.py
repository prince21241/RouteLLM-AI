"""Mechanics for the optional ML router.

The datasets in this file are synthetic fixtures. They are not evaluation
evidence and must not be used to claim that ML routing is better.
"""

from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.config import Settings
from app.evaluation.dataset import DATASET_PATH
from app.main import create_app
from app.ml.artifact import ArtifactError, Prediction, load_trusted_artifact, save_artifact
from app.ml.compare import compare_policies
from app.ml.labels import label_dataset
from app.ml.runtime import RouteSelector
from app.ml.schema import (
    NO_ACCEPTABLE_MODEL,
    TrainingDataError,
    TrainingDataset,
    load_training_dataset,
)
from app.ml.split import SplitError, assign_splits
from app.ml.train import fit_router
from app.ml.__main__ import main
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter
from app.ml.schema import dataset_hash
from tests.test_chat import PROMPT, RecordingProvider
from tests.test_lifecycle import MemoryStore


def _outcome(model_id: str, provider: str, verdict: str | None, cost: str | None, **overrides: object) -> dict:
    row = {
        "model_id": model_id,
        "provider": provider,
        "status": "succeeded",
        "quality_verdict": verdict,
        "quality_score": None,
        "cost": cost,
        "cost_completeness": "complete" if cost is not None else "unknown",
        "latency_ms": 10,
    }
    row.update(overrides)
    return row


def _prompt(prompt_id: str, group_id: str, text: str, outcomes: list[dict]) -> dict:
    return {
        "prompt_id": prompt_id,
        "text": text,
        "category": "factual",
        "group_id": group_id,
        "outcomes": outcomes,
    }


def _dataset(prompts: list[dict], **overrides: object) -> TrainingDataset:
    payload = {
        "format": "routellm-training-1",
        "version": "fixture-1",
        "provenance": "live",
        "evidence": "fixture",
        "rubric_version": "zzrubriczz",
        "quality_threshold": 0.75,
        "stronger_model_id": "strong-model",
        "candidates": [
            {"model_id": "cheap-model", "provider": "openai", "quality_tier": "low"},
            {"model_id": "strong-model", "provider": "anthropic", "quality_tier": "high"},
        ],
        "prompts": prompts,
    }
    payload.update(overrides)
    return TrainingDataset.model_validate(payload)


def _separable_dataset() -> TrainingDataset:
    prompts = []
    families = {
        "cheap-model": "refund receipt invoice",
        "strong-model": "theorem lemma proof",
        NO_ACCEPTABLE_MODEL: "refuse hazard unsafe",
    }
    for label, words in families.items():
        for index in range(4):
            if label == "cheap-model":
                outcomes = [
                    _outcome("cheap-model", "openai", "pass", "0.01"),
                    _outcome("strong-model", "anthropic", "pass", "0.20"),
                ]
            elif label == "strong-model":
                outcomes = [
                    _outcome("cheap-model", "openai", "fail", "0.01"),
                    _outcome("strong-model", "anthropic", "pass", "0.20"),
                ]
            else:
                outcomes = [
                    _outcome("cheap-model", "openai", "fail", "0.01"),
                    _outcome("strong-model", "anthropic", "fail", "0.20"),
                ]
            prompts.append(
                _prompt(
                    f"{label}-{index}",
                    f"{label}-group-{index}",
                    f"{words} example {index}",
                    outcomes,
                )
            )
    prompts.append(
        _prompt(
            "cheap-model-copy",
            "cheap-model-group-0",
            "refund receipt invoice near duplicate",
            [
                _outcome("cheap-model", "openai", "pass", "0.01"),
                _outcome("strong-model", "anthropic", "pass", "0.20"),
            ],
        )
    )
    return _dataset(prompts)


def test_cheapest_passing_model_is_the_label() -> None:
    dataset = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refund receipt",
                [
                    _outcome("cheap-model", "openai", "pass", "0.02"),
                    _outcome("strong-model", "anthropic", "pass", "0.20"),
                ],
            )
        ]
    )
    report = label_dataset(dataset)
    assert report.prompts[0].label == "cheap-model"
    assert report.excluded == 0


def test_all_definitive_failures_are_no_acceptable_model() -> None:
    dataset = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refuse hazard",
                [
                    _outcome("cheap-model", "openai", "fail", "0.02"),
                    _outcome("strong-model", "anthropic", "fail", "0.20"),
                ],
            )
        ]
    )
    assert label_dataset(dataset).prompts[0].label == NO_ACCEPTABLE_MODEL


def test_unknown_missing_and_incomplete_cost_are_excluded() -> None:
    unknown = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refund receipt",
                [
                    _outcome("cheap-model", "openai", "unknown", "0.02"),
                    _outcome("strong-model", "anthropic", "pass", "0.20"),
                ],
            )
        ]
    )
    missing = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refund receipt",
                [_outcome("cheap-model", "openai", "pass", "0.02")],
            )
        ]
    )
    failed = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refund receipt",
                [
                    _outcome("cheap-model", "openai", None, None, status="failed", cost_completeness=None),
                    _outcome("strong-model", "anthropic", "pass", "0.20"),
                ],
            )
        ]
    )
    incomplete = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refund receipt",
                [
                    _outcome("cheap-model", "openai", "pass", None, cost_completeness="unknown"),
                    _outcome("strong-model", "anthropic", "fail", "0.20"),
                ],
            )
        ]
    )
    judge_error = _dataset(
        [
            _prompt(
                "p",
                "g",
                "refund receipt",
                [
                    _outcome("cheap-model", "openai", "error", "0.02"),
                    _outcome("strong-model", "anthropic", "pass", "0.20"),
                ],
            )
        ]
    )
    assert label_dataset(unknown).prompts[0].exclusion_reason == "unknown_grade"
    assert label_dataset(judge_error).prompts[0].exclusion_reason == "unknown_grade"
    assert label_dataset(judge_error).prompts[0].label is None
    assert label_dataset(judge_error).label_counts == {}
    assert label_dataset(missing).prompts[0].exclusion_reason == "missing_evaluation"
    assert label_dataset(failed).prompts[0].exclusion_reason == "provider_failure"
    assert label_dataset(incomplete).prompts[0].exclusion_reason == "incomplete_cost"
    assert all(item.label is None for item in label_dataset(unknown).prompts)


def test_mock_and_smoke_evaluation_are_not_training_evidence(tmp_path: Path) -> None:
    mock = _dataset(
        [_prompt("p", "g", "refund receipt", [_outcome("cheap-model", "openai", "pass", "0.01"), _outcome("strong-model", "anthropic", "fail", "0.2")])],
        provenance="mock",
    )
    with pytest.raises(ValueError, match="Mock answers"):
        label_dataset(mock)
    with pytest.raises(TrainingDataError, match="not ML training evidence"):
        load_training_dataset(DATASET_PATH)
    leaked = _dataset(
        [_prompt("p", "g", "refund receipt", [_outcome("cheap-model", "openai", "pass", "0.01"), _outcome("strong-model", "anthropic", "pass", "0.2")])]
    ).model_dump(mode="json")
    leaked["prompts"][0]["expected"] = "reference answer"
    path = tmp_path / "leaked.json"
    path.write_text(__import__("json").dumps(leaked), encoding="utf-8")
    with pytest.raises(TrainingDataError):
        load_training_dataset(path)


def test_group_split_has_no_family_leakage_and_rejects_missing_classes() -> None:
    dataset = _separable_dataset()
    report = label_dataset(dataset)
    assignment = assign_splits(report.prompts, seed=17, dataset_hash="hash")
    assert assignment.group_splits["cheap-model-group-0"] == assignment.split_for("cheap-model-0")
    assert assignment.split_for("cheap-model-copy") == assignment.split_for("cheap-model-0")
    assert assignment.exploratory is True
    one_class = []
    for index in range(9):
        one_class.append(
            report.prompts[0].__class__(
                prompt_id=f"only-{index}",
                group_id=f"only-{index}",
                text=f"refund receipt invoice {index}",
                category="factual",
                label="cheap-model",
                exclusion_reason=None,
            )
        )
    with pytest.raises(SplitError, match="at least two label classes"):
        assign_splits(one_class, seed=17, dataset_hash="hash")


def test_training_is_reproducible_and_ignores_rubric_text() -> None:
    dataset = _separable_dataset()
    report = label_dataset(dataset)
    digest = dataset_hash(dataset)
    first = fit_router(assign_splits(report.prompts, seed=17, dataset_hash=digest), seed=17)
    second = fit_router(assign_splits(report.prompts, seed=17, dataset_hash=digest), seed=17)
    text = "refund receipt invoice example 0"
    assert first.pipeline.predict([text])[0] == second.pipeline.predict([text])[0]
    assert first.threshold == second.threshold
    vocabulary = first.pipeline.named_steps["tfidf"].vocabulary_
    assert "zzrubriczz" not in vocabulary
    assert "exploratory" in first.validation_metrics


def test_artifact_round_trip_and_rejection(tmp_path: Path) -> None:
    dataset = _separable_dataset()
    report = label_dataset(dataset)
    digest = dataset_hash(dataset)
    trained = fit_router(assign_splits(report.prompts, seed=17, dataset_hash=digest), seed=17)
    trusted = tmp_path / "trusted"
    model_path = save_artifact(
        trusted,
        trained=trained,
        dataset_hash=digest,
        split={"seed": 17},
        seed=17,
        supported_model_ids=("cheap-model", "strong-model"),
        evidence="measured",
        provenance="live",
    )
    loaded = load_trusted_artifact(model_path, trusted_root=trusted)
    assert loaded.predict("theorem lemma proof example 1").label == "strong-model"
    fixture = save_artifact(
        tmp_path / "fixture",
        trained=trained,
        dataset_hash=digest,
        split={"seed": 17},
        seed=17,
        supported_model_ids=("cheap-model", "strong-model"),
        evidence="fixture",
        provenance="live",
    )
    with pytest.raises(ArtifactError, match="Fixture artifacts"):
        load_trusted_artifact(fixture, trusted_root=tmp_path / "fixture")
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "model.joblib"
    external.write_bytes(b"nope")
    with pytest.raises(ArtifactError, match="outside the trusted directory"):
        load_trusted_artifact(external, trusted_root=trusted)
    bad = trusted / "model.metadata.json"
    metadata = __import__("json").loads(bad.read_text(encoding="utf-8"))
    metadata["library_versions"]["scikit-learn"] = "0.1.0"
    bad.write_text(__import__("json").dumps(metadata), encoding="utf-8")
    with pytest.raises(ArtifactError, match="incompatible"):
        load_trusted_artifact(model_path, trusted_root=trusted)


def test_cli_refuses_fixture_training(tmp_path: Path) -> None:
    dataset = _separable_dataset()
    path = tmp_path / "fixture.json"
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    assert main(["validate", "--dataset", str(path)]) == 0
    assert main(["train", "--dataset", str(path), "--output-dir", str(tmp_path / "out")]) == 2


def test_low_confidence_ineligible_and_abstention_use_rules() -> None:
    registry = _registry()
    rules = ModelRouter()
    settings = Settings(_env_file=None, routing_strategy="ml")
    low = RouteSelector(rules, settings, artifact=_scripted("cheap-model", 0.2))
    decision = _select(low, registry)
    assert decision.model.model_id == rules.select(decision.requested_tier, registry).model.model_id
    assert decision.trace.rules_used is True
    assert "below the validation threshold" in decision.trace.rules_reason
    ineligible = RouteSelector(rules, settings, artifact=_scripted("missing-model", 0.99))
    missed = _select(ineligible, registry)
    assert missed.trace.rules_used is True
    assert "not eligible" in missed.trace.rules_reason
    abstain = RouteSelector(rules, settings, artifact=_scripted(NO_ACCEPTABLE_MODEL, 0.99))
    held = _select(abstain, registry)
    assert held.trace.rules_used is True
    assert "does not guarantee answer quality" in held.trace.rules_reason


def test_provider_restriction_blocks_the_ml_model() -> None:
    registry = _registry()
    rules = ModelRouter()
    selector = RouteSelector(
        rules,
        Settings(_env_file=None, routing_strategy="ml"),
        artifact=_scripted("strong-model", 0.99),
    )
    restricted = ModelRegistry(
        [model for model in registry.list_all() if model.provider is Provider.OPENAI]
    )
    decision = _select(selector, restricted)
    assert decision.model.model_id == "cheap-model"
    assert decision.trace.rules_used is True
    unrestricted = _select(selector, registry)
    assert unrestricted.model.model_id == "strong-model"
    assert unrestricted.trace.effective_strategy == "ml"
    assert unrestricted.trace.rules_used is False


def test_default_chat_stays_rule_based_and_stores_metadata() -> None:
    provider = RecordingProvider()
    store = MemoryStore()
    registry = ModelRegistry(
        [
            ModelConfig(
                provider=Provider.OPENAI,
                model_name="GPT-5 nano",
                model_id="gpt-5-nano",
                quality_tier=QualityTier.LOW,
                input_cost_per_million_tokens=Decimal("0"),
                output_cost_per_million_tokens=Decimal("0"),
                context_window=1024,
                enabled=True,
                local=False,
            )
        ]
    )
    with TestClient(
        create_app(
            settings=Settings(_env_file=None),
            registry=registry,
            provider_factory=lambda _model: provider,
            store=store,  # type: ignore[arg-type]
        )
    ) as client:
        response = client.post("/api/v1/chat", json={"prompt": PROMPT})
        options = client.get("/api/v1/chat/options")
    body = response.json()
    assert response.status_code == 200
    assert body["routing"]["routing_metadata"]["requested_strategy"] == "rule_based"
    assert body["routing"]["routing_metadata"]["effective_strategy"] == "rule_based"
    assert body["routing"]["model"] == "gpt-5-nano"
    assert store.pending[0].routing_metadata["rules_used"] is True
    assert options.json()["routing_strategy"] == "rule_based"
    assert options.json()["effective_routing_strategy"] == "rule_based"


def test_comparison_reports_missing_outcomes_and_does_not_pick_a_winner() -> None:
    prompts = [
        _prompt(
            "covered",
            "covered-group",
            "refund receipt invoice",
            [
                _outcome("cheap-model", "openai", "pass", "0.01", latency_ms=5),
                _outcome("strong-model", "anthropic", "pass", "0.20", latency_ms=8),
            ],
        ),
        _prompt(
            "missing-strong",
            "missing-group",
            "refund receipt invoice other",
            [_outcome("cheap-model", "openai", "pass", "0.01", latency_ms=5)],
        ),
    ]
    dataset = _dataset(prompts)
    labels = label_dataset(dataset).prompts
    from app.ml.split import SplitAssignment

    assignment = SplitAssignment(
        seed=1,
        dataset_hash="fixture-hash",
        exploratory=False,
        exploratory_reason=None,
        group_splits={"covered-group": "test", "missing-group": "test"},
        prompts=tuple(labels),
    )
    report = compare_policies(dataset, labels, assignment, _scripted("cheap-model", 0.99))
    assert report["held_out_prompts"] == 2
    assert report["common_subset_count"] == 1
    assert report["coverage"]["stronger"]["uncovered"][0]["reason"] == "missing_outcome"
    assert report["promotion"]["winner"] is None
    assert report["model_selection"]["kind"] == "model_selection_accuracy"
    assert any("not end-to-end" in item for item in report["limitations"])


class _Scripted:
    def __init__(self, label: str, confidence: float) -> None:
        self.version = "scripted"
        self.threshold = 0.8
        self.classes = (label,)
        self._label = label
        self._confidence = confidence

    def predict(self, text: str) -> Prediction:
        return Prediction(self._label, self._confidence)


def _scripted(label: str, confidence: float) -> _Scripted:
    return _Scripted(label, confidence)


def _registry() -> ModelRegistry:
    return ModelRegistry(
        [
            ModelConfig(
                provider=Provider.OPENAI,
                model_name="cheap",
                model_id="cheap-model",
                quality_tier=QualityTier.LOW,
                input_cost_per_million_tokens=Decimal("0"),
                output_cost_per_million_tokens=Decimal("0"),
                context_window=1024,
                enabled=True,
                local=False,
            ),
            ModelConfig(
                provider=Provider.ANTHROPIC,
                model_name="strong",
                model_id="strong-model",
                quality_tier=QualityTier.HIGH,
                input_cost_per_million_tokens=Decimal("0"),
                output_cost_per_million_tokens=Decimal("0"),
                context_window=1024,
                enabled=True,
                local=False,
            ),
        ]
    )


def _select(selector: RouteSelector, registry: ModelRegistry):
    import asyncio

    return asyncio.run(selector.select("refund receipt invoice", QualityTier.LOW, registry))
