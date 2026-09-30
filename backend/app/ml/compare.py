"""Offline comparison of routing policies on held-out prompts.

Recorded per-model outcomes estimate what that model did on the prompt.
They do not include escalation, fallback, judge cost, or wall-clock latency.
"""

from __future__ import annotations

import math
from collections import Counter
from decimal import Decimal

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.ml.artifact import LoadedArtifact
from app.ml.labels import LabeledPrompt
from app.ml.schema import NO_ACCEPTABLE_MODEL, TrainingDataset
from app.ml.split import SplitAssignment
from app.pricing.money import money_to_api, quantize_money
from app.routing.complexity import ComplexityClassifier
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter

LIMITATIONS = (
    "Model-selection accuracy compares the ML class prediction with the "
    "supervised label. It is not the same as downstream answer quality.",
    "Downstream quality, cost, and latency are offline estimates from "
    "recorded single-model outcomes on the common comparable subset.",
    "Recorded generation cost omits escalation, fallback, and judge calls.",
    "Recorded generation latency is provider time for the selected model. "
    "It is not end-to-end wall-clock latency.",
    "These estimates are not measured production performance of the full "
    "chat path.",
    "Abstaining to the rule-based policy does not guarantee answer quality.",
)


def compare_policies(
    dataset: TrainingDataset,
    labels: list[LabeledPrompt] | tuple[LabeledPrompt, ...],
    assignment: SplitAssignment,
    artifact: LoadedArtifact,
    *,
    quality_tolerance: float | None = None,
    cost_objective: str | None = None,
) -> dict[str, object]:
    """Score rule-based, ML, and fixed-stronger policies without new calls."""
    held_out_ids = {
        prompt.prompt_id
        for prompt in assignment.prompts
        if assignment.group_splits[prompt.group_id] == "test"
    }
    prompts = [prompt for prompt in dataset.prompts if prompt.prompt_id in held_out_ids]
    registry = _registry(dataset)
    router = ModelRouter()
    classifier = ComplexityClassifier()
    label_by_id = {item.prompt_id: item for item in labels}
    policies = ("rule_based", "ml", "stronger")
    selections: dict[str, dict[str, str | None]] = {name: {} for name in policies}
    reasons: dict[str, dict[str, str | None]] = {name: {} for name in policies}
    for prompt in prompts:
        rule_model = _rule_model(prompt.text, classifier, router, registry)
        ml_model, ml_reason = _ml_model(prompt.text, artifact, rule_model, registry)
        selections["rule_based"][prompt.prompt_id] = rule_model
        selections["ml"][prompt.prompt_id] = ml_model
        selections["stronger"][prompt.prompt_id] = dataset.stronger_model_id
        reasons["rule_based"][prompt.prompt_id] = None
        reasons["ml"][prompt.prompt_id] = ml_reason
        reasons["stronger"][prompt.prompt_id] = None
    coverage = {
        name: _coverage(prompts, selections[name]) for name in policies
    }
    common_ids = [
        prompt.prompt_id
        for prompt in prompts
        if all(coverage[name]["covered"].get(prompt.prompt_id) for name in policies)
    ]
    common_prompts = [prompt for prompt in prompts if prompt.prompt_id in common_ids]
    policy_reports = {
        name: _policy_report(common_prompts, selections[name], coverage[name])
        for name in policies
    }
    return {
        "dataset_hash": assignment.dataset_hash,
        "seed": assignment.seed,
        "exploratory": assignment.exploratory,
        "exploratory_reason": assignment.exploratory_reason,
        "held_out_prompts": len(prompts),
        "common_subset_count": len(common_prompts),
        "common_subset_rule": (
            "A held-out prompt is in the common subset when the rule-based "
            "policy, the ML policy after its recorded fallback, and the fixed "
            "stronger model each have a succeeded outcome for the model that "
            "policy selected."
        ),
        "coverage": {
            name: {
                "covered": sum(1 for value in coverage[name]["covered"].values() if value),
                "uncovered": coverage[name]["uncovered"],
            }
            for name in policies
        },
        "policies": policy_reports,
        "model_selection": _selection_report(prompts, label_by_id, artifact),
        "by_category": _by_category(common_prompts, selections),
        "no_acceptable_model": _abstentions(prompts, label_by_id, artifact),
        "promotion": _promotion(
            policy_reports,
            exploratory=assignment.exploratory,
            quality_tolerance=quality_tolerance,
            cost_objective=cost_objective,
        ),
        "limitations": list(LIMITATIONS),
    }


def _registry(dataset: TrainingDataset) -> ModelRegistry:
    models = []
    for candidate in dataset.candidates:
        models.append(
            ModelConfig(
                provider=Provider(candidate.provider),
                model_name=candidate.model_id,
                model_id=candidate.model_id,
                quality_tier=QualityTier(candidate.quality_tier),
                input_cost_per_million_tokens=Decimal("0"),
                output_cost_per_million_tokens=Decimal("0"),
                context_window=1024,
                enabled=candidate.enabled,
                local=candidate.provider == "ollama",
            )
        )
    return ModelRegistry(models)


def _rule_model(
    text: str,
    classifier: ComplexityClassifier,
    router: ModelRouter,
    registry: ModelRegistry,
) -> str:
    assessment = classifier.classify(text, None, low=0.30, high=0.70)
    return router.select(assessment.tier, registry).model.model_id


def _ml_model(
    text: str,
    artifact: LoadedArtifact,
    rule_model: str,
    registry: ModelRegistry,
) -> tuple[str, str | None]:
    prediction = artifact.predict(text)
    enabled = {model.model_id for model in registry.list_enabled()}
    if prediction.label == NO_ACCEPTABLE_MODEL:
        return rule_model, "no_acceptable_model"
    if prediction.confidence < artifact.threshold:
        return rule_model, "low_confidence"
    if prediction.label not in enabled:
        return rule_model, "ineligible_model"
    return prediction.label, None


def _coverage(prompts: list, selected: dict[str, str | None]) -> dict[str, object]:
    covered: dict[str, bool] = {}
    uncovered = []
    for prompt in prompts:
        model_id = selected[prompt.prompt_id]
        outcome = next((item for item in prompt.outcomes if item.model_id == model_id), None)
        if outcome is None:
            covered[prompt.prompt_id] = False
            uncovered.append({"prompt_id": prompt.prompt_id, "reason": "missing_outcome", "model_id": model_id})
        elif outcome.status != "succeeded":
            covered[prompt.prompt_id] = False
            uncovered.append({"prompt_id": prompt.prompt_id, "reason": "provider_failure", "model_id": model_id})
        else:
            covered[prompt.prompt_id] = True
    return {"covered": covered, "uncovered": uncovered}


def _policy_report(prompts: list, selected: dict[str, str | None], coverage: dict[str, object]) -> dict[str, object]:
    verdicts = Counter()
    costs: list[Decimal] = []
    latencies: list[float] = []
    missing_cost = 0
    missing_latency = 0
    distribution: Counter[str] = Counter()
    for prompt in prompts:
        model_id = selected[prompt.prompt_id]
        distribution[model_id or ""] += 1
        outcome = next(item for item in prompt.outcomes if item.model_id == model_id)
        verdict = outcome.quality_verdict or "unknown"
        verdicts[verdict] += 1
        if outcome.cost is not None and outcome.cost_completeness in {"complete", "estimated"}:
            costs.append(Decimal(outcome.cost))
        else:
            missing_cost += 1
        if outcome.latency_ms is None:
            missing_latency += 1
        else:
            latencies.append(outcome.latency_ms)
    total = len(prompts)
    passes = verdicts.get("pass", 0)
    return {
        "count": total,
        "quality": {
            "pass": passes,
            "fail": verdicts.get("fail", 0),
            "unknown": verdicts.get("unknown", 0) + verdicts.get("error", 0),
            "pass_rate": None if total == 0 else passes / total,
            "pass_rate_interval_95": _wilson(passes, total),
        },
        "mean_recorded_generation_cost": _mean_money(costs),
        "cost_samples": len(costs),
        "cost_missing": missing_cost,
        "mean_recorded_generation_latency_ms": (
            None if not latencies else sum(latencies) / len(latencies)
        ),
        "latency_samples": len(latencies),
        "latency_missing": missing_latency,
        "selection_distribution": dict(distribution),
        "uncovered_outside_common_subset": coverage["uncovered"],
    }


def _selection_report(
    prompts: list,
    label_by_id: dict[str, LabeledPrompt],
    artifact: LoadedArtifact,
) -> dict[str, object]:
    truths = []
    predicted = []
    excluded = 0
    for prompt in prompts:
        label = label_by_id.get(prompt.prompt_id)
        if label is None or label.label is None:
            excluded += 1
            continue
        truths.append(label.label)
        predicted.append(artifact.predict(prompt.text).label)
    classes = sorted(set(truths) | set(predicted) | set(artifact.classes))
    matrix: list[list[int]] = []
    if truths:
        import warnings

        from sklearn.metrics import confusion_matrix

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found",
                category=UserWarning,
            )
            matrix = confusion_matrix(truths, predicted, labels=classes).tolist()
    correct = sum(int(left == right) for left, right in zip(truths, predicted, strict=True))
    return {
        "kind": "model_selection_accuracy",
        "count": len(truths),
        "excluded_unlabeled": excluded,
        "accuracy": None if not truths else correct / len(truths),
        "classes": classes,
        "confusion_matrix": matrix,
    }


def _by_category(prompts: list, selections: dict[str, dict[str, str | None]]) -> dict[str, object]:
    grouped: dict[str, list] = {}
    for prompt in prompts:
        grouped.setdefault(prompt.category, []).append(prompt)
    report = {}
    for category, members in sorted(grouped.items()):
        report[category] = {
            "count": len(members),
            "policies": {
                name: _pass_count(members, selections[name]) for name in selections
            },
        }
    return report


def _pass_count(prompts: list, selected: dict[str, str | None]) -> dict[str, object]:
    passes = 0
    for prompt in prompts:
        outcome = next(item for item in prompt.outcomes if item.model_id == selected[prompt.prompt_id])
        passes += int(outcome.quality_verdict == "pass")
    return {"pass": passes, "count": len(prompts)}


def _abstentions(
    prompts: list,
    label_by_id: dict[str, LabeledPrompt],
    artifact: LoadedArtifact,
) -> dict[str, object]:
    predicted = 0
    gold = 0
    for prompt in prompts:
        if artifact.predict(prompt.text).label == NO_ACCEPTABLE_MODEL:
            predicted += 1
        label = label_by_id.get(prompt.prompt_id)
        if label is not None and label.label == NO_ACCEPTABLE_MODEL:
            gold += 1
    return {
        "predicted": predicted,
        "gold_labels": gold,
        "note": (
            "A no-acceptable-model prediction abstains to the existing "
            "rule-based policy. That fallback does not guarantee quality."
        ),
    }


def _promotion(
    reports: dict[str, dict[str, object]],
    *,
    exploratory: bool,
    quality_tolerance: float | None,
    cost_objective: str | None,
) -> dict[str, object]:
    configured = quality_tolerance is not None and cost_objective == "min_mean_cost"
    if not configured:
        return {
            "winner": None,
            "reason": (
                "No quality tolerance and cost objective are configured. "
                "The comparison is reported without a winner."
            ),
        }
    if exploratory:
        return {
            "winner": None,
            "reason": "The dataset is exploratory, so no policy is promoted.",
        }
    ml_quality = reports["ml"]["quality"]
    strong_quality = reports["stronger"]["quality"]
    ml_cost = reports["ml"]["mean_recorded_generation_cost"]
    strong_cost = reports["stronger"]["mean_recorded_generation_cost"]
    if (
        ml_quality["pass_rate"] is None
        or strong_quality["pass_rate"] is None
        or ml_cost is None
        or strong_cost is None
        or ml_quality["unknown"]
    ):
        return {
            "winner": None,
            "reason": "Promotion requires known pass rates, known costs, and no unknown grades.",
        }
    if ml_quality["pass_rate"] + quality_tolerance < strong_quality["pass_rate"]:
        return {"winner": None, "reason": "ML pass rate is outside the quality tolerance."}
    if Decimal(str(ml_cost)) >= Decimal(str(strong_cost)):
        return {"winner": None, "reason": "ML mean recorded cost is not lower."}
    return {
        "winner": "ml",
        "reason": "ML met the configured quality tolerance at a lower recorded generation cost.",
    }


def _mean_money(values: list[Decimal]) -> str | None:
    if not values:
        return None
    average = quantize_money(sum(values, Decimal("0")) / Decimal(len(values)))
    return money_to_api(average)


def _wilson(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.96
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]
