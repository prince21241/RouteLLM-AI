"""Optional ML selection in front of the deterministic router.

The model is loaded once. Inference runs off the event loop. Low
confidence, an unavailable artifact, an ineligible model, or a
no-acceptable-model prediction all keep the existing rule-based choice.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.api.schemas import QualityTier
from app.config import Settings
from app.ml.artifact import ArtifactError, LoadedArtifact, Prediction, load_trusted_artifact
from app.ml.schema import NO_ACCEPTABLE_MODEL
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter, RoutingDecision

_RANK = {QualityTier.LOW: 0, QualityTier.MEDIUM: 1, QualityTier.HIGH: 2}


@dataclass(frozen=True)
class RoutingTrace:
    """What was requested, what ran, and why rules were used."""

    requested_strategy: str
    effective_strategy: str
    artifact_version: str | None
    predicted_model: str | None
    confidence: float | None
    rules_used: bool
    rules_reason: str | None
    initial_model_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_strategy": self.requested_strategy,
            "effective_strategy": self.effective_strategy,
            "artifact_version": self.artifact_version,
            "predicted_model": self.predicted_model,
            "confidence": self.confidence,
            "rules_used": self.rules_used,
            "rules_reason": self.rules_reason,
            "initial_model_id": self.initial_model_id,
        }


class RouteSelector:
    """Choose the initial model and record the routing metadata."""

    def __init__(
        self,
        rules: ModelRouter,
        settings: Settings,
        artifact: LoadedArtifact | None = None,
    ) -> None:
        self._rules = rules
        self._requested = settings.routing_strategy
        self._artifact: LoadedArtifact | None = None
        self._diagnostic: str | None = None
        if self._requested != "ml":
            return
        if artifact is not None:
            self._artifact = artifact
            return
        self._load(settings)

    @property
    def requested_strategy(self) -> str:
        return self._requested

    @property
    def effective_strategy(self) -> str:
        if self._artifact is None:
            return "rule_based"
        return "ml"

    @property
    def diagnostic(self) -> str | None:
        return self._diagnostic

    async def select(
        self,
        prompt: str,
        tier: QualityTier,
        registry: ModelRegistry,
    ) -> RoutingDecision:
        """Return one enabled model. ML never bypasses an empty registry."""
        rules = self._rules.select(tier, registry)
        if self._artifact is None:
            reason = self._diagnostic or "routing strategy is rule_based"
            return _with_trace(
                rules,
                _trace(
                    requested=self._requested,
                    effective="rule_based",
                    artifact=None,
                    prediction=None,
                    rules_used=True,
                    rules_reason=reason,
                    model_id=rules.model.model_id,
                ),
            )
        prediction = await asyncio.to_thread(self._artifact.predict, prompt)
        enabled = {model.model_id for model in registry.list_enabled()}
        if prediction.label == NO_ACCEPTABLE_MODEL:
            return _with_trace(
                rules,
                _trace(
                    requested="ml",
                    effective="rule_based",
                    artifact=self._artifact.version,
                    prediction=prediction,
                    rules_used=True,
                    rules_reason=(
                        "ml abstained because no acceptable model was predicted; "
                        "the rule-based policy selected the initial model. "
                        "This fallback does not guarantee answer quality."
                    ),
                    model_id=rules.model.model_id,
                ),
            )
        if prediction.confidence < self._artifact.threshold:
            return _with_trace(
                rules,
                _trace(
                    requested="ml",
                    effective="rule_based",
                    artifact=self._artifact.version,
                    prediction=prediction,
                    rules_used=True,
                    rules_reason=(
                        f"ml confidence {prediction.confidence:.3f} is below the "
                        f"validation threshold {self._artifact.threshold:.3f}; "
                        "the rule-based policy selected the initial model."
                    ),
                    model_id=rules.model.model_id,
                ),
            )
        if prediction.label not in enabled:
            return _with_trace(
                rules,
                _trace(
                    requested="ml",
                    effective="rule_based",
                    artifact=self._artifact.version,
                    prediction=prediction,
                    rules_used=True,
                    rules_reason=(
                        f"ml predicted {prediction.label}, which is not eligible "
                        "for this request; the rule-based policy selected the initial model."
                    ),
                    model_id=rules.model.model_id,
                ),
            )
        model = registry.get(prediction.label)
        degraded = _RANK[model.quality_tier] < _RANK[tier]
        decision = RoutingDecision(
            requested_tier=tier,
            model=model,
            selection_reason=(
                f"ml selected {model.model_id} with confidence {prediction.confidence:.3f}"
            ),
            degraded=degraded,
        )
        return _with_trace(
            decision,
            _trace(
                requested="ml",
                effective="ml",
                artifact=self._artifact.version,
                prediction=prediction,
                rules_used=False,
                rules_reason=None,
                model_id=model.model_id,
            ),
        )

    def _load(self, settings: Settings) -> None:
        if not settings.ml_artifact_path.strip():
            self._diagnostic = "ML artifact path is empty; rule-based routing remains in effect."
            return
        try:
            self._artifact = load_trusted_artifact(
                Path(settings.ml_artifact_path),
                trusted_root=trusted_ml_root(settings),
            )
        except ArtifactError as exc:
            self._diagnostic = f"{exc} Rule-based routing remains in effect."
        except Exception as exc:
            self._diagnostic = (
                f"ML artifact could not be loaded ({type(exc).__name__}). "
                "Rule-based routing remains in effect."
            )


def trusted_ml_root(settings: Settings) -> Path:
    """Return the only directory from which an ML artifact may be loaded."""
    if settings.ml_trusted_root.strip():
        return Path(settings.ml_trusted_root).expanduser()
    return Path(__file__).resolve().parents[2] / "var" / "ml"


def _trace(
    *,
    requested: str,
    effective: str,
    artifact: str | None,
    prediction: Prediction | None,
    rules_used: bool,
    rules_reason: str | None,
    model_id: str,
) -> RoutingTrace:
    return RoutingTrace(
        requested_strategy=requested,
        effective_strategy=effective,
        artifact_version=artifact,
        predicted_model=None if prediction is None else prediction.label,
        confidence=None if prediction is None else prediction.confidence,
        rules_used=rules_used,
        rules_reason=rules_reason,
        initial_model_id=model_id,
    )


def _with_trace(decision: RoutingDecision, trace: RoutingTrace) -> RoutingDecision:
    decision.trace = trace
    return decision
