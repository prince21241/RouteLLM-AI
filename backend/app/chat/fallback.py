"""Choose the next provider after a fallback-eligible failure.

Candidates come from the configured model-id order. A candidate must be a
different provider from the attempt that just failed, must be enabled, and
must not repeat a provider/model pair already called for this request.
"""

from dataclasses import dataclass

from app.api.schemas import ModelConfig, Provider
from app.routing.model_registry import ModelRegistry, UnknownModelError


@dataclass(frozen=True)
class SkippedCandidate:
    """A configured model that was not called, and why."""

    model_id: str
    reason: str


def select_fallback_candidates(
    order: tuple[str, ...],
    registry: ModelRegistry,
    *,
    used: set[tuple[str, str]],
    failed_provider: Provider,
    restricted_provider: Provider | None,
    limit: int,
) -> tuple[list[ModelConfig], list[SkippedCandidate]]:
    """Return up to ``limit`` eligible models and every skip reason."""
    chosen: list[ModelConfig] = []
    skipped: list[SkippedCandidate] = []
    if limit < 1:
        return chosen, skipped
    for model_id in order:
        if len(chosen) >= limit:
            break
        try:
            model = registry.get(model_id)
        except UnknownModelError:
            skipped.append(SkippedCandidate(model_id, "The model is not in the registry."))
            continue
        pair = (model.provider.value, model.model_id)
        if pair in used:
            skipped.append(
                SkippedCandidate(model_id, "This provider and model were already called.")
            )
            continue
        if not model.enabled:
            skipped.append(SkippedCandidate(model_id, "The model is not enabled."))
            continue
        if restricted_provider is not None and model.provider is not restricted_provider:
            skipped.append(SkippedCandidate(model_id, "The request restricts the provider."))
            continue
        if model.provider is failed_provider:
            skipped.append(
                SkippedCandidate(model_id, "Fallback requires a different provider.")
            )
            continue
        chosen.append(model)
    return chosen, skipped
