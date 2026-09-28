"""Deterministic model selection.

This chooses a configured model before any provider call. It does not check
whether a provider is reachable, and it does not rank models by price or
measured quality. Quality tiers are routing labels.

Preference order is a configured model-id list. Models named there come
first. Remaining models keep registry order, which is the stable tie-breaker.
"""

from collections.abc import Sequence

from app.api.schemas import ModelConfig, QualityTier
from app.routing.model_registry import ModelRegistry

_TIER_ORDER = (QualityTier.LOW, QualityTier.MEDIUM, QualityTier.HIGH)


class NoModelsAvailableError(LookupError):
    """Raised when routing has no enabled, configured model."""

    def __init__(self) -> None:
        super().__init__("No routing models are available")


class RoutingDecision:
    """The model selected for one complexity tier."""

    def __init__(
        self,
        *,
        requested_tier: QualityTier,
        model: ModelConfig,
        selection_reason: str,
        degraded: bool,
    ) -> None:
        self.requested_tier = requested_tier
        self.model = model
        self.selection_reason = selection_reason
        self.degraded = degraded


class ModelRouter:
    """Select one enabled model for a requested complexity tier."""

    def __init__(self, preference: Sequence[str] = ()) -> None:
        self._preference = tuple(preference)

    def select(self, requested_tier: QualityTier, registry: ModelRegistry) -> RoutingDecision:
        """Return one model. A lower tier than requested is marked degraded."""
        enabled = registry.list_enabled()
        if not enabled:
            raise NoModelsAvailableError()

        by_tier = {tier: [] for tier in _TIER_ORDER}
        for model in enabled:
            by_tier[model.quality_tier].append(model)

        higher = _higher_tiers(requested_tier)
        lower = _lower_tiers_highest_first(requested_tier)
        degraded = False
        if by_tier[requested_tier]:
            chosen_tier = requested_tier
            relation = f"the requested {requested_tier.value} tier"
        elif any(by_tier[tier] for tier in higher):
            chosen_tier = next(tier for tier in higher if by_tier[tier])
            relation = f"the nearest higher tier ({chosen_tier.value})"
        else:
            chosen_tier = next(tier for tier in lower if by_tier[tier])
            relation = f"the highest available lower tier ({chosen_tier.value})"
            degraded = True

        selected = _preferred(by_tier[chosen_tier], enabled, self._preference)
        if chosen_tier is requested_tier:
            reason = f"selected {selected.model_id} from {relation}"
        else:
            reason = (
                f"no {requested_tier.value}-tier model is enabled; "
                f"selected {selected.model_id} from {relation}"
            )
        return RoutingDecision(
            requested_tier=requested_tier,
            model=selected,
            selection_reason=reason,
            degraded=degraded,
        )


def parse_preference(value: str) -> tuple[str, ...]:
    """Split a comma-separated model-id preference list."""
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _higher_tiers(tier: QualityTier) -> tuple[QualityTier, ...]:
    index = _TIER_ORDER.index(tier)
    return _TIER_ORDER[index + 1 :]


def _lower_tiers_highest_first(tier: QualityTier) -> tuple[QualityTier, ...]:
    index = _TIER_ORDER.index(tier)
    return tuple(reversed(_TIER_ORDER[:index]))


def _preferred(
    candidates: list[ModelConfig],
    enabled_in_registration_order: list[ModelConfig],
    preference: tuple[str, ...],
) -> ModelConfig:
    registration_index = {
        model.model_id: index for index, model in enumerate(enabled_in_registration_order)
    }

    def sort_key(model: ModelConfig) -> tuple[int, int]:
        if model.model_id in preference:
            return (preference.index(model.model_id), registration_index[model.model_id])
        return (len(preference), registration_index[model.model_id])

    return sorted(candidates, key=sort_key)[0]
