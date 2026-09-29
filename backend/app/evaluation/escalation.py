"""Choose at most one stronger model.

The choice is an explicit model id or the next higher configured quality
tier. Price is not a quality signal. The same model is never selected.
"""

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.config import Settings
from app.routing.model_registry import ModelRegistry, UnknownModelError
from app.routing.router import _TIER_ORDER, _higher_tiers, _preferred

_NO_STRONGER = "No stronger eligible model is available."
_SAME_MODEL = "The escalation model is the same as the selected model."


class EscalationTarget:
    """The escalation decision. ``model`` is set only when a call is allowed."""

    def __init__(self, model: ModelConfig | None, reason: str) -> None:
        self.model = model
        self.reason = reason


def select_escalation_target(
    current: ModelConfig,
    registry: ModelRegistry,
    settings: Settings,
) -> EscalationTarget:
    """Return one different model, or a reason that no call should be made."""
    configured = settings.escalation_model.strip()
    if configured:
        return _configured_target(current, registry, settings, configured)
    target = _tier_target(current, registry, settings.routing_preference)
    if target is None:
        return EscalationTarget(None, _NO_STRONGER)
    return target


def _configured_target(
    current: ModelConfig,
    registry: ModelRegistry,
    settings: Settings,
    model_id: str,
) -> EscalationTarget:
    if model_id == current.model_id:
        return EscalationTarget(None, _SAME_MODEL)
    try:
        model = registry.get(model_id)
    except UnknownModelError:
        if settings.openai_api_key is None:
            return EscalationTarget(None, _NO_STRONGER)
        return EscalationTarget(
            _openai_target(model_id),
            f"configured escalation model {model_id} via OpenAI",
        )
    if not model.enabled or model.model_id == current.model_id:
        return EscalationTarget(None, _NO_STRONGER)
    return EscalationTarget(model, f"configured escalation model {model.model_id}")


def _tier_target(
    current: ModelConfig,
    registry: ModelRegistry,
    preference: str,
) -> EscalationTarget | None:
    """Pick an enabled model in a higher configured tier.

    Quality tiers are routing labels. A higher price is not used.
    """
    from app.routing.router import parse_preference

    enabled = [
        model
        for model in registry.list_enabled()
        if model.model_id != current.model_id and model.quality_tier in _higher_tiers(current.quality_tier)
    ]
    if not enabled:
        return None
    by_tier: dict[QualityTier, list[ModelConfig]] = {tier: [] for tier in _TIER_ORDER}
    for model in enabled:
        by_tier[model.quality_tier].append(model)
    chosen_tier = next(tier for tier in _higher_tiers(current.quality_tier) if by_tier[tier])
    selected = _preferred(by_tier[chosen_tier], registry.list_enabled(), parse_preference(preference))
    if selected.model_id == current.model_id:
        return None
    return EscalationTarget(
        selected,
        (
            f"selected {selected.model_id} from the next higher configured "
            f"quality tier ({chosen_tier.value})"
        ),
    )


def _openai_target(model_id: str) -> ModelConfig:
    """An explicitly configured OpenAI id that is not in the price book.

    The zero rates on this object are not used. Pricing looks up the verified
    book and leaves an unknown id unknown.
    """
    from decimal import Decimal

    return ModelConfig(
        provider=Provider.OPENAI,
        model_name=model_id,
        model_id=model_id,
        quality_tier=QualityTier.HIGH,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        context_window=1,
        enabled=True,
        local=False,
    )


def same_model_reason() -> str:
    return _SAME_MODEL


def missing_model_reason() -> str:
    return _NO_STRONGER
