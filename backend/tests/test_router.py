"""Router tests use fictional models. They do not call providers."""

from decimal import Decimal

import pytest

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.config import Settings
from app.routing.catalog import CatalogConfigurationError, build_catalog
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter, NoModelsAvailableError


def fictional(
    model_id: str,
    tier: QualityTier,
    *,
    enabled: bool = True,
    provider: Provider = Provider.OPENAI,
) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_name=model_id,
        model_id=model_id,
        quality_tier=tier,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        context_window=1024,
        enabled=enabled,
        local=False,
    )


def test_exact_tier_is_preferred() -> None:
    registry = ModelRegistry(
        [
            fictional("fictional-low", QualityTier.LOW),
            fictional("fictional-high", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ]
    )

    decision = ModelRouter().select(QualityTier.LOW, registry)

    assert decision.requested_tier is QualityTier.LOW
    assert decision.model.model_id == "fictional-low"
    assert decision.model.quality_tier is QualityTier.LOW
    assert decision.degraded is False
    assert "requested low tier" in decision.selection_reason


def test_nearest_higher_tier_is_used_when_the_requested_tier_is_missing() -> None:
    registry = ModelRegistry(
        [
            fictional("fictional-medium", QualityTier.MEDIUM),
            fictional("fictional-high", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ]
    )

    decision = ModelRouter().select(QualityTier.LOW, registry)

    assert decision.model.model_id == "fictional-medium"
    assert decision.model.quality_tier is QualityTier.MEDIUM
    assert decision.degraded is False
    assert "nearest higher tier (medium)" in decision.selection_reason
    assert decision.requested_tier is QualityTier.LOW


def test_lower_tier_selection_is_explicitly_degraded() -> None:
    registry = ModelRegistry(
        [
            fictional("fictional-low", QualityTier.LOW),
            fictional("fictional-medium", QualityTier.MEDIUM),
        ]
    )

    decision = ModelRouter().select(QualityTier.HIGH, registry)

    assert decision.requested_tier is QualityTier.HIGH
    assert decision.model.model_id == "fictional-medium"
    assert decision.model.quality_tier is QualityTier.MEDIUM
    assert decision.degraded is True
    assert "highest available lower tier (medium)" in decision.selection_reason
    assert "requested high tier" not in decision.selection_reason


def test_disabled_models_are_excluded() -> None:
    registry = ModelRegistry(
        [
            fictional("fictional-disabled-low", QualityTier.LOW, enabled=False),
            fictional("fictional-high", QualityTier.HIGH, provider=Provider.ANTHROPIC),
        ]
    )

    decision = ModelRouter().select(QualityTier.LOW, registry)

    assert decision.model.model_id == "fictional-high"
    assert decision.degraded is False


def test_preference_order_breaks_ties_inside_a_tier() -> None:
    registry = ModelRegistry(
        [
            fictional("fictional-b", QualityTier.LOW),
            fictional("fictional-a", QualityTier.LOW, provider=Provider.OLLAMA),
        ]
    )

    preferred = ModelRouter(preference=("fictional-a", "fictional-b")).select(
        QualityTier.LOW,
        registry,
    )
    registration_order = ModelRouter().select(QualityTier.LOW, registry)

    assert preferred.model.model_id == "fictional-a"
    assert registration_order.model.model_id == "fictional-b"


def test_unlisted_models_keep_registration_order() -> None:
    registry = ModelRegistry(
        [
            fictional("fictional-first", QualityTier.MEDIUM),
            fictional("fictional-second", QualityTier.MEDIUM, provider=Provider.ANTHROPIC),
        ]
    )

    decision = ModelRouter(preference=("fictional-other",)).select(QualityTier.MEDIUM, registry)

    assert decision.model.model_id == "fictional-first"


def test_no_available_models() -> None:
    registry = ModelRegistry(
        [fictional("fictional-disabled", QualityTier.HIGH, enabled=False)]
    )

    with pytest.raises(NoModelsAvailableError, match="No routing models are available"):
        ModelRouter().select(QualityTier.LOW, registry)


def test_catalog_keeps_unconfigured_and_disabled_models_out_of_routing() -> None:
    without_credentials = build_catalog(Settings(_env_file=None))
    configured = build_catalog(
        Settings(
            _env_file=None,
            openai_api_key="fictional-openai-key",
            anthropic_api_key="fictional-anthropic-key",
        )
    )

    assert without_credentials.list_enabled() == []
    assert without_credentials.get("gpt-5-nano").enabled is False
    assert [model.model_id for model in configured.list_enabled()] == ["gpt-5-nano"]
    assert configured.get("claude-sonnet-4-6").enabled is False
    assert configured.get("llama3.2").enabled is False
    assert configured.get("gpt-5-nano").quality_tier is QualityTier.LOW
    assert configured.get("gpt-5-nano").context_window == 400_000


def test_unknown_model_id_is_not_given_invented_metadata() -> None:
    with pytest.raises(CatalogConfigurationError, match="gpt-unknown"):
        build_catalog(Settings(_env_file=None, openai_model="gpt-unknown"))


def test_ollama_stays_disabled_until_its_routing_flag_is_set() -> None:
    registry = build_catalog(
        Settings(_env_file=None, ollama_routing_enabled=True, openai_routing_enabled=False)
    )

    enabled = registry.list_enabled()

    assert [model.model_id for model in enabled] == ["llama3.2"]
    assert enabled[0].local is True
    assert registry.get("gpt-5-nano").enabled is False
