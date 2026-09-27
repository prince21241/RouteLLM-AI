"""In-memory registry tests using fictional model metadata."""

from decimal import Decimal

import pytest

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.config import Settings
from app.main import create_app
from app.routing.model_registry import (
    DuplicateModelError,
    ModelRegistry,
    UnknownModelError,
)


def fictional_model(**overrides: object) -> ModelConfig:
    """Build an obviously fictional model. These are not live products or prices."""
    payload: dict[str, object] = {
        "provider": Provider.OLLAMA,
        "model_name": "Fictional Local Small",
        "model_id": "fictional-local-small",
        "quality_tier": QualityTier.LOW,
        "input_cost_per_million_tokens": Decimal("0"),
        "output_cost_per_million_tokens": Decimal("0"),
        "context_window": 4096,
        "enabled": True,
        "local": True,
    }
    payload.update(overrides)
    return ModelConfig.model_validate(payload)


def test_empty_registry_lists_no_models() -> None:
    registry = ModelRegistry()

    assert registry.list_enabled() == []


def test_lookup_by_model_id() -> None:
    local_model = fictional_model()
    hosted_model = fictional_model(
        provider=Provider.ANTHROPIC,
        model_name="Fictional Hosted Large",
        model_id="fictional-hosted-large",
        quality_tier=QualityTier.HIGH,
        input_cost_per_million_tokens=Decimal("1.00"),
        output_cost_per_million_tokens=Decimal("2.00"),
        context_window=8192,
        local=False,
    )
    registry = ModelRegistry([local_model, hosted_model])

    assert registry.get("fictional-local-small") == local_model
    assert registry.get("fictional-hosted-large") == hosted_model


def test_unknown_model_id_is_reported() -> None:
    registry = ModelRegistry([fictional_model()])

    with pytest.raises(UnknownModelError, match="fictional-missing") as exc_info:
        registry.get("fictional-missing")

    assert exc_info.value.model_id == "fictional-missing"


def test_duplicate_model_id_is_rejected() -> None:
    original = fictional_model()
    duplicate = fictional_model(model_name="Fictional Local Small Copy")
    registry = ModelRegistry([original])

    with pytest.raises(DuplicateModelError, match="fictional-local-small") as exc_info:
        registry.register(duplicate)

    assert exc_info.value.model_id == "fictional-local-small"
    assert registry.get("fictional-local-small") == original


def test_duplicate_in_initial_models_is_rejected() -> None:
    with pytest.raises(DuplicateModelError):
        ModelRegistry(
            [
                fictional_model(),
                fictional_model(model_name="Fictional Duplicate"),
            ]
        )


def test_list_enabled_filters_and_keeps_registration_order() -> None:
    enabled_local = fictional_model()
    disabled_hosted = fictional_model(
        provider=Provider.OPENAI,
        model_name="Fictional Hosted Disabled",
        model_id="fictional-hosted-disabled",
        quality_tier=QualityTier.MEDIUM,
        input_cost_per_million_tokens=Decimal("3"),
        output_cost_per_million_tokens=Decimal("4"),
        enabled=False,
        local=False,
    )
    enabled_hosted = fictional_model(
        provider=Provider.ANTHROPIC,
        model_name="Fictional Hosted Medium",
        model_id="fictional-hosted-medium",
        quality_tier=QualityTier.MEDIUM,
        input_cost_per_million_tokens=Decimal("0.50"),
        output_cost_per_million_tokens=Decimal("1.50"),
        context_window=2048,
        local=False,
    )
    registry = ModelRegistry([enabled_local, disabled_hosted, enabled_hosted])

    assert registry.list_enabled() == [enabled_local, enabled_hosted]
    assert registry.get("fictional-hosted-disabled") == disabled_hosted


def test_registry_is_injectable(settings: Settings) -> None:
    registry = ModelRegistry([fictional_model()])
    app = create_app(settings=settings, registry=registry)

    assert app.state.registry is registry
    assert [model.model_id for model in app.state.registry.list_enabled()] == [
        "fictional-local-small"
    ]
