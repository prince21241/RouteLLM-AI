"""Validation tests for shared schemas."""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.api.schemas import LLMResponse, ModelConfig, Provider, QualityTier


def model_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": Provider.OLLAMA,
        "model_name": "Fictional Local Small",
        "model_id": "fictional-local-small",
        "quality_tier": QualityTier.LOW,
        "input_cost_per_million_tokens": Decimal("0.00"),
        "output_cost_per_million_tokens": Decimal("0.00"),
        "context_window": 4096,
        "enabled": True,
        "local": True,
    }
    payload.update(overrides)
    return payload


def response_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": Provider.OLLAMA,
        "model": "fictional-local-small",
        "content": "fictional reply",
        "input_tokens": 12,
        "output_tokens": 4,
        "latency_ms": 15.5,
        "estimated_cost": None,
    }
    payload.update(overrides)
    return payload


def test_model_config_accepts_decimal_costs() -> None:
    model = ModelConfig(**model_payload(input_cost_per_million_tokens="1.25"))

    assert model.provider is Provider.OLLAMA
    assert model.quality_tier is QualityTier.LOW
    assert model.input_cost_per_million_tokens == Decimal("1.25")
    assert isinstance(model.input_cost_per_million_tokens, Decimal)
    assert isinstance(model.output_cost_per_million_tokens, Decimal)


def test_model_config_rejects_invalid_values() -> None:
    invalid_payloads = [
        model_payload(provider="fictional-provider"),
        model_payload(quality_tier="ultra"),
        model_payload(model_id=""),
        model_payload(input_cost_per_million_tokens=Decimal("-0.01")),
        model_payload(output_cost_per_million_tokens=Decimal("-1")),
        model_payload(context_window=0),
        model_payload(context_window=-1),
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            ModelConfig(**payload)


def test_llm_response_allows_missing_cost() -> None:
    response = LLMResponse(**response_payload())

    assert response.estimated_cost is None
    assert response.input_tokens == 12
    assert response.latency_ms == 15.5


def test_llm_response_accepts_zero_cost_tokens_and_latency() -> None:
    response = LLMResponse(
        **response_payload(
            input_tokens=0,
            output_tokens=0,
            latency_ms=0,
            estimated_cost=Decimal("0"),
        )
    )

    assert response.estimated_cost == Decimal("0")
    assert isinstance(response.estimated_cost, Decimal)


def test_llm_response_rejects_negative_measurements() -> None:
    invalid_payloads = [
        response_payload(input_tokens=-1),
        response_payload(output_tokens=-1),
        response_payload(latency_ms=-0.1),
        response_payload(estimated_cost=Decimal("-0.01")),
        response_payload(model=""),
    ]

    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            LLMResponse(**payload)
