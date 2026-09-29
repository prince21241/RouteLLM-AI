"""Decimal pricing. No provider or database calls."""

from decimal import Decimal

from app.api.schemas import Provider, ReportedUsage
from app.pricing.money import money_to_api
from app.pricing.service import (
    CostCompleteness,
    estimate_baseline,
    estimate_cost,
    lookup_prices,
    observed_input_volume,
    savings,
)
from app.routing.catalog import VerifiedModelMetadata


def _usage(**overrides: object) -> ReportedUsage:
    values: dict[str, object] = {"input_tokens": 0, "output_tokens": 0}
    values.update(overrides)
    return ReportedUsage(**values)  # type: ignore[arg-type]


def _prices(**overrides: object) -> VerifiedModelMetadata:
    values: dict[str, object] = {
        "model_name": "Example",
        "provider": Provider.OPENAI,
        "context_window": 1000,
        "input_cost_per_million_tokens": Decimal("0.05"),
        "output_cost_per_million_tokens": Decimal("0.40"),
        "local": False,
        "price_source_url": "https://example.test/prices",
        "price_verified_on": "2026-09-28",
        "api_cost_note": "test rates",
        "cached_input_per_million": Decimal("0.005"),
    }
    values.update(overrides)
    return VerifiedModelMetadata(**values)  # type: ignore[arg-type]


def test_exact_million_token_cost() -> None:
    result = estimate_cost(
        _usage(input_tokens=1_000_000, output_tokens=1_000_000, cached_input_tokens=0),
        _prices(),
    )

    assert result.input_cost == Decimal("0.05")
    assert result.output_cost == Decimal("0.40")
    assert result.total == Decimal("0.45")
    assert result.completeness is CostCompleteness.COMPLETE


def test_small_cost_keeps_fractional_cents() -> None:
    result = estimate_cost(
        _usage(input_tokens=12, output_tokens=0, cached_input_tokens=0),
        _prices(),
    )

    assert result.total == Decimal("0.0000006")
    assert money_to_api(result.total) == "0.0000006"


def test_cached_openai_tokens_are_not_billed_twice() -> None:
    usage = _usage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=250_000)
    prices = _prices(
        input_cost_per_million_tokens=Decimal("1"),
        cached_input_per_million=Decimal("0.10"),
        output_cost_per_million_tokens=Decimal("1"),
    )

    result = estimate_cost(usage, prices)

    assert result.total == Decimal("0.775")
    assert result.completeness is CostCompleteness.COMPLETE
    double_counted = Decimal("1") + Decimal("0.025")
    assert result.total != double_counted


def test_cache_writes_are_not_added_on_top_of_uncached_input() -> None:
    usage = _usage(
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=0,
        cache_write_input_tokens=1_000_000,
    )
    prices = _prices(
        input_cost_per_million_tokens=Decimal("1"),
        cached_input_per_million=Decimal("0.10"),
        output_cost_per_million_tokens=Decimal("0"),
    )

    result = estimate_cost(usage, prices)

    assert result.total == Decimal("1")


def test_reasoning_tokens_are_not_added_to_output() -> None:
    prices = _prices(output_cost_per_million_tokens=Decimal("0.40"))
    with_reasoning = estimate_cost(
        _usage(input_tokens=0, output_tokens=20, cached_input_tokens=0, reasoning_tokens=15),
        prices,
    )
    without_reasoning = estimate_cost(
        _usage(input_tokens=0, output_tokens=20, cached_input_tokens=0),
        prices,
    )

    assert with_reasoning.total == Decimal("0.000008")
    assert with_reasoning.total == without_reasoning.total


def test_missing_cache_detail_is_an_estimate_at_standard_rates() -> None:
    result = estimate_cost(_usage(input_tokens=1_000_000, output_tokens=0), _prices())

    assert result.total == Decimal("0.05")
    assert result.completeness is CostCompleteness.ESTIMATED


def test_missing_model_price_is_not_zero() -> None:
    result = estimate_cost(_usage(input_tokens=1_000_000, output_tokens=50), None)

    assert result.total is None
    assert result.completeness is CostCompleteness.UNKNOWN
    assert lookup_prices("not-a-real-model") is None
    assert estimate_cost(
        _usage(input_tokens=12, output_tokens=4),
        lookup_prices("not-a-real-model"),
    ).total is None


def test_ollama_zero_is_a_real_provider_api_cost() -> None:
    metadata = lookup_prices("llama3.2")
    assert metadata is not None

    result = estimate_cost(_usage(input_tokens=500, output_tokens=20), metadata)

    assert result.total == Decimal("0")
    assert result.completeness is CostCompleteness.COMPLETE
    assert "electricity" in metadata.api_cost_note


def test_anthropic_cache_categories_are_not_folded_into_input() -> None:
    usage = _usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_write_5m_tokens=1_000_000,
        cache_write_1h_tokens=0,
        cache_read_input_tokens=1_000_000,
        cache_write_input_tokens=1_000_000,
    )
    prices = _prices(
        provider=Provider.ANTHROPIC,
        input_cost_per_million_tokens=Decimal("3"),
        output_cost_per_million_tokens=Decimal("15"),
        cached_input_per_million=None,
        cache_read_per_million=Decimal("0.30"),
        cache_write_5m_per_million=Decimal("3.75"),
        cache_write_1h_per_million=Decimal("6"),
    )

    result = estimate_cost(usage, prices)

    assert result.total == Decimal("22.05")
    assert result.completeness is CostCompleteness.COMPLETE


def test_anthropic_aggregate_cache_write_is_estimated() -> None:
    usage = _usage(
        input_tokens=0,
        output_tokens=0,
        cache_write_input_tokens=1_000_000,
        cache_read_input_tokens=0,
    )
    prices = lookup_prices("claude-sonnet-4-6")
    assert prices is not None

    result = estimate_cost(usage, prices)

    assert result.total == Decimal("3.75")
    assert result.completeness is CostCompleteness.ESTIMATED


def test_baseline_negative_savings_and_zero_baseline() -> None:
    usage = _usage(input_tokens=1_000_000, output_tokens=1_000_000, cached_input_tokens=0)
    actual = _prices(
        input_cost_per_million_tokens=Decimal("5"),
        output_cost_per_million_tokens=Decimal("20"),
    )
    premium = _prices(
        provider=Provider.ANTHROPIC,
        input_cost_per_million_tokens=Decimal("3"),
        output_cost_per_million_tokens=Decimal("15"),
    )
    actual_cost = estimate_cost(usage, actual).total
    baseline = estimate_baseline(usage, Provider.OPENAI, premium)

    assert actual_cost == Decimal("25")
    assert baseline == Decimal("18")
    assert savings(baseline, actual_cost) == Decimal("-7")

    zero = lookup_prices("llama3.2")
    zero_baseline = estimate_baseline(usage, Provider.OPENAI, zero)
    assert zero_baseline == Decimal("0")
    assert savings(zero_baseline, actual_cost) == Decimal("-25")
    assert savings(Decimal("0"), Decimal("0")) == Decimal("0")


def test_savings_are_null_when_either_cost_is_unknown() -> None:
    assert savings(None, Decimal("1")) is None
    assert savings(Decimal("1"), None) is None
    assert savings(None, None) is None


def test_anthropic_baseline_volume_includes_reported_cache_tokens() -> None:
    usage = _usage(
        input_tokens=100,
        output_tokens=0,
        cache_read_input_tokens=50,
        cache_write_input_tokens=25,
    )

    assert observed_input_volume(usage, Provider.ANTHROPIC) == 175
    assert observed_input_volume(usage, Provider.OPENAI) == 100


def test_verified_prices_record_source_and_date() -> None:
    openai = lookup_prices("gpt-5-nano")
    anthropic = lookup_prices("claude-sonnet-4-6")
    assert openai is not None
    assert anthropic is not None

    assert openai.input_cost_per_million_tokens == Decimal("0.05")
    assert openai.cached_input_per_million == Decimal("0.005")
    assert openai.output_cost_per_million_tokens == Decimal("0.40")
    assert openai.price_verified_on == "2026-09-28"
    assert openai.price_source_url == "https://developers.openai.com/api/docs/models/gpt-5-nano"
    assert anthropic.input_cost_per_million_tokens == Decimal("3")
    assert anthropic.output_cost_per_million_tokens == Decimal("15")
    assert anthropic.cache_read_per_million == Decimal("0.30")
    assert anthropic.cache_write_5m_per_million == Decimal("3.75")
    assert anthropic.cache_write_1h_per_million == Decimal("6")
    assert anthropic.price_source_url == "https://platform.claude.com/docs/en/about-claude/pricing"
    assert money_to_api(Decimal("0.0500")) == "0.05"


def test_ollama_cloud_model_has_no_published_token_price() -> None:
    assert lookup_prices("gemma4:31b") is None
