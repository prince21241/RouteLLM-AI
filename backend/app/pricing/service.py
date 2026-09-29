"""Estimate request cost from the verified price book.

This module does not call a provider. Missing prices stay missing. A
published zero, such as local Ollama's API price, is a real zero.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from app.api.schemas import LLMResponse, Provider, ReportedUsage
from app.pricing.money import decimal_text, line_cost, quantize_money
from app.routing.catalog import VerifiedModelMetadata, verified_metadata


class CostCompleteness(StrEnum):
    """Whether every billed category was priced."""

    UNKNOWN = "unknown"
    ESTIMATED = "estimated"
    COMPLETE = "complete"


@dataclass(frozen=True)
class CostEstimate:
    """One cost calculation.

    ``total`` is ``None`` when the cost is unknown. It is never coerced to
    zero in that case.
    """

    total: Decimal | None
    completeness: CostCompleteness
    input_cost: Decimal | None = None
    output_cost: Decimal | None = None
    cached_input_cost: Decimal | None = None
    cache_write_cost: Decimal | None = None
    cache_read_cost: Decimal | None = None


def estimate_cost(usage: ReportedUsage, metadata: VerifiedModelMetadata | None) -> CostEstimate:
    """Price ``usage`` with ``metadata``.

    An unknown model returns an unknown cost. Cache categories that the
    provider did not report are not invented. When only the standard input
    and output counts are present, the result is an estimate at standard
    rates. Reasoning tokens are ignored because they are already inside
    ``output_tokens`` when a provider reports them there.
    """
    if metadata is None:
        return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
    if metadata.local and _is_zero(metadata):
        total = quantize_money(
            line_cost(usage.input_tokens, metadata.input_cost_per_million_tokens)
            + line_cost(usage.output_tokens, metadata.output_cost_per_million_tokens)
        )
        return CostEstimate(
            total=total,
            completeness=CostCompleteness.COMPLETE,
            input_cost=quantize_money(
                line_cost(usage.input_tokens, metadata.input_cost_per_million_tokens)
            ),
            output_cost=quantize_money(
                line_cost(usage.output_tokens, metadata.output_cost_per_million_tokens)
            ),
        )
    if metadata.provider is Provider.ANTHROPIC:
        return _anthropic_cost(usage, metadata)
    if metadata.provider is Provider.OPENAI:
        return _openai_cost(usage, metadata)
    return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)


def estimate_baseline(
    usage: ReportedUsage,
    source_provider: Provider,
    baseline: VerifiedModelMetadata | None,
) -> Decimal | None:
    """Estimate the premium model's cost at the observed token volume.

    The baseline uses standard input and output rates. It does not apply
    cache discounts and it does not call the premium model. OpenAI cached
    tokens are already inside ``input_tokens``, so they are not added again.
    Anthropic cache tokens are outside ``input_tokens``, so reported cache
    writes and reads are added to the baseline input volume.
    """
    if baseline is None:
        return None
    volume = observed_input_volume(usage, source_provider)
    total = line_cost(volume, baseline.input_cost_per_million_tokens) + line_cost(
        usage.output_tokens,
        baseline.output_cost_per_million_tokens,
    )
    return quantize_money(total)


def savings(baseline_cost: Decimal | None, actual_cost: Decimal | None) -> Decimal | None:
    """Return ``baseline - actual``.

    Either side being unavailable returns ``None``. The result may be
    negative. This is a subtraction, so a zero baseline does not divide.
    """
    if baseline_cost is None or actual_cost is None:
        return None
    return quantize_money(baseline_cost - actual_cost)


def observed_input_volume(usage: ReportedUsage, provider: Provider) -> int:
    """Return the input-side token volume used for a same-token estimate."""
    if provider is not Provider.ANTHROPIC:
        return usage.input_tokens
    extra = 0
    if usage.cache_write_5m_tokens is not None or usage.cache_write_1h_tokens is not None:
        extra += usage.cache_write_5m_tokens or 0
        extra += usage.cache_write_1h_tokens or 0
    elif usage.cache_write_input_tokens is not None:
        extra += usage.cache_write_input_tokens
    if usage.cache_read_input_tokens is not None:
        extra += usage.cache_read_input_tokens
    return usage.input_tokens + extra


def pricing_snapshot(model_id: str, metadata: VerifiedModelMetadata) -> dict[str, object]:
    """Freeze the rates used for one attempt."""
    return {
        "model_id": model_id,
        "provider": metadata.provider.value,
        "input_per_million": decimal_text(metadata.input_cost_per_million_tokens),
        "output_per_million": decimal_text(metadata.output_cost_per_million_tokens),
        "cached_input_per_million": decimal_text(metadata.cached_input_per_million),
        "cache_read_per_million": decimal_text(metadata.cache_read_per_million),
        "cache_write_5m_per_million": decimal_text(metadata.cache_write_5m_per_million),
        "cache_write_1h_per_million": decimal_text(metadata.cache_write_1h_per_million),
        "source_url": metadata.price_source_url,
        "verified_on": metadata.price_verified_on,
        "api_cost_note": metadata.api_cost_note,
    }


def usage_from_response(response: LLMResponse) -> ReportedUsage:
    """Copy preserved usage fields from a normalized provider response."""
    return ReportedUsage(
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_input_tokens=response.cached_input_tokens,
        cache_write_input_tokens=response.cache_write_input_tokens,
        cache_read_input_tokens=response.cache_read_input_tokens,
        cache_write_5m_tokens=response.cache_write_5m_tokens,
        cache_write_1h_tokens=response.cache_write_1h_tokens,
        reasoning_tokens=response.reasoning_tokens,
    )


def lookup_prices(model_id: str) -> VerifiedModelMetadata | None:
    """Return verified prices for ``model_id``, or ``None`` when unpublished."""
    if model_id == "":
        return None
    metadata = verified_metadata(model_id)
    if metadata is None or not metadata.list_price_published:
        return None
    return metadata


def _openai_cost(usage: ReportedUsage, metadata: VerifiedModelMetadata) -> CostEstimate:
    output_cost = line_cost(usage.output_tokens, metadata.output_cost_per_million_tokens)
    cached = usage.cached_input_tokens
    if cached is None:
        input_cost = line_cost(usage.input_tokens, metadata.input_cost_per_million_tokens)
        return _finished(
            input_cost,
            output_cost,
            cached_cost=None,
            completeness=CostCompleteness.ESTIMATED,
        )
    if cached > usage.input_tokens:
        return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
    uncached_tokens = usage.input_tokens - cached
    if (
        usage.cache_write_input_tokens is not None
        and usage.cache_write_input_tokens > uncached_tokens
    ):
        return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
    if cached > 0 and metadata.cached_input_per_million is None:
        input_cost = line_cost(usage.input_tokens, metadata.input_cost_per_million_tokens)
        return _finished(
            input_cost,
            output_cost,
            cached_cost=None,
            completeness=CostCompleteness.ESTIMATED,
        )
    cache_price = metadata.cached_input_per_million or Decimal("0")
    input_cost = line_cost(uncached_tokens, metadata.input_cost_per_million_tokens)
    cached_cost = line_cost(cached, cache_price)
    return _finished(
        input_cost,
        output_cost,
        cached_cost=cached_cost,
        completeness=CostCompleteness.COMPLETE,
    )


def _anthropic_cost(usage: ReportedUsage, metadata: VerifiedModelMetadata) -> CostEstimate:
    input_cost = line_cost(usage.input_tokens, metadata.input_cost_per_million_tokens)
    output_cost = line_cost(usage.output_tokens, metadata.output_cost_per_million_tokens)
    has_split = (
        usage.cache_write_5m_tokens is not None or usage.cache_write_1h_tokens is not None
    )
    has_aggregate = usage.cache_write_input_tokens is not None
    has_read = usage.cache_read_input_tokens is not None
    if not has_split and not has_aggregate and not has_read:
        return _finished(
            input_cost,
            output_cost,
            cached_cost=None,
            completeness=CostCompleteness.ESTIMATED,
        )

    write_cost = Decimal("0")
    read_cost = Decimal("0")
    estimated = False
    if has_split:
        five = usage.cache_write_5m_tokens or 0
        hour = usage.cache_write_1h_tokens or 0
        if five > 0 and metadata.cache_write_5m_per_million is None:
            return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
        if hour > 0 and metadata.cache_write_1h_per_million is None:
            return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
        if metadata.cache_write_5m_per_million is not None:
            write_cost += line_cost(five, metadata.cache_write_5m_per_million)
        if metadata.cache_write_1h_per_million is not None:
            write_cost += line_cost(hour, metadata.cache_write_1h_per_million)
        if usage.cache_write_5m_tokens is None or usage.cache_write_1h_tokens is None:
            estimated = True
    elif has_aggregate:
        tokens = usage.cache_write_input_tokens or 0
        if tokens > 0 and metadata.cache_write_5m_per_million is None:
            return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
        if metadata.cache_write_5m_per_million is not None:
            write_cost = line_cost(tokens, metadata.cache_write_5m_per_million)
        estimated = True

    if has_read:
        tokens = usage.cache_read_input_tokens or 0
        if tokens > 0 and metadata.cache_read_per_million is None:
            return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
        if metadata.cache_read_per_million is not None:
            read_cost = line_cost(tokens, metadata.cache_read_per_million)
    else:
        estimated = True

    total = quantize_money(input_cost + output_cost + write_cost + read_cost)
    return CostEstimate(
        total=total,
        completeness=CostCompleteness.ESTIMATED if estimated else CostCompleteness.COMPLETE,
        input_cost=quantize_money(input_cost),
        output_cost=quantize_money(output_cost),
        cache_write_cost=quantize_money(write_cost),
        cache_read_cost=quantize_money(read_cost),
    )


def _finished(
    input_cost: Decimal,
    output_cost: Decimal,
    *,
    cached_cost: Decimal | None,
    completeness: CostCompleteness,
) -> CostEstimate:
    cached = Decimal("0") if cached_cost is None else cached_cost
    total = quantize_money(input_cost + output_cost + cached)
    return CostEstimate(
        total=total,
        completeness=completeness,
        input_cost=quantize_money(input_cost),
        output_cost=quantize_money(output_cost),
        cached_input_cost=None if cached_cost is None else quantize_money(cached_cost),
    )


def combine_costs(estimates: list[CostEstimate]) -> CostEstimate:
    """Sum known costs. Any unknown part makes the total unknown.

    A missing total is not replaced with zero. An empty list is unknown.
    """
    if not estimates or any(
        item.total is None or item.completeness is CostCompleteness.UNKNOWN for item in estimates
    ):
        return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
    total = quantize_money(sum((item.total for item in estimates), Decimal("0")))
    completeness = (
        CostCompleteness.ESTIMATED
        if any(item.completeness is CostCompleteness.ESTIMATED for item in estimates)
        else CostCompleteness.COMPLETE
    )
    return CostEstimate(total=total, completeness=completeness)


def _is_zero(metadata: VerifiedModelMetadata) -> bool:
    return (
        metadata.input_cost_per_million_tokens == 0
        and metadata.output_cost_per_million_tokens == 0
    )
