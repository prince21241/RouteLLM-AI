"""Decimal money helpers.

Request cost uses this formula, with ``Decimal`` values throughout::

    input_cost = input_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price
    total_cost = input_cost + output_cost

The stored and API value is quantized to 12 decimal places, half up.
JSON responses encode money as a plain decimal string, not a JSON number,
so small costs are not rounded through binary floating point. Trailing
zeros are removed. ``1_000_000`` input tokens at ``$0.05`` per million is
``"0.05"``. Twelve input tokens at that rate is ``"0.0000006"``.
"""

from decimal import Decimal, ROUND_HALF_UP

MILLION = Decimal("1000000")
MONEY_QUANTUM = Decimal("0.000000000001")


def line_cost(tokens: int, price_per_million: Decimal) -> Decimal:
    """Return ``tokens / 1_000_000 * price`` without rounding."""
    return (Decimal(tokens) / MILLION) * price_per_million


def quantize_money(value: Decimal) -> Decimal:
    """Round a money amount to 12 decimal places, half up."""
    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def money_to_api(value: Decimal | None) -> str | None:
    """Serialize money as a plain decimal string, or ``None``."""
    if value is None:
        return None
    quantized = quantize_money(value)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def decimal_text(value: Decimal | None) -> str | None:
    """Serialize an unrounded rate for a pricing snapshot."""
    if value is None:
        return None
    return format(value, "f")
