"""Build the routing catalog from settings and verified model metadata.

Model ids, quality tiers, and routing flags come from settings. Context
windows and list prices are looked up for the documented model ids only.
The pricing service reads this same table. An unknown model id does not
inherit another model's prices.

Quality tiers are routing policy labels, not benchmark results.
A model is enabled only when its routing flag is on and its required
configuration is present. An adapter on its own does not enable routing.
"""

from dataclasses import dataclass
from decimal import Decimal

from app.api.schemas import ModelConfig, Provider, QualityTier
from app.config import Settings
from app.routing.model_registry import ModelRegistry


class CatalogConfigurationError(ValueError):
    """Raised when a configured model id has no verified catalog metadata."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(
            "No verified catalog metadata for model id "
            f"{model_id!r}. Documented ids are gpt-5-nano, claude-sonnet-4-6, "
            "llama3.2, and gemma4:31b."
        )


@dataclass(frozen=True)
class VerifiedModelMetadata:
    """Official context window and list price for one documented model id.

    Prices were checked on 2026-09-28.

    OpenAI ``gpt-5-nano``: 400,000-token context. Standard rates are $0.05
    input, $0.005 cached input, and $0.40 output per million tokens
    (https://developers.openai.com/api/docs/models/gpt-5-nano). The model
    card headline is $0.05 input and $0.40 output. Cached input on that page
    is $0.005, which is 10 percent of the headline input price. The extracted
    token table is labeled "Batch API price"; the headline and the
    quick-comparison input price match these input and output figures.
    Models before GPT-5.6 have no separate cache-write price
    (https://developers.openai.com/api/docs/guides/prompt-caching). Write
    tokens are billed as uncached input and are not added a second time.
    ``output_tokens`` already includes reasoning tokens.

    Anthropic ``claude-sonnet-4-6``: 1,000,000-token context. Standard rates
    are $3 input and $15 output per million. Cache writes are $3.75 for the
    5-minute TTL and $6 for the 1-hour TTL. Cache reads are $0.30
    (https://platform.claude.com/docs/en/about-claude/pricing). Reported
    ``input_tokens`` excludes cache creation and cache reads.

    Ollama ``llama3.2``: 131,072-token context on the library model
    (https://ollama.com/library/llama3.2). The provider API price is 0.
    That zero does not include hardware or electricity.

    Ollama Cloud ``gemma4:31b``: 262,144-token context
    (https://ollama.com/library/gemma4). Direct cloud requests use this name.
    Ollama Cloud does not publish a per-token list price, so this entry is
    not used to calculate a dollar cost.
    """

    model_name: str
    provider: Provider
    context_window: int
    input_cost_per_million_tokens: Decimal
    output_cost_per_million_tokens: Decimal
    local: bool
    price_source_url: str
    price_verified_on: str
    api_cost_note: str
    list_price_published: bool = True
    cached_input_per_million: Decimal | None = None
    cache_read_per_million: Decimal | None = None
    cache_write_5m_per_million: Decimal | None = None
    cache_write_1h_per_million: Decimal | None = None


_VERIFIED: dict[str, VerifiedModelMetadata] = {
    "gpt-5-nano": VerifiedModelMetadata(
        model_name="GPT-5 nano",
        provider=Provider.OPENAI,
        context_window=400_000,
        input_cost_per_million_tokens=Decimal("0.05"),
        output_cost_per_million_tokens=Decimal("0.40"),
        local=False,
        price_source_url="https://developers.openai.com/api/docs/models/gpt-5-nano",
        price_verified_on="2026-09-28",
        api_cost_note=(
            "Standard rates. Cached input is a subset of input tokens. "
            "Cache writes have no separate rate and stay in the uncached input bucket. "
            "Reasoning tokens are already included in output tokens."
        ),
        cached_input_per_million=Decimal("0.005"),
    ),
    "claude-sonnet-4-6": VerifiedModelMetadata(
        model_name="Claude Sonnet 4.6",
        provider=Provider.ANTHROPIC,
        context_window=1_000_000,
        input_cost_per_million_tokens=Decimal("3"),
        output_cost_per_million_tokens=Decimal("15"),
        local=False,
        price_source_url="https://platform.claude.com/docs/en/about-claude/pricing",
        price_verified_on="2026-09-28",
        api_cost_note=(
            "Standard rates. input_tokens excludes cache writes and cache reads. "
            "An aggregate cache-write count without a 5-minute and 1-hour split "
            "is priced at the 5-minute write rate and marked as an estimate."
        ),
        cache_read_per_million=Decimal("0.30"),
        cache_write_5m_per_million=Decimal("3.75"),
        cache_write_1h_per_million=Decimal("6"),
    ),
    "llama3.2": VerifiedModelMetadata(
        model_name="Llama 3.2",
        provider=Provider.OLLAMA,
        context_window=131_072,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        local=True,
        price_source_url="https://ollama.com/library/llama3.2",
        price_verified_on="2026-09-28",
        api_cost_note=(
            "Provider API cost is zero. Hardware and electricity are excluded. "
            "Cached prompt tokens are not added on top of prompt_eval_count."
        ),
    ),
    "gemma4:31b": VerifiedModelMetadata(
        model_name="Gemma 4 31B",
        provider=Provider.OLLAMA,
        context_window=262_144,
        input_cost_per_million_tokens=Decimal("0"),
        output_cost_per_million_tokens=Decimal("0"),
        local=False,
        price_source_url="https://ollama.com/library/gemma4",
        price_verified_on="2026-09-29",
        api_cost_note=(
            "Ollama Cloud does not publish a per-token list price. "
            "The zero rates on this entry are not a charge and are not used for cost."
        ),
        list_price_published=False,
    ),
}


def verified_metadata(model_id: str) -> VerifiedModelMetadata | None:
    """Return the price book entry for a documented model id, or ``None``."""
    return _VERIFIED.get(model_id)


def build_catalog(settings: Settings) -> ModelRegistry:
    """Register the three documented models. Only explicitly ready ones are enabled."""
    return ModelRegistry(
        [
            _entry(
                settings.openai_model,
                provider=Provider.OPENAI,
                quality_tier=settings.openai_quality_tier,
                enabled=settings.openai_routing_enabled and settings.openai_api_key is not None,
            ),
            _entry(
                settings.anthropic_model,
                provider=Provider.ANTHROPIC,
                quality_tier=settings.anthropic_quality_tier,
                enabled=(
                    settings.anthropic_routing_enabled
                    and settings.anthropic_api_key is not None
                ),
            ),
            _entry(
                settings.ollama_model,
                provider=Provider.OLLAMA,
                quality_tier=settings.ollama_quality_tier,
                enabled=(
                    settings.ollama_routing_enabled and settings.ollama_base_url.strip() != ""
                ),
            ),
        ]
    )


def _entry(
    model_id: str,
    *,
    provider: Provider,
    quality_tier: QualityTier,
    enabled: bool,
) -> ModelConfig:
    metadata = _VERIFIED.get(model_id)
    if metadata is None:
        raise CatalogConfigurationError(model_id)
    return ModelConfig(
        provider=provider,
        model_name=metadata.model_name,
        model_id=model_id,
        quality_tier=quality_tier,
        input_cost_per_million_tokens=metadata.input_cost_per_million_tokens,
        output_cost_per_million_tokens=metadata.output_cost_per_million_tokens,
        context_window=metadata.context_window,
        enabled=enabled,
        local=metadata.local,
    )
