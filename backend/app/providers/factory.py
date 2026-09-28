"""Construct the provider selected by the router.

The model id comes from the registry entry. Timeouts and output limits stay
on the provider settings. Callers close the returned client.
"""

from app.api.schemas import ModelConfig, Provider
from app.config import Settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import LLMProvider
from app.providers.errors import ProviderConfigurationError
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider


def build_provider(model: ModelConfig, settings: Settings) -> LLMProvider:
    """Build one client for ``model``. A missing credential raises here."""
    if model.provider is Provider.OPENAI:
        if settings.openai_api_key is None:
            raise ProviderConfigurationError("OPENAI_API_KEY is not configured")
        return OpenAIProvider(
            settings.openai_api_key.get_secret_value(),
            model=model.model_id,
            max_output_tokens=settings.openai_max_output_tokens,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    if model.provider is Provider.ANTHROPIC:
        if settings.anthropic_api_key is None:
            raise ProviderConfigurationError("ANTHROPIC_API_KEY is not configured")
        return AnthropicProvider(
            settings.anthropic_api_key.get_secret_value(),
            model=model.model_id,
            max_tokens=settings.anthropic_max_tokens,
            timeout_seconds=settings.anthropic_timeout_seconds,
        )
    if model.provider is Provider.OLLAMA:
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=model.model_id,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    raise ProviderConfigurationError("The selected provider is not supported")
