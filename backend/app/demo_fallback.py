"""Mocked provider-fallback demonstration. This does not call a paid API.

Run from ``backend``:

    python -m app.demo_fallback
"""

import asyncio

from app.api.schemas import LLMResponse, ModelConfig, Provider, QualityTier
from app.chat.service import ChatService
from app.config import Settings
from app.providers.errors import ProviderTimeoutError
from app.routing.complexity import ComplexityClassifier
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter

_PROMPT = "Explain what an API is in two sentences"


class _Script:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def factory(self, model: ModelConfig):
        script = self

        class _Provider:
            async def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
                del prompt, system_prompt
                script.calls.append(model.model_id)
                if model.model_id == "gpt-5-nano":
                    raise ProviderTimeoutError("OpenAI request timed out")
                return LLMResponse(
                    provider=model.provider,
                    model=model.model_id,
                    content="Fallback answer from the second provider.",
                    input_tokens=11,
                    output_tokens=9,
                    latency_ms=42.5,
                )

            async def aclose(self) -> None:
                return None

        return _Provider()


def _model(model_id: str, provider: Provider, tier: QualityTier) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model_name=model_id,
        model_id=model_id,
        quality_tier=tier,
        input_cost_per_million_tokens=0,  # type: ignore[arg-type]
        output_cost_per_million_tokens=0,  # type: ignore[arg-type]
        context_window=1024,
        enabled=True,
        local=provider is Provider.OLLAMA,
    )


def main() -> None:
    """Show one timeout followed by a successful fallback."""
    script = _Script()
    service = ChatService(
        settings=Settings(
            _env_file=None,
            fallback_enabled=True,
            fallback_models="claude-sonnet-4-6",
        ),
        registry=ModelRegistry(
            [
                _model("gpt-5-nano", Provider.OPENAI, QualityTier.LOW),
                _model("claude-sonnet-4-6", Provider.ANTHROPIC, QualityTier.HIGH),
            ]
        ),
        classifier=ComplexityClassifier(),
        router=ModelRouter(),
        provider_factory=script.factory,
    )
    response = asyncio.run(service.complete(_PROMPT))
    print(f"calls: {', '.join(script.calls)}")
    print(f"fallback_used: {response.fallback_used}")
    print(f"fallback_reason: {response.fallback_reason}")
    print(f"final_provider: {response.final_provider}")
    print(f"response: {response.response}")


if __name__ == "__main__":
    main()
