"""Live quality checks have no reference answer.

When the model judge is off, the result is unknown. Unknown does not
escalate. Dataset graders are not used on live chat traffic.
"""

from collections.abc import Awaitable, Callable

from app.api.schemas import LLMResponse
from app.config import Settings
from app.evaluation.judge import LIVE_RUBRIC, run_judge
from app.evaluation.schema import EvaluationResult
from app.providers.openai import OpenAIProvider

Generate = Callable[[str, str | None], Awaitable[LLMResponse]]


class LiveEvaluator:
    """One live check. A missing judge reports unknown and does not call a model."""

    def __init__(
        self,
        generate: Generate | None,
        *,
        model_id: str | None,
        minimum_score: float,
        provider: object | None = None,
    ) -> None:
        self._generate = generate
        self._model_id = model_id
        self._minimum_score = minimum_score
        self._provider = provider

    async def evaluate(self, prompt: str, answer: str) -> EvaluationResult:
        if self._generate is None or self._model_id is None:
            return EvaluationResult(
                verdict="unknown",
                score=None,
                reasons=(
                    "Live evaluation has no reference answer and the model judge is disabled.",
                ),
                method="live",
            )
        result = await run_judge(
            self._generate,
            model_id=self._model_id,
            prompt=prompt,
            answer=answer,
            rubric=LIVE_RUBRIC,
        )
        return apply_score_threshold(result, self._minimum_score)

    async def aclose(self) -> None:
        close = getattr(self._provider, "aclose", None)
        if close is not None:
            await close()


def build_live_evaluator(settings: Settings) -> LiveEvaluator:
    """Build a judge from the OpenAI settings, or a no-network unknown result."""
    if not settings.quality_judge_enabled or settings.openai_api_key is None:
        return LiveEvaluator(None, model_id=None, minimum_score=settings.min_quality_score)
    provider = OpenAIProvider(
        settings.openai_api_key.get_secret_value(),
        model=settings.quality_judge_model,
        max_output_tokens=settings.openai_max_output_tokens,
        timeout_seconds=settings.openai_timeout_seconds,
    )

    async def generate(prompt: str, system_prompt: str | None = None) -> LLMResponse:
        return await provider.generate(prompt, system_prompt)

    return LiveEvaluator(
        generate,
        model_id=settings.quality_judge_model,
        minimum_score=settings.min_quality_score,
        provider=provider,
    )


def apply_score_threshold(result: EvaluationResult, minimum_score: float) -> EvaluationResult:
    """Turn an explicit pass below the configured score into a failure.

    Unknown and error verdicts are left unchanged. A missing score is not
    treated as a failure.
    """
    if result.verdict != "pass" or result.score is None or result.score >= minimum_score:
        return result
    return EvaluationResult(
        verdict="fail",
        score=result.score,
        reasons=result.reasons + (f"score is below the configured minimum of {minimum_score}",),
        method=result.method,
        judge_model=result.judge_model,
        judge_usage=result.judge_usage,
        judge_latency_ms=result.judge_latency_ms,
        judge_cost=result.judge_cost,
        error_message=result.error_message,
        error_category=result.error_category,
    )
