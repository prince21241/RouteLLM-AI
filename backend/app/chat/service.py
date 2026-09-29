"""Classify a request, select one model, and call that provider.

Quality checks and escalation are off unless settings enable them. A
provider failure on the first call stops the request. There is no provider
retry and no provider fallback. At most one extra generation runs, and only
after an explicit quality failure. When a store is configured, the pending
row is committed before the provider call and the outcome is committed
afterward. The database transaction is not held open across the network.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from app.api.chat_schemas import ChatMetrics, ChatResponse, RoutingDetails
from app.api.schemas import LLMResponse, ModelConfig, Provider, ReportedUsage
from app.config import Settings
from app.db.records import AttemptOutcome, EvaluationWrite, PendingRequest
from app.db.store import RequestStore
from app.evaluation.escalation import select_escalation_target
from app.evaluation.live import build_live_evaluator
from app.evaluation.schema import EvaluationResult
from app.pricing.money import money_to_api
from app.pricing.service import (
    CostCompleteness,
    CostEstimate,
    combine_costs,
    estimate_baseline,
    estimate_cost,
    lookup_prices,
    pricing_snapshot,
    savings,
    usage_from_response,
)
from app.providers.base import LLMProvider
from app.providers.client import monotonic_now
from app.providers.errors import ProviderError, ProviderResponseError
from app.routing.catalog import VerifiedModelMetadata
from app.routing.complexity import ComplexityClassifier
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter

logger = logging.getLogger("app.chat")

_PERSISTENCE_WARNING = "The response was generated but could not be saved."


class InputValidationError(ValueError):
    """Raised when the chat input fails the application checks."""


class PersistenceUnavailableError(RuntimeError):
    """Raised when the request cannot be stored before generation."""

    def __init__(self) -> None:
        super().__init__("Request storage is unavailable")


ProviderFactory = Callable[[ModelConfig], LLMProvider]


class ChatService:
    """Offline-testable orchestration. Dependencies are injected."""

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        classifier: ComplexityClassifier,
        router: ModelRouter,
        provider_factory: ProviderFactory,
        store: RequestStore | None = None,
        evaluator: object | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._classifier = classifier
        self._router = router
        self._provider_factory = provider_factory
        self._store = store
        self._evaluator = evaluator

    async def complete(self, prompt: str, system_prompt: str | None = None) -> ChatResponse:
        """Validate, route, and generate once."""
        started = monotonic_now()
        user_prompt, system = _validated_input(
            prompt,
            system_prompt,
            max_characters=self._settings.max_input_characters,
        )
        assessment = self._classifier.classify(
            user_prompt,
            system,
            low=self._settings.low_complexity_threshold,
            high=self._settings.high_complexity_threshold,
        )
        decision = self._router.select(assessment.tier, self._registry)
        request_id = uuid4()
        created_at = datetime.now(timezone.utc)
        if self._store is not None:
            try:
                await self._store.create_pending(
                    PendingRequest(
                        request_id=request_id,
                        prompt=user_prompt,
                        system_prompt=system,
                        created_at=created_at,
                        complexity_score=_score(assessment.score),
                        complexity_tier=assessment.tier.value,
                        complexity_reasons=list(assessment.reasons),
                        provider=decision.model.provider.value,
                        model_id=decision.model.model_id,
                        selected_model_tier=decision.model.quality_tier.value,
                        selection_reason=decision.selection_reason,
                        degraded=decision.degraded,
                    )
                )
            except Exception as exc:
                _log_persistence(request_id, exc)
                raise PersistenceUnavailableError() from None

        provider: LLMProvider | None = None
        second_provider: LLMProvider | None = None
        owned_evaluator: object | None = None
        provider_started = monotonic_now()
        try:
            try:
                provider = self._provider_factory(decision.model)
                provider_started = monotonic_now()
                generated = await provider.generate(user_prompt, system)
            except ProviderError as exc:
                await self._record_failure(
                    request_id=request_id,
                    decision_provider=decision.model.provider.value,
                    configured_model_id=decision.model.model_id,
                    exc=exc,
                    provider_latency_ms=(monotonic_now() - provider_started) * 1000,
                    started=started,
                )
                raise
            priced = _price_success(generated, decision.model.model_id, self._settings)
            evaluation: EvaluationResult | None = None
            escalation_reason: str | None = None
            escalation_error: str | None = None
            escalated = False
            returned = generated
            returned_model_id = decision.model.model_id
            returned_attempt = 1
            second_priced: _Priced | None = None
            second_error: str | None = None
            second_model_id: str | None = None
            second_provider_name = decision.model.provider.value
            if self._settings.quality_evaluation_enabled:
                evaluator = self._evaluator
                if evaluator is None:
                    evaluator = build_live_evaluator(self._settings)
                    owned_evaluator = evaluator
                try:
                    evaluation = await evaluator.evaluate(user_prompt, generated.content)  # type: ignore[attr-defined]
                except Exception:
                    evaluation = EvaluationResult(
                        verdict="error",
                        score=None,
                        reasons=("The quality evaluator failed.",),
                        method="live",
                        error_message="The quality evaluator failed.",
                    )
                if self._settings.escalation_enabled and evaluation.verdict == "fail":
                    target = select_escalation_target(
                        decision.model,
                        self._registry,
                        self._settings,
                    )
                    escalation_reason = f"initial answer failed quality; {target.reason}"
                    if target.model is None:
                        escalation_error = target.reason
                    else:
                        second_model_id = target.model.model_id
                        second_provider_name = target.model.provider.value
                        try:
                            second_provider = self._provider_factory(target.model)
                            second = await second_provider.generate(user_prompt, system)
                        except ProviderError as exc:
                            second_error = str(exc)[:500]
                            escalation_error = second_error
                        else:
                            returned = second
                            returned_model_id = target.model.model_id
                            returned_attempt = 2
                            escalated = True
                            second_priced = _price_success(
                                second,
                                target.model.model_id,
                                self._settings,
                            )
            returned_priced = (
                second_priced
                if second_priced is not None
                else _price_success(returned, returned_model_id, self._settings)
            )
            total_estimate = priced.estimate
            if self._settings.quality_evaluation_enabled:
                parts = [priced.estimate]
                if second_priced is not None:
                    parts.append(second_priced.estimate)
                if evaluation is not None and evaluation.judge_cost is not None:
                    parts.append(evaluation.judge_cost)
                total_estimate = combine_costs(parts)
            if self._settings.quality_evaluation_enabled:
                persistence_status, persistence_warning = await self._record_checked(
                    request_id=request_id,
                    generated=generated,
                    priced=priced,
                    returned=returned,
                    returned_priced=returned_priced,
                    total_estimate=total_estimate,
                    started=started,
                    evaluation=evaluation,
                    escalated=escalated,
                    escalation_reason=escalation_reason,
                    escalation_error=escalation_error,
                    returned_model_id=returned_model_id,
                    returned_attempt=returned_attempt,
                    second_error=second_error,
                    second_model_id=second_model_id,
                    second_provider_name=second_provider_name,
                    second_priced=second_priced,
                )
            else:
                persistence_status, persistence_warning = await self._record_success(
                    request_id=request_id,
                    generated=generated,
                    priced=priced,
                    started=started,
                )
            return _chat_response(
                request_id=str(request_id),
                generated=returned,
                assessment_score=assessment.score,
                assessment_tier=assessment.tier,
                assessment_reasons=list(assessment.reasons),
                provider=decision.model.provider,
                model_id=decision.model.model_id,
                model_tier=decision.model.quality_tier,
                selection_reason=decision.selection_reason,
                degraded=decision.degraded,
                priced=returned_priced,
                total_estimate=total_estimate,
                end_to_end_ms=_elapsed_ms(started),
                persistence_status=persistence_status,
                persistence_warning=persistence_warning,
                escalated=escalated,
                evaluation=evaluation,
                escalation_reason=escalation_reason,
                escalation_error=escalation_error,
                returned_model=returned_model_id,
                returned_attempt=returned_attempt,
            )
        finally:
            if provider is not None:
                await _close_provider(provider)
            if second_provider is not None:
                await _close_provider(second_provider)
            if owned_evaluator is not None:
                close_evaluator = getattr(owned_evaluator, "aclose", None)
                if close_evaluator is not None:
                    await close_evaluator()

    async def _record_failure(
        self,
        *,
        request_id: object,
        decision_provider: str,
        configured_model_id: str,
        exc: ProviderError,
        provider_latency_ms: float,
        started: float,
    ) -> None:
        if self._store is None:
            return
        usage = exc.usage if isinstance(exc, ProviderResponseError) else None
        if not isinstance(usage, ReportedUsage):
            usage = None
        priced = _price_usage(usage, configured_model_id, decision_provider, self._settings)
        try:
            await self._store.record_failure(
                _outcome(
                    request_id=request_id,
                    status="failed",
                    provider=decision_provider,
                    configured_model_id=configured_model_id,
                    reported_model_id=None,
                    usage=usage,
                    provider_latency_ms=provider_latency_ms,
                    priced=priced,
                    error_message=str(exc)[:500],
                    response_text=None,
                    started=started,
                )
            )
        except Exception as persist_exc:
            _log_persistence(request_id, persist_exc)

    async def _record_success(
        self,
        *,
        request_id: object,
        generated: LLMResponse,
        priced: "_Priced",
        started: float,
    ) -> tuple[str, str | None]:
        if self._store is None:
            return "not_configured", None
        try:
            await self._store.record_success(
                _outcome(
                    request_id=request_id,
                    status="succeeded",
                    provider=generated.provider.value,
                    configured_model_id=priced.configured_model_id,
                    reported_model_id=generated.model,
                    usage=usage_from_response(generated),
                    provider_latency_ms=generated.latency_ms,
                    priced=priced,
                    error_message=None,
                    response_text=generated.content,
                    started=started,
                )
            )
        except Exception as exc:
            _log_persistence(request_id, exc)
            return "failed", _PERSISTENCE_WARNING
        return "stored", None

    async def _record_checked(
        self,
        *,
        request_id: object,
        generated: LLMResponse,
        priced: _Priced,
        returned: LLMResponse,
        returned_priced: _Priced,
        total_estimate: CostEstimate,
        started: float,
        evaluation: EvaluationResult | None,
        escalated: bool,
        escalation_reason: str | None,
        escalation_error: str | None,
        returned_model_id: str,
        returned_attempt: int,
        second_error: str | None,
        second_model_id: str | None,
        second_provider_name: str,
        second_priced: _Priced | None,
    ) -> tuple[str, str | None]:
        if self._store is None:
            return "not_configured", None
        final = _outcome(
            request_id=request_id,
            status="succeeded",
            provider=returned.provider.value,
            configured_model_id=returned_model_id,
            reported_model_id=returned.model,
            usage=usage_from_response(returned),
            provider_latency_ms=returned.latency_ms,
            priced=returned_priced,
            error_message=None,
            response_text=returned.content,
            started=started,
        )
        final = _with_check(
            final,
            total_estimate=total_estimate,
            escalated=escalated,
            evaluation=evaluation,
            escalation_reason=escalation_reason,
            escalation_error=escalation_error,
            final_model_id=returned_model_id,
            returned_attempt=returned_attempt,
        )
        first = _outcome(
            request_id=request_id,
            status="succeeded",
            provider=generated.provider.value,
            configured_model_id=priced.configured_model_id,
            reported_model_id=generated.model,
            usage=usage_from_response(generated),
            provider_latency_ms=generated.latency_ms,
            priced=priced,
            error_message=None,
            response_text=generated.content,
            started=started,
        )
        try:
            save_attempt = getattr(self._store, "save_attempt", None)
            finalize = getattr(self._store, "finalize_request", None)
            if save_attempt is None or finalize is None:
                await self._store.record_success(final)
                return "stored", None
            await save_attempt(first)
            if second_model_id is not None:
                if second_priced is not None:
                    second = _outcome(
                        request_id=request_id,
                        status="succeeded",
                        provider=second_provider_name,
                        configured_model_id=second_model_id,
                        reported_model_id=returned.model,
                        usage=usage_from_response(returned),
                        provider_latency_ms=returned.latency_ms,
                        priced=second_priced,
                        error_message=None,
                        response_text=returned.content,
                        started=started,
                    )
                    second = _replace_attempt(second, 2)
                    await save_attempt(second)
                else:
                    failed = _outcome(
                        request_id=request_id,
                        status="failed",
                        provider=second_provider_name,
                        configured_model_id=second_model_id,
                        reported_model_id=None,
                        usage=None,
                        provider_latency_ms=0,
                        priced=_empty_price(second_model_id),
                        error_message=second_error,
                        response_text=None,
                        started=started,
                    )
                    await save_attempt(_replace_attempt(failed, 2))
            await finalize(final)
        except Exception as exc:
            _log_persistence(request_id, exc)
            return "failed", _PERSISTENCE_WARNING
        return "stored", None


class _Priced:
    """Cost fields shared by the response and the stored attempt."""

    def __init__(
        self,
        *,
        configured_model_id: str,
        estimate: CostEstimate,
        baseline_model: str | None,
        baseline_cost: Decimal | None,
        estimated_savings: Decimal | None,
        snapshot: dict[str, object] | None,
    ) -> None:
        self.configured_model_id = configured_model_id
        self.estimate = estimate
        self.baseline_model = baseline_model
        self.baseline_cost = baseline_cost
        self.estimated_savings = estimated_savings
        self.snapshot = snapshot


def _with_check(
    outcome: AttemptOutcome,
    *,
    total_estimate: CostEstimate,
    escalated: bool,
    evaluation: EvaluationResult | None,
    escalation_reason: str | None,
    escalation_error: str | None,
    final_model_id: str,
    returned_attempt: int,
) -> AttemptOutcome:
    evaluations: tuple[EvaluationWrite, ...] = ()
    if evaluation is not None:
        usage = evaluation.judge_usage
        judge_cost = evaluation.judge_cost
        evaluations = (
            EvaluationWrite(
                attempt_number=1,
                source="live",
                method=evaluation.method,
                verdict=evaluation.verdict,
                score=evaluation.score_decimal(),
                reasons=list(evaluation.reasons),
                judge_model=evaluation.judge_model,
                judge_input_tokens=None if usage is None else usage.input_tokens,
                judge_output_tokens=None if usage is None else usage.output_tokens,
                judge_cost=None if judge_cost is None else judge_cost.total,
                judge_cost_completeness=None if judge_cost is None else judge_cost.completeness.value,
                judge_latency_ms=(
                    None if evaluation.judge_latency_ms is None else _latency(evaluation.judge_latency_ms)
                ),
                error_message=evaluation.error_message,
                created_at=outcome.completed_at,
            ),
        )
    return replace(
        outcome,
        total_cost=total_estimate.total,
        cost_completeness=total_estimate.completeness.value,
        escalated=escalated,
        quality_verdict=None if evaluation is None else evaluation.verdict,
        quality_score=None if evaluation is None else evaluation.score_decimal(),
        quality_reasons=None if evaluation is None else list(evaluation.reasons),
        escalation_reason=escalation_reason,
        escalation_error=escalation_error,
        final_model_id=final_model_id,
        returned_attempt_number=returned_attempt,
        evaluations=evaluations,
    )


def _replace_attempt(outcome: AttemptOutcome, attempt_number: int) -> AttemptOutcome:
    return replace(outcome, attempt_number=attempt_number)


def _empty_price(model_id: str) -> _Priced:
    return _Priced(
        configured_model_id=model_id,
        estimate=CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN),
        baseline_model=None,
        baseline_cost=None,
        estimated_savings=None,
        snapshot=None,
    )


def _price_success(generated: LLMResponse, model_id: str, settings: Settings) -> _Priced:
    return _price_usage(
        usage_from_response(generated),
        model_id,
        generated.provider.value,
        settings,
    )


def _price_usage(
    usage: ReportedUsage | None,
    model_id: str,
    source_provider: str,
    settings: Settings,
) -> _Priced:
    metadata = lookup_prices(model_id)
    snapshot = pricing_snapshot(model_id, metadata) if metadata is not None else None
    if usage is None or metadata is None:
        estimate = CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
        baseline_model, baseline_cost = _baseline(usage, source_provider, settings)
        return _Priced(
            configured_model_id=model_id,
            estimate=estimate,
            baseline_model=baseline_model,
            baseline_cost=baseline_cost,
            estimated_savings=savings(baseline_cost, None),
            snapshot=snapshot,
        )
    estimate = estimate_cost(usage, metadata)
    baseline_model, baseline_cost = _baseline(usage, source_provider, settings)
    return _Priced(
        configured_model_id=model_id,
        estimate=estimate,
        baseline_model=baseline_model,
        baseline_cost=baseline_cost,
        estimated_savings=savings(baseline_cost, estimate.total),
        snapshot=snapshot,
    )


def _baseline(
    usage: ReportedUsage | None,
    source_provider: str,
    settings: Settings,
) -> tuple[str | None, Decimal | None]:
    model_id = settings.premium_baseline_model
    if model_id == "" or usage is None:
        return None, None
    metadata: VerifiedModelMetadata | None = lookup_prices(model_id)
    if metadata is None:
        return None, None
    return model_id, estimate_baseline(usage, Provider(source_provider), metadata)


def _outcome(
    *,
    request_id: object,
    status: str,
    provider: str,
    configured_model_id: str,
    reported_model_id: str | None,
    usage: ReportedUsage | None,
    provider_latency_ms: float,
    priced: _Priced,
    error_message: str | None,
    response_text: str | None,
    started: float,
) -> AttemptOutcome:
    return AttemptOutcome(
        request_id=request_id,  # type: ignore[arg-type]
        status=status,
        completed_at=datetime.now(timezone.utc),
        provider=provider,
        configured_model_id=configured_model_id,
        reported_model_id=reported_model_id,
        input_tokens=None if usage is None else usage.input_tokens,
        output_tokens=None if usage is None else usage.output_tokens,
        cached_input_tokens=None if usage is None else usage.cached_input_tokens,
        cache_write_input_tokens=None if usage is None else usage.cache_write_input_tokens,
        cache_read_input_tokens=None if usage is None else usage.cache_read_input_tokens,
        cache_write_5m_tokens=None if usage is None else usage.cache_write_5m_tokens,
        cache_write_1h_tokens=None if usage is None else usage.cache_write_1h_tokens,
        reasoning_tokens=None if usage is None else usage.reasoning_tokens,
        provider_latency_ms=_latency(provider_latency_ms),
        estimated_cost=priced.estimate.total,
        cost_completeness=priced.estimate.completeness.value,
        error_message=error_message,
        pricing_snapshot=priced.snapshot,
        response_text=response_text,
        total_cost=priced.estimate.total,
        end_to_end_latency_ms=_elapsed_ms(started),
        premium_baseline_model=priced.baseline_model,
        premium_baseline_cost=priced.baseline_cost,
        estimated_savings=priced.estimated_savings,
    )


def _chat_response(
    *,
    request_id: str,
    generated: LLMResponse,
    assessment_score: float,
    assessment_tier: object,
    assessment_reasons: list[str],
    provider: object,
    model_id: str,
    model_tier: object,
    selection_reason: str,
    degraded: bool,
    priced: _Priced,
    total_estimate: CostEstimate,
    end_to_end_ms: Decimal,
    persistence_status: str,
    persistence_warning: str | None,
    escalated: bool = False,
    evaluation: EvaluationResult | None = None,
    escalation_reason: str | None = None,
    escalation_error: str | None = None,
    returned_model: str | None = None,
    returned_attempt: int = 1,
) -> ChatResponse:
    return ChatResponse(
        request_id=request_id,
        response=generated.content,
        routing=RoutingDetails(
            complexity_score=assessment_score,
            complexity_tier=assessment_tier,  # type: ignore[arg-type]
            complexity_reasons=assessment_reasons,
            provider=provider,  # type: ignore[arg-type]
            model=model_id,
            selected_model_tier=model_tier,  # type: ignore[arg-type]
            selection_reason=selection_reason,
            degraded=degraded,
            escalated=escalated,
        ),
        metrics=ChatMetrics(
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            latency_ms=generated.latency_ms,
            cost=money_to_api(total_estimate.total),
            cost_completeness=total_estimate.completeness.value,  # type: ignore[arg-type]
            quality_score=None if evaluation is None else evaluation.score,
        ),
        baseline_model=priced.baseline_model,
        baseline_cost=money_to_api(priced.baseline_cost),
        estimated_savings=money_to_api(priced.estimated_savings),
        savings_basis="same_token_volume",
        persistence_status=persistence_status,  # type: ignore[arg-type]
        persistence_warning=persistence_warning,
        end_to_end_latency_ms=float(end_to_end_ms),
        quality_verdict=None if evaluation is None else evaluation.verdict,
        quality_reasons=[] if evaluation is None else list(evaluation.reasons),
        escalation_reason=escalation_reason,
        escalation_error=escalation_error,
        returned_model=returned_model,
        returned_attempt=returned_attempt,
    )


def _validated_input(
    prompt: str,
    system_prompt: str | None,
    *,
    max_characters: int,
) -> tuple[str, str | None]:
    """Return stripped prompt parts or raise an application-limit error.

    The limit counts characters in the user prompt and system prompt together.
    It is not a provider token or context-window check.
    """
    user_prompt = prompt.strip()
    system = system_prompt.strip() if system_prompt is not None else ""
    if user_prompt == "":
        raise InputValidationError("Prompt must not be blank")
    if len(user_prompt) + len(system) > max_characters:
        raise InputValidationError(
            f"Input exceeds the application limit of {max_characters} characters"
        )
    return user_prompt, system or None


async def _close_provider(provider: LLMProvider) -> None:
    close = getattr(provider, "aclose", None)
    if close is not None:
        await close()


def _score(value: float) -> Decimal:
    return Decimal(f"{value:.6f}")


def _latency(value: float) -> Decimal:
    return Decimal(str(round(value, 3)))


def _elapsed_ms(started: float) -> Decimal:
    return Decimal(str(round((monotonic_now() - started) * 1000, 3)))


def _log_persistence(request_id: object, exc: Exception) -> None:
    logger.error(
        "Failed to persist request %s (%s)",
        request_id,
        type(exc).__name__,
    )
