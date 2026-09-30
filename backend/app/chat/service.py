"""Classify a request, select one model, and call that provider.

Quality checks, escalation, and provider fallback are off unless settings
enable them. With fallback disabled, a provider failure on the first call
stops the request. There is no provider retry. Escalation still allows at
most one extra generation after an explicit quality failure. With fallback
enabled, a fallback-eligible failure may call one other provider, and the
original call, escalation, and fallback share one generation budget. When a
store is configured, the pending row is committed before the provider call
and the outcome is committed afterward. The database transaction is not
held open across the network.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from app.api.chat_schemas import (
    ChatMetrics,
    ChatModelChoice,
    ChatOptionsResponse,
    ChatResponse,
    FallbackSkip,
    RoutingDetails,
    RoutingMetadata,
    ServerFeature,
)
from app.api.schemas import LLMResponse, ModelConfig, Provider, ReportedUsage
from app.chat.fallback import SkippedCandidate, select_fallback_candidates
from app.config import Settings, fallback_model_ids
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
from app.providers.errors import (
    ProviderError,
    ProviderFallbackExhaustedError,
    ProviderTimeoutError,
)
from app.providers.failure import ProviderFailure, classify_provider_error
from app.ml.runtime import RouteSelector
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
        selector: RouteSelector | None = None,
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._classifier = classifier
        self._router = router
        self._provider_factory = provider_factory
        self._store = store
        self._evaluator = evaluator
        self._selector = selector if selector is not None else RouteSelector(router, settings)

    def describe_options(self) -> ChatOptionsResponse:
        """Return catalog and server-feature status. No credentials are included."""
        return ChatOptionsResponse(
            max_input_characters=self._settings.max_input_characters,
            models=[
                ChatModelChoice(
                    model_id=model.model_id,
                    model_name=model.model_name,
                    provider=model.provider,
                    quality_tier=model.quality_tier,
                    enabled=model.enabled,
                    local=model.local,
                )
                for model in self._registry.list_all()
            ],
            quality_evaluation=ServerFeature(enabled=self._settings.quality_evaluation_enabled),
            escalation=ServerFeature(enabled=self._settings.escalation_enabled),
            fallback=ServerFeature(enabled=self._settings.fallback_enabled),
            routing_strategy=self._selector.requested_strategy,
            effective_routing_strategy=self._selector.effective_strategy,
            ml_diagnostic=self._selector.diagnostic,
        )

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        provider: Provider | None = None,
    ) -> ChatResponse:
        """Validate, route, and generate. Fallback runs only when enabled."""
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
        decision = await self._selector.select(
            user_prompt,
            assessment.tier,
            _registry_for(self._registry, provider),
        )
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
                        routing_metadata=_trace_dict(decision),
                    )
                )
            except Exception as exc:
                _log_persistence(request_id, exc)
                raise PersistenceUnavailableError() from None

        provider_client: LLMProvider | None = None
        second_provider: LLMProvider | None = None
        owned_evaluator: object | None = None
        provider_started = monotonic_now()
        try:
            if self._settings.fallback_enabled:
                return await self._complete_with_fallback(
                    user_prompt=user_prompt,
                    system=system,
                    assessment=assessment,
                    decision=decision,
                    request_id=request_id,
                    started=started,
                    restricted_provider=provider,
                )
            try:
                provider_client = self._provider_factory(decision.model)
                provider_started = monotonic_now()
                generated = await provider_client.generate(user_prompt, system)
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
                final_provider=returned.provider.value,
                routing_metadata=_trace_dict(decision),
            )
        finally:
            if provider_client is not None:
                await _close_provider(provider_client)
            if second_provider is not None:
                await _close_provider(second_provider)
            if owned_evaluator is not None:
                close_evaluator = getattr(owned_evaluator, "aclose", None)
                if close_evaluator is not None:
                    await close_evaluator()

    async def _complete_with_fallback(
        self,
        *,
        user_prompt: str,
        system: str | None,
        assessment: object,
        decision: object,
        request_id: object,
        started: float,
        restricted_provider: Provider | None,
    ) -> ChatResponse:
        """Run routing, at most one fallback, and at most one quality escalation."""
        clients: list[LLMProvider] = []
        owned_evaluator: object | None = None
        calls: list[_Call] = []
        skips: list[SkippedCandidate] = []
        generations = 0
        fallbacks = 0
        judge_calls = 0
        fallback_used = False
        fallback_reason: str | None = None
        try:
            primary = await self._one_generation(
                model=decision.model,  # type: ignore[attr-defined]
                purpose="routing",
                user_prompt=user_prompt,
                system=system,
                started=started,
                clients=clients,
            )
            assert primary is not None
            generations += 1
            calls.append(primary)
            if primary.failure is not None and primary.failure.fallback_eligible:
                extra, extra_skips, fallbacks = await self._fallback_after(
                    failed=primary,
                    calls=calls,
                    skips=skips,
                    generations=generations,
                    fallbacks=fallbacks,
                    user_prompt=user_prompt,
                    system=system,
                    started=started,
                    clients=clients,
                    restricted_provider=restricted_provider,
                )
                generations += extra
                if extra:
                    fallback_used = True
                    fallback_reason = primary.failure.category.value
                skips.extend(extra_skips)
            success = next((call for call in reversed(calls) if call.generated is not None), None)
            if success is None:
                await self._persist_fallback(
                    request_id=request_id,
                    calls=calls,
                    returned=None,
                    evaluation=None,
                    escalated=False,
                    escalation_reason=None,
                    escalation_error=None,
                    fallback_used=fallback_used,
                    fallback_reason=fallback_reason,
                    skips=skips,
                    started=started,
                    status="failed",
                )
                if fallback_used:
                    raise ProviderFallbackExhaustedError()
                assert primary.error is not None
                raise primary.error

            evaluation, owned_evaluator, judge_calls = await self._maybe_judge(
                user_prompt=user_prompt,
                answer=success.generated.content if success.generated is not None else "",
                started=started,
                judge_calls=judge_calls,
                owned_evaluator=owned_evaluator,
            )
            returned = success
            escalated = False
            escalation_reason: str | None = None
            escalation_error: str | None = None
            if (
                evaluation is not None
                and evaluation.verdict == "fail"
                and self._settings.escalation_enabled
                and generations < self._settings.max_model_attempts
                and _deadline_open(started, self._settings.request_deadline_seconds)
            ):
                target = select_escalation_target(
                    success.model,
                    self._registry,
                    self._settings,
                )
                escalation_reason = f"initial answer failed quality; {target.reason}"
                if target.model is None:
                    escalation_error = target.reason
                elif (target.model.provider.value, target.model.model_id) in _used_pairs(calls):
                    escalation_error = "This provider and model were already called."
                else:
                    escalated_call = await self._one_generation(
                        model=target.model,
                        purpose="escalation",
                        user_prompt=user_prompt,
                        system=system,
                        started=started,
                        clients=clients,
                    )
                    if escalated_call is None:
                        escalation_error = "The request deadline was reached."
                    else:
                        generations += 1
                        calls.append(escalated_call)
                        if escalated_call.generated is not None:
                            returned = escalated_call
                            escalated = True
                        else:
                            escalation_error = (
                                None if escalated_call.failure is None else escalated_call.failure.message
                            )
                            if (
                                escalated_call.failure is not None
                                and escalated_call.failure.fallback_eligible
                            ):
                                extra, extra_skips, fallbacks = await self._fallback_after(
                                    failed=escalated_call,
                                    calls=calls,
                                    skips=skips,
                                    generations=generations,
                                    fallbacks=fallbacks,
                                    user_prompt=user_prompt,
                                    system=system,
                                    started=started,
                                    clients=clients,
                                    restricted_provider=restricted_provider,
                                )
                                generations += extra
                                skips.extend(extra_skips)
                                if extra:
                                    fallback_used = True
                                    fallback_reason = escalated_call.failure.category.value
                                replacement = next(
                                    (
                                        call
                                        for call in reversed(calls)
                                        if call.generated is not None and call is not success
                                    ),
                                    None,
                                )
                                if replacement is not None:
                                    returned = replacement
            if returned.generated is None:
                returned = success
            status, warning = await self._persist_fallback(
                request_id=request_id,
                calls=calls,
                returned=returned,
                evaluation=evaluation,
                escalated=escalated,
                escalation_reason=escalation_reason,
                escalation_error=escalation_error,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                skips=skips,
                started=started,
                status="succeeded",
            )
            assert returned.generated is not None and returned.priced is not None
            total = _combined_cost(calls, evaluation)
            savings_price = returned.priced
            if total.total is None:
                savings_price = replace_savings_unknown(returned.priced)
            return _chat_response(
                request_id=str(request_id),
                generated=returned.generated,
                assessment_score=assessment.score,  # type: ignore[attr-defined]
                assessment_tier=assessment.tier,  # type: ignore[attr-defined]
                assessment_reasons=list(assessment.reasons),  # type: ignore[attr-defined]
                provider=decision.model.provider,  # type: ignore[attr-defined]
                model_id=decision.model.model_id,  # type: ignore[attr-defined]
                model_tier=decision.model.quality_tier,  # type: ignore[attr-defined]
                selection_reason=decision.selection_reason,  # type: ignore[attr-defined]
                degraded=decision.degraded,  # type: ignore[attr-defined]
                priced=savings_price,
                total_estimate=total,
                end_to_end_ms=_elapsed_ms(started),
                persistence_status=status,
                persistence_warning=warning,
                escalated=escalated,
                evaluation=evaluation,
                escalation_reason=escalation_reason,
                escalation_error=escalation_error,
                returned_model=returned.model.model_id,
                returned_attempt=calls.index(returned) + 1,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                fallback_skips=skips,
                final_provider=returned.model.provider.value,
                routing_metadata=_trace_dict(decision),
            )
        finally:
            for client in clients:
                await _close_provider(client)
            if owned_evaluator is not None:
                close_evaluator = getattr(owned_evaluator, "aclose", None)
                if close_evaluator is not None:
                    await close_evaluator()

    async def _fallback_after(
        self,
        *,
        failed: "_Call",
        calls: list["_Call"],
        skips: list[SkippedCandidate],
        generations: int,
        fallbacks: int,
        user_prompt: str,
        system: str | None,
        started: float,
        clients: list[LLMProvider],
        restricted_provider: Provider | None,
    ) -> tuple[int, list[SkippedCandidate], int]:
        """Call eligible fallback models. Returns generations added, new skips, fallback count."""
        del skips
        remaining_fallbacks = self._settings.max_fallback_attempts - fallbacks
        remaining_generations = self._settings.max_model_attempts - generations
        limit = min(remaining_fallbacks, remaining_generations)
        candidates, skipped = select_fallback_candidates(
            fallback_model_ids(self._settings),
            self._registry,
            used=_used_pairs(calls),
            failed_provider=failed.model.provider,
            restricted_provider=restricted_provider,
            limit=max(limit, 0),
        )
        if limit < 1:
            skipped = [
                *skipped,
                *[
                    SkippedCandidate(model_id, "The generation attempt limit was reached.")
                    for model_id in fallback_model_ids(self._settings)
                    if model_id not in {item.model_id for item in skipped}
                    and model_id not in {call.model.model_id for call in calls}
                ],
            ]
            return 0, skipped, fallbacks
        added = 0
        for model in candidates:
            if not _deadline_open(started, self._settings.request_deadline_seconds):
                skipped.append(SkippedCandidate(model.model_id, "The request deadline was reached."))
                break
            if generations + added >= self._settings.max_model_attempts:
                skipped.append(
                    SkippedCandidate(model.model_id, "The generation attempt limit was reached.")
                )
                break
            if fallbacks >= self._settings.max_fallback_attempts:
                break
            call = await self._one_generation(
                model=model,
                purpose="fallback",
                user_prompt=user_prompt,
                system=system,
                started=started,
                clients=clients,
            )
            if call is None:
                skipped.append(SkippedCandidate(model.model_id, "The request deadline was reached."))
                break
            calls.append(call)
            added += 1
            fallbacks += 1
            if call.generated is not None or (
                call.failure is not None and not call.failure.fallback_eligible
            ):
                break
        return added, skipped, fallbacks

    async def _one_generation(
        self,
        *,
        model: ModelConfig,
        purpose: str,
        user_prompt: str,
        system: str | None,
        started: float,
        clients: list[LLMProvider],
    ) -> "_Call | None":
        timeout = _call_timeout(self._settings, model, started)
        if timeout is None:
            exc = ProviderTimeoutError("The request deadline was reached")
            failure = classify_provider_error(exc)
            return _Call(
                model=model,
                purpose=purpose,
                generated=None,
                error=exc,
                failure=failure,
                latency_ms=0,
                priced=_price_usage(None, model.model_id, model.provider.value, self._settings),
            )
        call_started = monotonic_now()
        client: LLMProvider | None = None
        try:
            client = self._provider_factory(model)
            clients.append(client)
            generated = await asyncio.wait_for(
                client.generate(user_prompt, system),
                timeout=timeout,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            exc: ProviderError = ProviderTimeoutError(f"{model.provider.value} request timed out")
            failure = classify_provider_error(exc)
            return _Call(
                model=model,
                purpose=purpose,
                generated=None,
                error=exc,
                failure=failure,
                latency_ms=(monotonic_now() - call_started) * 1000,
                priced=_price_usage(None, model.model_id, model.provider.value, self._settings),
            )
        except ProviderError as exc:
            failure = classify_provider_error(exc)
            return _Call(
                model=model,
                purpose=purpose,
                generated=None,
                error=exc,
                failure=failure,
                latency_ms=(monotonic_now() - call_started) * 1000,
                priced=_price_usage(
                    failure.usage,
                    model.model_id,
                    model.provider.value,
                    self._settings,
                ),
            )
        return _Call(
            model=model,
            purpose=purpose,
            generated=generated,
            error=None,
            failure=None,
            latency_ms=generated.latency_ms,
            priced=_price_success(generated, model.model_id, self._settings),
        )

    async def _maybe_judge(
        self,
        *,
        user_prompt: str,
        answer: str,
        started: float,
        judge_calls: int,
        owned_evaluator: object | None,
    ) -> tuple[EvaluationResult | None, object | None, int]:
        if not self._settings.quality_evaluation_enabled:
            return None, owned_evaluator, judge_calls
        if judge_calls >= self._settings.max_judge_calls:
            return (
                EvaluationResult(
                    verdict="unknown",
                    score=None,
                    reasons=("The judge call limit was reached.",),
                    method="live",
                ),
                owned_evaluator,
                judge_calls,
            )
        if not _deadline_open(started, self._settings.request_deadline_seconds):
            return (
                EvaluationResult(
                    verdict="unknown",
                    score=None,
                    reasons=("The request deadline was reached before the quality check.",),
                    method="live",
                ),
                owned_evaluator,
                judge_calls,
            )
        evaluator = self._evaluator
        if evaluator is None:
            evaluator = build_live_evaluator(self._settings)
            owned_evaluator = evaluator
        remaining = _remaining_seconds(started, self._settings.request_deadline_seconds)
        try:
            result = await asyncio.wait_for(
                evaluator.evaluate(user_prompt, answer),  # type: ignore[attr-defined]
                timeout=remaining,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            result = EvaluationResult(
                verdict="unknown",
                score=None,
                reasons=("The request deadline was reached before the quality check.",),
                method="live",
            )
        except Exception:
            result = EvaluationResult(
                verdict="error",
                score=None,
                reasons=("The quality evaluator failed.",),
                method="live",
                error_message="The quality evaluator failed.",
            )
        return result, owned_evaluator, judge_calls + 1

    async def _persist_fallback(
        self,
        *,
        request_id: object,
        calls: list["_Call"],
        returned: "_Call | None",
        evaluation: EvaluationResult | None,
        escalated: bool,
        escalation_reason: str | None,
        escalation_error: str | None,
        fallback_used: bool,
        fallback_reason: str | None,
        skips: list[SkippedCandidate],
        started: float,
        status: str,
    ) -> tuple[str, str | None]:
        if self._store is None:
            return "not_configured", None
        total = _combined_cost(calls, evaluation)
        outcomes = [
            _call_outcome(
                request_id=request_id,
                call=call,
                attempt_number=index,
                started=started,
                settings=self._settings,
            )
            for index, call in enumerate(calls, start=1)
        ]
        anchor = outcomes[-1] if outcomes else _failed_anchor(request_id, started)
        if returned is not None and returned.priced is not None and returned.generated is not None:
            anchor = _call_outcome(
                request_id=request_id,
                call=returned,
                attempt_number=calls.index(returned) + 1,
                started=started,
                settings=self._settings,
            )
        savings_total = None if total.total is None else anchor.estimated_savings
        final = _with_check(
            replace(
                anchor,
                status=status,
                response_text=None if returned is None or returned.generated is None else returned.generated.content,
                total_cost=total.total,
                cost_completeness=total.completeness.value,
                estimated_savings=savings_total,
                end_to_end_latency_ms=_elapsed_ms(started),
                error_message=None if status == "succeeded" else anchor.error_message,
            ),
            total_estimate=total,
            escalated=escalated,
            evaluation=evaluation,
            escalation_reason=escalation_reason,
            escalation_error=escalation_error,
            final_model_id=None if returned is None else returned.model.model_id,
            returned_attempt=1 if returned is None else calls.index(returned) + 1,
        )
        final = replace(
            final,
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
            fallback_skips=tuple((item.model_id, item.reason) for item in skips),
            final_provider=None if returned is None else returned.model.provider.value,
        )
        try:
            save_attempt = getattr(self._store, "save_attempt", None)
            finalize = getattr(self._store, "finalize_request", None)
            if save_attempt is not None and finalize is not None:
                for outcome in outcomes:
                    await save_attempt(outcome)
                await finalize(final)
            elif status == "succeeded":
                await self._store.record_success(final)
            else:
                await self._store.record_failure(final)
        except Exception as exc:
            _log_persistence(request_id, exc)
            if status != "succeeded":
                return "failed", None
            return "failed", _PERSISTENCE_WARNING
        return "stored", None

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
        failure = classify_provider_error(exc)
        usage = failure.usage
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
                    error_message=failure.message,
                    response_text=None,
                    started=started,
                    purpose="routing",
                    error_category=failure.category.value,
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
                replace(
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
                    ),
                    final_model_id=priced.configured_model_id,
                    final_provider=generated.provider.value,
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
        final_provider=outcome.provider,
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
    purpose: str = "routing",
    error_category: str | None = None,
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
        purpose=purpose,
        error_category=error_category,
    )


def _trace_dict(decision: object) -> dict[str, object] | None:
    trace = getattr(decision, "trace", None)
    if trace is None:
        return None
    return trace.to_dict()


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
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    fallback_skips: list[SkippedCandidate] | None = None,
    final_provider: str | None = None,
    routing_metadata: dict[str, object] | None = None,
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
            routing_metadata=None
            if routing_metadata is None
            else RoutingMetadata.model_validate(routing_metadata),
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
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        fallback_skips=[
            FallbackSkip(model_id=item.model_id, reason=item.reason)
            for item in (fallback_skips or [])
        ],
        final_provider=final_provider,
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


@dataclass
class _Call:
    """One generation inside a fallback-enabled request."""

    model: ModelConfig
    purpose: str
    generated: LLMResponse | None
    error: ProviderError | None
    failure: ProviderFailure | None
    latency_ms: float
    priced: _Priced | None


def _registry_for(registry: ModelRegistry, provider: Provider | None) -> ModelRegistry:
    if provider is None:
        return registry
    return ModelRegistry([model for model in registry.list_all() if model.provider is provider])


def _deadline_open(started: float, deadline_seconds: float) -> bool:
    return _remaining_seconds(started, deadline_seconds) > 0


def _remaining_seconds(started: float, deadline_seconds: float) -> float:
    return deadline_seconds - (monotonic_now() - started)


def _call_timeout(settings: Settings, model: ModelConfig, started: float) -> float | None:
    remaining = _remaining_seconds(started, settings.request_deadline_seconds)
    if remaining <= 0:
        return None
    per_attempt = {
        Provider.OPENAI: settings.openai_timeout_seconds,
        Provider.ANTHROPIC: settings.anthropic_timeout_seconds,
        Provider.OLLAMA: settings.ollama_timeout_seconds,
    }[model.provider]
    return min(per_attempt, remaining)


def _used_pairs(calls: list[_Call]) -> set[tuple[str, str]]:
    return {(call.model.provider.value, call.model.model_id) for call in calls}


def _combined_cost(calls: list[_Call], evaluation: EvaluationResult | None) -> CostEstimate:
    parts = [call.priced.estimate for call in calls if call.priced is not None]
    if evaluation is not None and evaluation.judge_cost is not None:
        parts.append(evaluation.judge_cost)
    if not parts:
        return CostEstimate(total=None, completeness=CostCompleteness.UNKNOWN)
    return combine_costs(parts)


def replace_savings_unknown(priced: _Priced) -> _Priced:
    """Hide a savings figure when the request total cannot be priced."""
    return _Priced(
        configured_model_id=priced.configured_model_id,
        estimate=priced.estimate,
        baseline_model=priced.baseline_model,
        baseline_cost=priced.baseline_cost,
        estimated_savings=None,
        snapshot=priced.snapshot,
    )


def _call_outcome(
    *,
    request_id: object,
    call: _Call,
    attempt_number: int,
    started: float,
    settings: Settings,
) -> AttemptOutcome:
    generated = call.generated
    usage = None if call.failure is None else call.failure.usage
    if generated is not None:
        usage = usage_from_response(generated)
    priced = call.priced or _empty_price(call.model.model_id)
    outcome = _outcome(
        request_id=request_id,
        status="succeeded" if generated is not None else "failed",
        provider=call.model.provider.value,
        configured_model_id=call.model.model_id,
        reported_model_id=None if generated is None else generated.model,
        usage=usage,
        provider_latency_ms=call.latency_ms,
        priced=priced,
        error_message=None if call.failure is None else call.failure.message,
        response_text=None if generated is None else generated.content,
        started=started,
        purpose=call.purpose,
        error_category=None if call.failure is None else call.failure.category.value,
    )
    del settings
    return _replace_attempt(outcome, attempt_number)


def _failed_anchor(request_id: object, started: float) -> AttemptOutcome:
    return _outcome(
        request_id=request_id,
        status="failed",
        provider="unavailable",
        configured_model_id="unavailable",
        reported_model_id=None,
        usage=None,
        provider_latency_ms=0,
        priced=_empty_price("unavailable"),
        error_message="The request deadline was reached.",
        response_text=None,
        started=started,
        purpose="routing",
        error_category="timeout",
    )


def _log_persistence(request_id: object, exc: Exception) -> None:
    logger.error(
        "Failed to persist request %s (%s)",
        request_id,
        type(exc).__name__,
    )
