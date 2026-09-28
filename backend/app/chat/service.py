"""Classify a request, select one model, and call that provider once.

A provider failure stops the request. This service does not try another
model, score quality, or escalate. When a store is configured, the pending
row is committed before the provider call and the outcome is committed
afterward. The database transaction is not held open across the network.
"""

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

from app.api.chat_schemas import ChatMetrics, ChatResponse, RoutingDetails
from app.api.schemas import LLMResponse, ModelConfig, Provider, ReportedUsage
from app.config import Settings
from app.db.records import AttemptOutcome, PendingRequest
from app.db.store import RequestStore
from app.pricing.money import money_to_api
from app.pricing.service import (
    CostCompleteness,
    CostEstimate,
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
    ) -> None:
        self._settings = settings
        self._registry = registry
        self._classifier = classifier
        self._router = router
        self._provider_factory = provider_factory
        self._store = store

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
            persistence_status, persistence_warning = await self._record_success(
                request_id=request_id,
                generated=generated,
                priced=priced,
                started=started,
            )
            return _chat_response(
                request_id=str(request_id),
                generated=generated,
                assessment_score=assessment.score,
                assessment_tier=assessment.tier,
                assessment_reasons=list(assessment.reasons),
                provider=decision.model.provider,
                model_id=decision.model.model_id,
                model_tier=decision.model.quality_tier,
                selection_reason=decision.selection_reason,
                degraded=decision.degraded,
                priced=priced,
                end_to_end_ms=_elapsed_ms(started),
                persistence_status=persistence_status,
                persistence_warning=persistence_warning,
            )
        finally:
            if provider is not None:
                await _close_provider(provider)

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
    end_to_end_ms: Decimal,
    persistence_status: str,
    persistence_warning: str | None,
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
            escalated=False,
        ),
        metrics=ChatMetrics(
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            latency_ms=generated.latency_ms,
            cost=money_to_api(priced.estimate.total),
            cost_completeness=priced.estimate.completeness.value,  # type: ignore[arg-type]
            quality_score=None,
        ),
        baseline_model=priced.baseline_model,
        baseline_cost=money_to_api(priced.baseline_cost),
        estimated_savings=money_to_api(priced.estimated_savings),
        savings_basis="same_token_volume",
        persistence_status=persistence_status,  # type: ignore[arg-type]
        persistence_warning=persistence_warning,
        end_to_end_latency_ms=float(end_to_end_ms),
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
