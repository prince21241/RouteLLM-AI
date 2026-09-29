"""POST /api/v1/chat."""

from fastapi import FastAPI, HTTPException

from app.api.chat_schemas import ChatOptionsResponse, ChatRequest, ChatResponse
from app.chat.service import ChatService, InputValidationError, PersistenceUnavailableError
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUpstreamError,
)
from app.routing.router import NoModelsAvailableError


def register_chat_route(app: FastAPI, service: ChatService) -> None:
    """Add the chat route. The service closes the provider it creates."""

    @app.get("/api/v1/chat/options", response_model=ChatOptionsResponse)
    def chat_options() -> ChatOptionsResponse:
        return service.describe_options()

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest) -> ChatResponse:
        try:
            return await service.complete(
                body.prompt,
                body.system_prompt,
                provider=body.provider,
            )
        except InputValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except NoModelsAvailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except PersistenceUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
        except ProviderError as exc:
            raise HTTPException(status_code=_status_for(exc), detail=str(exc)) from None

    return None


def _status_for(exc: ProviderError) -> int:
    if isinstance(exc, ProviderAuthenticationError):
        if "HTTP 403" in str(exc):
            return 403
        return 401
    if isinstance(exc, ProviderRateLimitError):
        return 429
    if isinstance(exc, ProviderTimeoutError):
        return 504
    if isinstance(exc, ProviderConfigurationError):
        return 503
    if isinstance(exc, (ProviderUpstreamError, ProviderResponseError)):
        return 502
    return 502
