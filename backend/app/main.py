"""FastAPI application factory."""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import __version__
from app.api.chat import register_chat_route
from app.api.dashboard import register_dashboard_routes
from app.api.history import register_history_routes
from app.api.schemas import ModelConfig
from app.chat.service import ChatService
from app.config import Settings, get_settings
from app.db.session import build_store
from app.db.store import RequestStore
from app.providers.base import LLMProvider
from app.providers.factory import build_provider
from app.routing.catalog import build_catalog
from app.routing.complexity import ComplexityClassifier
from app.routing.model_registry import ModelRegistry
from app.routing.router import ModelRouter, parse_preference

logger = logging.getLogger("app.ready")
_STATIC_DIR = Path(__file__).resolve().parent / "static"


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    detail: str | None = None


def create_app(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    classifier: ComplexityClassifier | None = None,
    router: ModelRouter | None = None,
    provider_factory: Callable[[ModelConfig], LLMProvider] | None = None,
    store: RequestStore | None = None,
    evaluator: object | None = None,
    selector: object | None = None,
) -> FastAPI:
    """Build the API.

    Defaults are enough to serve ``/health`` with no API keys, database, or
    running Ollama process. Provider clients are created per chat request,
    not during startup. Tables are not created here. Pass dependencies to
    override them in tests.
    """
    resolved_settings = get_settings() if settings is None else settings
    resolved_registry = build_catalog(resolved_settings) if registry is None else registry
    resolved_router = (
        ModelRouter(parse_preference(resolved_settings.routing_preference))
        if router is None
        else router
    )
    resolved_factory = (
        provider_factory
        if provider_factory is not None
        else (lambda model: build_provider(model, resolved_settings))
    )
    db_engine = None
    resolved_store = store
    if resolved_store is None and resolved_settings.database_url:
        resolved_store, db_engine = build_store(resolved_settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        engine = getattr(app.state, "db_engine", None)
        if engine is not None:
            await engine.dispose()

    app = FastAPI(title="RouteLLM AI", version=__version__, lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.registry = resolved_registry
    app.state.store = resolved_store
    app.state.db_engine = db_engine
    service = ChatService(
        resolved_settings,
        resolved_registry,
        classifier or ComplexityClassifier(),
        resolved_router,
        resolved_factory,
        store=resolved_store,
        evaluator=evaluator,
        selector=selector,
    )
    register_chat_route(app, service)
    register_history_routes(app)
    register_dashboard_routes(app)
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/dashboard")
    def dashboard() -> FileResponse:
        return FileResponse(_STATIC_DIR / "dashboard.html")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> JSONResponse | ReadyResponse:
        if app.state.store is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "detail": "Database is not configured"},
            )
        try:
            await app.state.store.ping()
        except Exception as exc:
            logger.error("Readiness check failed (%s)", type(exc).__name__)
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "detail": "Database is not ready"},
            )
        return ReadyResponse(status="ok")

    return app


app = create_app()
