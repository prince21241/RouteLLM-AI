"""FastAPI application factory."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings
from app.routing.model_registry import ModelRegistry


class HealthResponse(BaseModel):
    status: Literal["ok"]


def create_app(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
) -> FastAPI:
    """Build the API.

    Defaults are enough to serve ``/health`` with no API keys, database, or
    running Ollama process. Pass ``settings`` or ``registry`` to override them.
    """
    app = FastAPI(title="RouteLLM AI", version=__version__)
    app.state.settings = get_settings() if settings is None else settings
    app.state.registry = ModelRegistry() if registry is None else registry

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app


app = create_app()
