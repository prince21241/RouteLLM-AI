"""Fixtures that keep tests offline and independent of local credentials."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app

_SETTINGS_ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OLLAMA_BASE_URL",
    "LOW_COMPLEXITY_THRESHOLD",
    "HIGH_COMPLEXITY_THRESHOLD",
    "MIN_QUALITY_SCORE",
    "MAX_MODEL_ATTEMPTS",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Drop RouteLLM settings from the process environment for each test."""
    for name in _SETTINGS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Default settings that do not read a developer ``.env`` file."""
    return Settings(_env_file=None)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client
