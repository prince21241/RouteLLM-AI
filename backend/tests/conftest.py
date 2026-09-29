"""Fixtures that keep tests offline and independent of local credentials."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app

_SETTINGS_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_MAX_OUTPUT_TOKENS",
    "OPENAI_TIMEOUT_SECONDS",
    "OPENAI_ROUTING_ENABLED",
    "OPENAI_QUALITY_TIER",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_MAX_TOKENS",
    "ANTHROPIC_TIMEOUT_SECONDS",
    "ANTHROPIC_ROUTING_ENABLED",
    "ANTHROPIC_QUALITY_TIER",
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT_SECONDS",
    "OLLAMA_ROUTING_ENABLED",
    "OLLAMA_QUALITY_TIER",
    "ROUTING_PREFERENCE",
    "MAX_INPUT_CHARACTERS",
    "LOW_COMPLEXITY_THRESHOLD",
    "HIGH_COMPLEXITY_THRESHOLD",
    "MIN_QUALITY_SCORE",
    "MAX_MODEL_ATTEMPTS",
    "DATABASE_URL",
    "PREMIUM_BASELINE_MODEL",
    "QUALITY_EVALUATION_ENABLED",
    "ESCALATION_ENABLED",
    "ESCALATION_MODEL",
    "QUALITY_JUDGE_ENABLED",
    "QUALITY_JUDGE_MODEL",
    "EVALUATION_BASELINE_MODEL",
    "FALLBACK_ENABLED",
    "FALLBACK_MODELS",
    "MAX_FALLBACK_ATTEMPTS",
    "REQUEST_DEADLINE_SECONDS",
    "MAX_JUDGE_CALLS",
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
