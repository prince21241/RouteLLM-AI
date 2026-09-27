"""Settings defaults, environment overrides, and validation."""

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.main import create_app


def test_defaults(settings: Settings) -> None:
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.low_complexity_threshold == 0.30
    assert settings.high_complexity_threshold == 0.70
    assert settings.min_quality_score == 0.75
    assert settings.max_model_attempts == 3


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LOW_COMPLEXITY_THRESHOLD", "0.20")
    monkeypatch.setenv("HIGH_COMPLEXITY_THRESHOLD", "0.80")
    monkeypatch.setenv("MIN_QUALITY_SCORE", "0.50")
    monkeypatch.setenv("MAX_MODEL_ATTEMPTS", "4")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder-openai-key")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.low_complexity_threshold == 0.20
    assert settings.high_complexity_threshold == 0.80
    assert settings.min_quality_score == 0.50
    assert settings.max_model_attempts == 4
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "placeholder-openai-key"


def test_blank_api_keys_are_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "   ")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    settings = Settings(_env_file=None)

    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_secrets_are_redacted() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="placeholder-openai-key",
        anthropic_api_key="placeholder-anthropic-key",
    )

    rendered = repr(settings)
    dumped = str(settings.model_dump())

    assert "placeholder-openai-key" not in rendered
    assert "placeholder-anthropic-key" not in rendered
    assert "placeholder-openai-key" not in dumped
    assert "placeholder-anthropic-key" not in dumped
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "placeholder-openai-key"
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "placeholder-anthropic-key"


def test_threshold_boundaries() -> None:
    settings = Settings(
        _env_file=None,
        low_complexity_threshold=0,
        high_complexity_threshold=1,
        min_quality_score=0,
        max_model_attempts=1,
    )

    assert settings.low_complexity_threshold == 0
    assert settings.high_complexity_threshold == 1
    assert settings.min_quality_score == 0
    assert settings.max_model_attempts == 1

    top_quality = Settings(_env_file=None, min_quality_score=1)
    assert top_quality.min_quality_score == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"low_complexity_threshold": -0.01},
        {"high_complexity_threshold": 1.01},
        {"low_complexity_threshold": 0.70, "high_complexity_threshold": 0.70},
        {"low_complexity_threshold": 0.80, "high_complexity_threshold": 0.30},
        {"min_quality_score": -0.01},
        {"min_quality_score": 1.01},
        {"max_model_attempts": 0},
    ],
)
def test_invalid_settings(overrides: dict[str, float | int]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_invalid_thresholds_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOW_COMPLEXITY_THRESHOLD", "0.90")
    monkeypatch.setenv("HIGH_COMPLEXITY_THRESHOLD", "0.20")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_are_injectable(settings: Settings) -> None:
    custom = Settings(_env_file=None, max_model_attempts=5)
    app = create_app(settings=custom)

    assert app.state.settings is custom
    assert app.state.settings is not settings
    assert get_settings() is not custom
