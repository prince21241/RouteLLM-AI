"""Settings defaults, environment overrides, and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.api.schemas import QualityTier
from app.config import Settings, get_settings, migration_database_url
from app.main import create_app


def test_defaults(settings: Settings) -> None:
    assert settings.openai_api_key is None
    assert settings.openai_model == "gpt-5-nano"
    assert settings.openai_max_output_tokens == 256
    assert settings.openai_timeout_seconds == 30.0
    assert settings.openai_routing_enabled is True
    assert settings.openai_quality_tier == QualityTier.LOW
    assert settings.anthropic_api_key is None
    assert settings.anthropic_model == "claude-sonnet-4-6"
    assert settings.anthropic_max_tokens == 1024
    assert settings.anthropic_timeout_seconds == 30.0
    assert settings.anthropic_routing_enabled is False
    assert settings.anthropic_quality_tier == QualityTier.HIGH
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.ollama_model == "llama3.2"
    assert settings.ollama_timeout_seconds == 60.0
    assert settings.ollama_routing_enabled is False
    assert settings.ollama_quality_tier == QualityTier.LOW
    assert settings.routing_preference == ""
    assert settings.max_input_characters == 8000
    assert settings.low_complexity_threshold == 0.30
    assert settings.high_complexity_threshold == 0.70
    assert settings.min_quality_score == 0.75
    assert settings.max_model_attempts == 3
    assert settings.database_url is None
    assert settings.premium_baseline_model == "claude-sonnet-4-6"
    assert settings.quality_evaluation_enabled is False
    assert settings.escalation_enabled is False
    assert settings.escalation_model == ""
    assert settings.quality_judge_enabled is False
    assert settings.quality_judge_model == "gpt-5-nano"
    assert settings.evaluation_baseline_model == "gpt-5-nano"


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

    with_database = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://routellm:secret-pass@localhost/routellm",
    )
    rendered_database = repr(with_database)
    dumped_database = str(with_database.model_dump())
    assert "secret-pass" not in rendered_database
    assert "secret-pass" not in dumped_database
    assert with_database.database_url is not None
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


@pytest.mark.parametrize(
    "overrides",
    [
        {"openai_max_output_tokens": 0},
        {"openai_timeout_seconds": 0},
        {"anthropic_max_tokens": 0},
        {"anthropic_timeout_seconds": -1},
        {"ollama_model": ""},
        {"ollama_timeout_seconds": 0},
        {"max_input_characters": 0},
    ],
)
def test_invalid_provider_settings(overrides: dict[str, float | int | str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_blank_environment_does_not_hide_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty or whitespace DATABASE_URL must not override the env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql+asyncpg://db.example/routellm\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", "")

    from_empty = Settings(_env_file=env_file)

    monkeypatch.setenv("DATABASE_URL", "   ")
    from_whitespace = Settings(_env_file=env_file)

    assert from_empty.database_url == "postgresql+asyncpg://db.example/routellm"
    assert from_whitespace.database_url == "postgresql+asyncpg://db.example/routellm"
    assert migration_database_url("", from_empty) == "postgresql+asyncpg://db.example/routellm"
    assert migration_database_url("   ", from_whitespace) == "postgresql+asyncpg://db.example/routellm"


def test_process_environment_still_overrides_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql+asyncpg://db.example/from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://db.example/from-env")

    settings = Settings(_env_file=env_file)

    assert settings.database_url == "postgresql+asyncpg://db.example/from-env"
    assert (
        migration_database_url("postgresql+asyncpg://db.example/explicit", settings)
        == "postgresql+asyncpg://db.example/explicit"
    )


def test_settings_env_file_is_the_absolute_repository_root() -> None:
    from app import config as config_module

    expected = Path(config_module.__file__).resolve().parents[2] / ".env"

    assert config_module._ENV_FILE.is_absolute()
    assert config_module._ENV_FILE == expected.resolve()


def test_migration_url_requires_configuration() -> None:
    with pytest.raises(RuntimeError, match="DATABASE_URL is not configured"):
        migration_database_url(None, Settings(_env_file=None))


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
