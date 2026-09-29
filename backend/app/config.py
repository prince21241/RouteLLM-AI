"""Environment-backed settings.

API keys stay in ``SecretStr``. Repr and serialization show a redacted
placeholder, so logs of the settings object do not include the key.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_serializer, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.api.schemas import QualityTier

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = (_REPO_ROOT / ".env").resolve()


def _is_blank(value: object) -> bool:
    """Return whether a settings value is empty or only whitespace."""
    return isinstance(value, str) and value.strip() == ""


class _SkipBlankEnv:
    """Drop blank process environment values so the env file can supply them.

    Pydantic settings prefer process environment variables over ``.env``.
    An empty ``DATABASE_URL`` in the shell would otherwise hide the file.
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __call__(self) -> dict[str, object]:
        loaded = self._inner()
        return {key: value for key, value in loaded.items() if not _is_blank(value)}


class Settings(BaseSettings):
    """Runtime configuration from the process environment and the repo-root ``.env``.

    Provider clients are created by callers, not while settings load. Missing
    API keys stay empty so ``/health`` can run. A blank process environment
    value does not override the same name in the repository-root ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        env_ignore_empty=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: object,
        env_settings: object,
        dotenv_settings: object,
        file_secret_settings: object,
    ) -> tuple[object, ...]:
        return (
            init_settings,
            _SkipBlankEnv(env_settings),
            dotenv_settings,
            file_secret_settings,
        )

    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_model: str = Field(default="gpt-5-nano", min_length=1)
    openai_max_output_tokens: int = Field(default=256, ge=1)
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    openai_routing_enabled: bool = True
    openai_quality_tier: QualityTier = QualityTier.LOW
    anthropic_api_key: SecretStr | None = Field(default=None, repr=False)
    anthropic_model: str = Field(default="claude-sonnet-4-6", min_length=1)
    anthropic_max_tokens: int = Field(default=1024, ge=1)
    anthropic_timeout_seconds: float = Field(default=30.0, gt=0)
    anthropic_routing_enabled: bool = False
    anthropic_quality_tier: QualityTier = QualityTier.HIGH
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = Field(default="llama3.2", min_length=1)
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)
    ollama_routing_enabled: bool = False
    ollama_quality_tier: QualityTier = QualityTier.LOW
    routing_preference: str = ""
    max_input_characters: int = Field(default=8000, ge=1)
    low_complexity_threshold: float = Field(default=0.30, ge=0, le=1)
    high_complexity_threshold: float = Field(default=0.70, ge=0, le=1)
    min_quality_score: float = Field(default=0.75, ge=0, le=1)
    max_model_attempts: int = Field(default=3, ge=1)
    database_url: str | None = Field(default=None, repr=False)
    premium_baseline_model: str = "claude-sonnet-4-6"
    quality_evaluation_enabled: bool = False
    escalation_enabled: bool = False
    escalation_model: str = ""
    quality_judge_enabled: bool = False
    quality_judge_model: str = Field(default="gpt-5-nano", min_length=1)
    evaluation_baseline_model: str = Field(default="gpt-5-nano", min_length=1)

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_api_key_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def blank_database_url_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator(
        "premium_baseline_model",
        "escalation_model",
        "evaluation_baseline_model",
        "quality_judge_model",
        mode="before",
    )
    @classmethod
    def strip_model_setting(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_serializer("openai_api_key", "anthropic_api_key")
    def redact_api_key(self, value: SecretStr | None) -> str | None:
        if value is None:
            return None
        return "**********"

    @field_serializer("database_url")
    def redact_database_url(self, value: str | None) -> str | None:
        if value is None:
            return None
        return "[redacted]"

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> Self:
        if self.low_complexity_threshold >= self.high_complexity_threshold:
            raise ValueError(
                "low_complexity_threshold must be less than high_complexity_threshold"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load process settings once.

    Pass a ``Settings`` instance to ``create_app`` in tests. After changing
    environment variables in-process, call ``get_settings.cache_clear()``.
    The env file is the absolute repository-root ``.env``.
    """
    return Settings()


def migration_database_url(
    explicit_url: str | None = None,
    settings: Settings | None = None,
) -> str:
    """Return the database URL Alembic should use.

    A non-blank URL passed by Alembic wins, which keeps the disposable test
    database separate. Otherwise this uses the same settings load as the API.
    The URL is not logged.
    """
    if isinstance(explicit_url, str) and explicit_url.strip():
        return explicit_url.strip()
    resolved = get_settings() if settings is None else settings
    url = resolved.database_url
    if url is None or url.strip() == "":
        raise RuntimeError(
            "DATABASE_URL is not configured. "
            f"Set it in the process environment or in {_ENV_FILE}."
        )
    return url
