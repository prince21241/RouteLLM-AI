"""Environment-backed settings.

API keys stay in ``SecretStr``. Repr and serialization show a redacted
placeholder, so logs of the settings object do not include the key.
"""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_serializer, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration from the process environment and the repo-root ``.env``.

    Provider clients are created by callers, not while settings load. Missing
    API keys stay empty so ``/health`` can run.
    """

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_model: str = Field(default="gpt-5-nano", min_length=1)
    openai_max_output_tokens: int = Field(default=256, ge=1)
    openai_timeout_seconds: float = Field(default=30.0, gt=0)
    anthropic_api_key: SecretStr | None = Field(default=None, repr=False)
    anthropic_model: str = Field(default="claude-sonnet-4-6", min_length=1)
    anthropic_max_tokens: int = Field(default=1024, ge=1)
    anthropic_timeout_seconds: float = Field(default=30.0, gt=0)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = Field(default="llama3.2", min_length=1)
    ollama_timeout_seconds: float = Field(default=60.0, gt=0)
    low_complexity_threshold: float = Field(default=0.30, ge=0, le=1)
    high_complexity_threshold: float = Field(default=0.70, ge=0, le=1)
    min_quality_score: float = Field(default=0.75, ge=0, le=1)
    max_model_attempts: int = Field(default=3, ge=1)

    @field_validator("openai_api_key", "anthropic_api_key", mode="before")
    @classmethod
    def blank_api_key_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_serializer("openai_api_key", "anthropic_api_key")
    def redact_api_key(self, value: SecretStr | None) -> str | None:
        if value is None:
            return None
        return "**********"

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
    """
    return Settings()
