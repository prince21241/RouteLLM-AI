"""In-memory catalog of model metadata."""

from collections.abc import Sequence

from app.api.schemas import ModelConfig


class DuplicateModelError(ValueError):
    """Raised when a model ID is registered more than once."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Duplicate model ID: {model_id}")


class UnknownModelError(LookupError):
    """Raised when a model ID is not in the registry."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Unknown model ID: {model_id}")


class ModelRegistry:
    """Injectable in-memory registry. It may be empty at startup."""

    def __init__(self, models: Sequence[ModelConfig] | None = None) -> None:
        self._models: dict[str, ModelConfig] = {}
        for model in models or ():
            self.register(model)

    def register(self, model: ModelConfig) -> None:
        """Store ``model``. Model IDs must be unique."""
        if model.model_id in self._models:
            raise DuplicateModelError(model.model_id)
        self._models[model.model_id] = model

    def get(self, model_id: str) -> ModelConfig:
        """Return the model stored under ``model_id``, including disabled models."""
        try:
            return self._models[model_id]
        except KeyError:
            raise UnknownModelError(model_id) from None

    def list_all(self) -> list[ModelConfig]:
        """Return every model in registration order, including disabled ones."""
        return list(self._models.values())

    def list_enabled(self) -> list[ModelConfig]:
        """Return enabled models in registration order."""
        return [model for model in self._models.values() if model.enabled]
