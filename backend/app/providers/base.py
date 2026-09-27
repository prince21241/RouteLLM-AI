"""Abstract provider contract.

Concrete clients are added in a later phase. Implementations return token
counts and latency, and leave ``estimated_cost`` empty.
"""

from abc import ABC, abstractmethod

from app.api.schemas import LLMResponse


class LLMProvider(ABC):
    """Async text generation shared by every provider client."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a completion for ``prompt``."""
        raise NotImplementedError
