"""Abstract provider contract.

Concrete clients return token counts and latency, and leave ``estimated_cost``
empty. Pricing is filled in later.
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
