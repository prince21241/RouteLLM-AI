"""Send one OpenAI Responses API request through OpenAIProvider.

This script is intentionally outside the pytest suite. It performs one paid
network call and must be run manually. It does not retry.
"""

import asyncio
import re
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.api.schemas import LLMResponse
from app.config import get_settings
from app.providers.errors import ProviderError
from app.providers.openai import OpenAIProvider

MODEL = "gpt-5-nano"
PROMPT = "Reply with exactly: Connection successful"
TIMEOUT_SECONDS = 30.0

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]+"),
    re.compile(r"Bearer\s+\S+", re.IGNORECASE),
)


def sanitize(value: str, secret: str) -> str:
    """Remove credentials from text that may be printed."""
    redacted = value.replace(secret, "[redacted]") if secret else value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


async def _once(api_key: str, max_output_tokens: int) -> LLMResponse:
    provider = OpenAIProvider(
        api_key,
        model=MODEL,
        max_output_tokens=max_output_tokens,
        timeout_seconds=TIMEOUT_SECONDS,
    )
    try:
        return await provider.generate(PROMPT)
    finally:
        await provider.aclose()


def main() -> int:
    settings = get_settings()
    key = settings.openai_api_key
    if key is None or key.get_secret_value().strip() == "":
        print("OPENAI_API_KEY is not configured")
        return 1

    secret = key.get_secret_value()
    try:
        response = asyncio.run(_once(secret, settings.openai_max_output_tokens))
    except ProviderError as exc:
        print(sanitize(str(exc), secret))
        return 1

    print(f"model={sanitize(response.model, secret)}")
    print(f"text={sanitize(response.content, secret)}")
    print(
        f"input_tokens={response.input_tokens} output_tokens={response.output_tokens}"
    )
    print(f"latency_ms={response.latency_ms:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
