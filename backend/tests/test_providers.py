"""Offline provider tests. HTTP is mocked; no credentials or network calls."""

import asyncio
import json
from collections.abc import Callable

import httpx
import pytest

from app.api.schemas import Provider
from app.config import Settings
from app.providers.anthropic import ANTHROPIC_VERSION, AnthropicProvider
from app.providers.base import LLMProvider
from app.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUpstreamError,
)
from app.providers.ollama import OllamaProvider
from app.providers.openai import OpenAIProvider

OPENAI_KEY = "fictional-openai-key"
ANTHROPIC_KEY = "fictional-anthropic-key"
UPSTREAM_MARKER = "raw-upstream-body-marker"
ProviderFactory = Callable[[httpx.AsyncClient], LLMProvider]


def openai_body(
    text: str = "Connection successful",
    *,
    status: str = "completed",
    usage: dict[str, object] | None = None,
    include_usage: bool = True,
    output: list[object] | None = None,
    model: str | None = "gpt-5-nano-2025-08-07",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": "resp_test",
        "object": "response",
        "status": status,
        "output": output
        if output is not None
        else [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }
    if model is not None:
        payload["model"] = model
    if usage is not None:
        payload["usage"] = usage
    elif include_usage and status == "completed":
        payload["usage"] = {
            "input_tokens": 12,
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 3},
            "total_tokens": 16,
        }
    return payload


def anthropic_body(
    text: str = "Connection successful",
    *,
    stop_reason: str = "end_turn",
    content: list[object] | None = None,
    usage: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "stop_reason": stop_reason,
        "content": content
        if content is not None
        else [
            {"type": "thinking", "thinking": "hidden reasoning", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "usage": usage
        if usage is not None
        else {
            "input_tokens": 25,
            "output_tokens": 15,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 7,
        },
    }


def ollama_body(
    text: str = "Connection successful",
    *,
    done: bool = True,
    done_reason: str | None = "stop",
    include_usage: bool = True,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": text},
        "done": done,
    }
    if done_reason is not None:
        payload["done_reason"] = done_reason
    if include_usage:
        payload["prompt_eval_count"] = 26
        payload["eval_count"] = 298
    return payload


def scripted(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[list[httpx.Request], httpx.MockTransport]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    return seen, httpx.MockTransport(record)


def json_response(status: int, body: object) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return handler


def error_response(status: int) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={"error": {"message": UPSTREAM_MARKER, "type": "provider_error"}},
        )

    return handler


def body_of(request: httpx.Request) -> dict[str, object]:
    parsed = json.loads(request.content.decode())
    assert isinstance(parsed, dict)
    return parsed


def build(name: str, client: httpx.AsyncClient) -> LLMProvider:
    if name == "openai":
        return OpenAIProvider(OPENAI_KEY, client=client, timeout_seconds=5)
    if name == "anthropic":
        return AnthropicProvider(ANTHROPIC_KEY, client=client, timeout_seconds=5)
    if name == "ollama":
        return OllamaProvider(
            base_url="http://localhost:11434",
            client=client,
            timeout_seconds=5,
        )
    raise AssertionError(name)


def success_body(name: str) -> dict[str, object]:
    if name == "openai":
        return openai_body()
    if name == "anthropic":
        return anthropic_body()
    return ollama_body()


def assert_secret_free(message: str) -> None:
    assert UPSTREAM_MARKER not in message
    assert OPENAI_KEY not in message
    assert ANTHROPIC_KEY not in message
    assert "Bearer" not in message
    assert "Authorization" not in message


def test_openai_normalizes_text_usage_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((5.0, 5.25))
    monkeypatch.setattr("app.providers.client.monotonic_now", lambda: next(ticks))

    async def scenario() -> None:
        seen, transport = scripted(json_response(200, openai_body()))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OpenAIProvider(OPENAI_KEY, client=client)
            response = await provider.generate("Hello")
            assert "fictional-openai-key" not in repr(provider)

        request = seen[0]
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == f"Bearer {OPENAI_KEY}"
        assert OPENAI_KEY not in request.content.decode()
        sent = body_of(request)
        assert sent == {
            "model": "gpt-5-nano",
            "input": "Hello",
            "max_output_tokens": 2048,
            "reasoning": {"effort": "minimal"},
        }
        assert response.provider is Provider.OPENAI
        assert response.model == "gpt-5-nano-2025-08-07"
        assert response.content == "Connection successful"
        assert response.input_tokens == 12
        assert response.output_tokens == 4
        assert response.reasoning_tokens == 3
        assert response.cached_input_tokens is None
        assert response.latency_ms == 250
        assert response.estimated_cost is None

    asyncio.run(scenario())


def test_openai_maps_system_prompt_and_reuses_client() -> None:
    async def scenario() -> None:
        seen, transport = scripted(json_response(200, openai_body()))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OpenAIProvider.from_settings(
                Settings(
                    _env_file=None,
                    openai_api_key=OPENAI_KEY,
                    openai_model="gpt-5-nano",
                    openai_max_output_tokens=128,
                ),
                client=client,
            )
            await provider.generate("Hello")
            await provider.generate("Hello", system_prompt="Be brief")
            await provider.aclose()
            assert client.is_closed is False

        assert len(seen) == 2
        assert body_of(seen[0])["model"] == "gpt-5-nano"
        assert body_of(seen[0])["max_output_tokens"] == 128
        assert "instructions" not in body_of(seen[0])
        assert body_of(seen[1])["instructions"] == "Be brief"
        assert body_of(seen[1])["reasoning"] == {"effort": "minimal"}

    asyncio.run(scenario())


def test_openai_reads_message_text_after_reasoning_item() -> None:
    output = [
        {
            "type": "reasoning",
            "id": "rs_test",
            "summary": [{"type": "summary_text", "text": "not the answer"}],
            "content": [{"type": "output_text", "text": "not the answer either"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Connection "},
                {"type": "output_text", "text": "successful"},
            ],
        },
    ]

    async def scenario() -> None:
        _seen, transport = scripted(json_response(200, openai_body(output=output)))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await OpenAIProvider(OPENAI_KEY, client=client).generate("Hello")
        assert response.content == "Connection successful"

    asyncio.run(scenario())


def test_openai_refusal_text_is_content() -> None:
    output = [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "refusal", "refusal": "I can't help with that."}],
        }
    ]

    async def scenario() -> None:
        _seen, transport = scripted(json_response(200, openai_body(output=output)))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await OpenAIProvider(OPENAI_KEY, client=client).generate("Hello")
        assert response.content == "I can't help with that."
        assert response.provider is Provider.OPENAI

    asyncio.run(scenario())


def test_openai_keeps_reported_zero_usage() -> None:
    async def scenario() -> None:
        payload = openai_body(usage={"input_tokens": 0, "output_tokens": 0})
        _seen, transport = scripted(json_response(200, payload))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await OpenAIProvider(OPENAI_KEY, client=client).generate("Hello")
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    asyncio.run(scenario())


def test_openai_uses_configured_model_when_response_omits_it() -> None:
    async def scenario() -> None:
        payload = openai_body(model=None)
        _seen, transport = scripted(json_response(200, payload))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await OpenAIProvider(OPENAI_KEY, client=client).generate("Hello")
        assert response.model == "gpt-5-nano"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (openai_body(status="incomplete", usage={"input_tokens": 10, "output_tokens": 16}), "incomplete"),
        (
            {
                **openai_body(status="incomplete"),
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            "max_output_tokens",
        ),
        (
            {
                "status": "failed",
                "error": {"code": "server_error", "message": UPSTREAM_MARKER},
            },
            "server_error",
        ),
        (openai_body(text="   "), "did not include text"),
        (openai_body(include_usage=False), "did not include input_tokens"),
        (openai_body(usage={"input_tokens": 4}), "did not include output_tokens"),
        (openai_body(usage={"input_tokens": True, "output_tokens": 1}), "invalid input_tokens"),
        ({"status": "completed", "output": "nope"}, "did not include text"),
    ],
)
def test_openai_rejects_incomplete_or_unusable_responses(
    payload: dict[str, object],
    match: str,
) -> None:
    async def scenario() -> None:
        _seen, transport = scripted(json_response(200, payload))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ProviderResponseError, match=match) as exc_info:
                await OpenAIProvider(OPENAI_KEY, client=client).generate("Hello")
        assert_secret_free(str(exc_info.value))

    asyncio.run(scenario())


def test_anthropic_normalizes_text_usage_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((1.0, 1.5, 2.0, 2.25))
    monkeypatch.setattr("app.providers.client.monotonic_now", lambda: next(ticks))

    async def scenario() -> None:
        seen, transport = scripted(json_response(200, anthropic_body()))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = AnthropicProvider(ANTHROPIC_KEY, client=client)
            first = await provider.generate("Hello")
            await provider.generate("Hello", system_prompt="Be brief")

        assert str(seen[0].url) == "https://api.anthropic.com/v1/messages"
        assert seen[0].headers["x-api-key"] == ANTHROPIC_KEY
        assert seen[0].headers["anthropic-version"] == ANTHROPIC_VERSION
        assert ANTHROPIC_KEY not in seen[0].content.decode()
        assert body_of(seen[0]) == {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }
        assert body_of(seen[1])["system"] == "Be brief"
        assert "reasoning" not in body_of(seen[1])
        assert first.provider is Provider.ANTHROPIC
        assert first.model == "claude-sonnet-4-6"
        assert first.content == "Connection successful"
        assert "hidden reasoning" not in first.content
        assert first.input_tokens == 25
        assert first.output_tokens == 15
        assert first.cache_write_input_tokens == 5
        assert first.cache_read_input_tokens == 7
        assert first.latency_ms == 500
        assert first.estimated_cost is None

    asyncio.run(scenario())


def test_anthropic_refusal_text_is_not_a_transport_failure() -> None:
    async def scenario() -> None:
        payload = anthropic_body(
            stop_reason="refusal",
            content=[{"type": "text", "text": "I can't help with that."}],
        )
        _seen, transport = scripted(json_response(200, payload))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await AnthropicProvider(ANTHROPIC_KEY, client=client).generate(
                "Hello"
            )
        assert response.content == "I can't help with that."

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            anthropic_body(stop_reason="refusal", content=[]),
            "did not include text",
        ),
        (anthropic_body(stop_reason="max_tokens"), "incomplete"),
        (anthropic_body(stop_reason="tool_use"), "not completed"),
        (anthropic_body(text="  "), "did not include text"),
        (anthropic_body(usage={}), "did not include input_tokens"),
        (
            anthropic_body(usage={"input_tokens": 3, "output_tokens": None}),
            "did not include output_tokens",
        ),
    ],
)
def test_anthropic_rejects_incomplete_or_unusable_responses(
    payload: dict[str, object],
    match: str,
) -> None:
    async def scenario() -> None:
        _seen, transport = scripted(json_response(200, payload))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ProviderResponseError, match=match) as exc_info:
                await AnthropicProvider(ANTHROPIC_KEY, client=client).generate("Hello")
        assert_secret_free(str(exc_info.value))

    asyncio.run(scenario())


def test_ollama_normalizes_text_usage_and_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((2.0, 2.125))
    monkeypatch.setattr("app.providers.client.monotonic_now", lambda: next(ticks))

    async def scenario() -> None:
        seen, transport = scripted(json_response(200, ollama_body()))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(base_url="http://localhost:11434/", client=client)
            response = await provider.generate("Hello", system_prompt="Be brief")

        assert str(seen[0].url) == "http://localhost:11434/api/chat"
        assert "authorization" not in seen[0].headers
        assert body_of(seen[0]) == {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
        }
        assert response.provider is Provider.OLLAMA
        assert response.model == "llama3.2"
        assert response.content == "Connection successful"
        assert response.input_tokens == 26
        assert response.output_tokens == 298
        assert response.latency_ms == 125
        assert response.estimated_cost is None

    asyncio.run(scenario())


def test_ollama_sends_bearer_token_when_configured() -> None:
    cloud_key = "placeholder-ollama-key"

    async def scenario() -> None:
        seen, transport = scripted(json_response(200, ollama_body()))
        async with httpx.AsyncClient(transport=transport) as client:
            provider = OllamaProvider(
                base_url="https://ollama.com",
                api_key=cloud_key,
                client=client,
            )
            await provider.generate("Hello")
            assert cloud_key not in repr(provider)

        assert str(seen[0].url) == "https://ollama.com/api/chat"
        assert seen[0].headers["authorization"] == f"Bearer {cloud_key}"
        assert cloud_key not in seen[0].content.decode()

    asyncio.run(scenario())


def test_ollama_omits_system_message_without_system_prompt() -> None:
    async def scenario() -> None:
        seen, transport = scripted(json_response(200, ollama_body(done_reason=None)))
        async with httpx.AsyncClient(transport=transport) as client:
            response = await OllamaProvider(
                base_url="http://localhost:11434",
                client=client,
            ).generate("Hello")
        assert body_of(seen[0])["messages"] == [{"role": "user", "content": "Hello"}]
        assert response.content == "Connection successful"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (ollama_body(done=False), "incomplete"),
        (ollama_body(done_reason="length"), "length"),
        (ollama_body(text=""), "did not include text"),
        (ollama_body(include_usage=False), "did not include prompt_eval_count"),
        (
            ollama_body() | {"eval_count": None},
            "did not include eval_count",
        ),
    ],
)
def test_ollama_rejects_incomplete_or_unusable_responses(
    payload: dict[str, object],
    match: str,
) -> None:
    async def scenario() -> None:
        _seen, transport = scripted(json_response(200, payload))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(ProviderResponseError, match=match) as exc_info:
                await OllamaProvider(
                    base_url="http://localhost:11434",
                    client=client,
                ).generate("Hello")
        assert_secret_free(str(exc_info.value))

    asyncio.run(scenario())


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "ollama"])
def test_malformed_payloads_are_rejected(provider_name: str) -> None:
    async def scenario() -> None:
        cases: list[Callable[[httpx.Request], httpx.Response]] = [
            lambda _request: httpx.Response(200, content=b"not-json"),
            json_response(200, ["not", "an", "object"]),
        ]
        for handler in cases:
            _seen, transport = scripted(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(ProviderResponseError) as exc_info:
                    await build(provider_name, client).generate("Hello")
            assert_secret_free(str(exc_info.value))

    asyncio.run(scenario())


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "ollama"])
@pytest.mark.parametrize(
    ("status", "error_type", "match"),
    [
        (401, ProviderAuthenticationError, "HTTP 401"),
        (403, ProviderAuthenticationError, "HTTP 403"),
        (429, ProviderRateLimitError, "HTTP 429"),
        (500, ProviderUpstreamError, "HTTP 500"),
        (503, ProviderUpstreamError, "HTTP 503"),
    ],
)
def test_http_errors_are_typed_and_do_not_retry(
    provider_name: str,
    status: int,
    error_type: type[ProviderError],
    match: str,
) -> None:
    async def scenario() -> None:
        seen, transport = scripted(error_response(status))
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(error_type, match=match) as exc_info:
                await build(provider_name, client).generate("Hello")
        assert len(seen) == 1
        assert_secret_free(str(exc_info.value))

    asyncio.run(scenario())


@pytest.mark.parametrize("provider_name", ["openai", "anthropic", "ollama"])
def test_timeouts_and_connection_failures(provider_name: str) -> None:
    async def scenario() -> None:
        failures: list[tuple[Exception, str]] = [
            (httpx.ReadTimeout("socket details"), "request timed out"),
            (httpx.ConnectError("connection refused to host"), "connection failed"),
        ]
        for failure, match in failures:
            def handler(_request: httpx.Request) -> httpx.Response:
                raise failure

            _seen, transport = scripted(handler)
            async with httpx.AsyncClient(transport=transport) as client:
                with pytest.raises(ProviderTimeoutError, match=match) as exc_info:
                    await build(provider_name, client).generate("Hello")
            message = str(exc_info.value)
            assert "socket details" not in message
            assert "connection refused" not in message
            assert_secret_free(message)

    asyncio.run(scenario())


def test_missing_credentials_raise_only_when_building_a_provider() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        OpenAIProvider.from_settings(settings)
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        OpenAIProvider("   ")
    with pytest.raises(ProviderConfigurationError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider.from_settings(settings)
    with pytest.raises(ProviderConfigurationError, match="OLLAMA_BASE_URL"):
        OllamaProvider.from_settings(Settings(_env_file=None, ollama_base_url="   "))
    with pytest.raises(ProviderConfigurationError, match="OLLAMA_BASE_URL"):
        OllamaProvider(base_url="localhost:11434")
    with pytest.raises(ProviderConfigurationError, match="OLLAMA_MODEL"):
        OllamaProvider(base_url="http://localhost:11434", model=" ")


def test_owned_client_closes_and_can_be_used_as_a_context_manager() -> None:
    async def scenario() -> None:
        async with OpenAIProvider(OPENAI_KEY, timeout_seconds=1) as provider:
            assert provider._http._client.is_closed is False
        assert provider._http._client.is_closed is True
        await provider.aclose()
        with pytest.raises(ProviderError, match="client is closed"):
            await provider.generate("Hello")

    asyncio.run(scenario())
