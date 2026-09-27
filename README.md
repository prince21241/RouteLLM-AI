# RouteLLM AI

Phase 1 is the FastAPI backend: configuration, shared models, the provider interface, and an in-memory model catalog.

Phase 2 adds non-streaming text clients for OpenAI, Anthropic, and Ollama. Each client implements `LLMProvider` and returns `LLMResponse`. Routing, complexity analysis, quality evaluation, escalation, pricing, database storage, and the frontend are later phases.

The API still starts and serves `/health` when provider credentials are missing. A provider reports missing configuration only when that provider is used. There is no cross-provider fallback and no automatic retry.

All commands below use `backend` as the working directory.

## Requirements

- Windows PowerShell
- Python 3.12 or newer

## Setup

From the repository root:

```powershell
Set-Location backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` installs the runtime packages plus pytest. To install runtime packages only:

```powershell
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked by the execution policy, call the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Optional local configuration lives at the repository root, one level above `backend`. The app also starts with no `.env` file and no API keys.

```powershell
Copy-Item -Path ..\.env.example -Destination ..\.env
```

Leave `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` blank until you have real keys. `.env` is gitignored.

## Provider configuration

Settings load from the process environment and the repository-root `.env`. Model ids and timeouts have defaults. API keys stay unset until you fill them in.

### OpenAI

Uses the Responses API at `https://api.openai.com/v1/responses`.

| Variable | Default | Role |
| --- | --- | --- |
| `OPENAI_API_KEY` | unset | Bearer credential. Required only when `OpenAIProvider` is constructed. |
| `OPENAI_MODEL` | `gpt-5-nano` | Model id sent on each request. |
| `OPENAI_MAX_OUTPUT_TOKENS` | `256` | Upper bound for visible output and reasoning tokens. |
| `OPENAI_TIMEOUT_SECONDS` | `30` | Finite HTTP timeout. |

The OpenAI client always sends `reasoning.effort` of `minimal`. That option is specific to this provider. A system prompt is sent as `instructions`. The user prompt is the string `input`. Text is read from message `output_text` items, including when a reasoning item comes first. Refusal text on a completed message is returned as content.

### Anthropic

Uses the Messages API at `https://api.anthropic.com/v1/messages` with `anthropic-version: 2023-06-01` and streaming disabled.

| Variable | Default | Role |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | unset | `x-api-key` credential. Required only when `AnthropicProvider` is constructed. |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Model id sent on each request. |
| `ANTHROPIC_MAX_TOKENS` | `1024` | Required `max_tokens` field. This is not the OpenAI output limit. |
| `ANTHROPIC_TIMEOUT_SECONDS` | `30` | Finite HTTP timeout. |

A system prompt is sent as the top-level `system` string. Thinking blocks are ignored. Text blocks are the completion. A `refusal` stop reason that includes text is a normal completion.

### Ollama

Uses `POST {OLLAMA_BASE_URL}/api/chat` with `stream` set to `false`. No API key is sent.

| Variable | Default | Role |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local server. Startup does not connect to it. |
| `OLLAMA_MODEL` | `llama3.2` | Model name already available on that server. |
| `OLLAMA_TIMEOUT_SECONDS` | `60` | Finite HTTP timeout. |

A system prompt is a `system` message before the user message. Input tokens come from `prompt_eval_count` and output tokens from `eval_count`.

`estimated_cost` is left empty for every provider. Token counts are taken from the provider response. A missing count is an error, not a zero.

## Run the API

From `backend`, with the virtual environment activated:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Startup uses default settings when API keys, PostgreSQL, and Ollama are absent. `/health` does not call a provider.

## Run the tests

From `backend`, with the virtual environment activated:

```powershell
python -m pytest
```

The suite uses FastAPI's `TestClient` and `httpx.MockTransport`. It runs offline and does not need provider credentials. Fictional keys in the provider tests never leave the process.

## Verify OpenAI

From `backend`, after `OPENAI_API_KEY` is set in the repository-root `.env`:

```powershell
python scripts\verify_openai.py
```

The script sends one Responses API request through `OpenAIProvider` and does not retry. The call uses `gpt-5-nano`, minimal reasoning effort, and `max_output_tokens` of 256. It prints the model, text, token counts, and latency.

Anthropic and Ollama are not called by this script.

## What was exercised

- OpenAI: offline mocks, plus one live `gpt-5-nano` request from `scripts/verify_openai.py`
- Anthropic: offline mocks only
- Ollama: offline mocks only

No Anthropic credit purchase, Ollama install, or local model download is required to run the tests.

## Layout

```text
backend/
  app/                 application package
    main.py            create_app factory and GET /health
    config.py          pydantic-settings configuration
    api/schemas.py     shared enums and models
    providers/         LLMProvider and OpenAI, Anthropic, Ollama clients
    routing/           in-memory model registry
  scripts/
    verify_openai.py   one manual OpenAI request
  tests/               offline pytest suite
.env.example           placeholder environment variables
```

Pass a `Settings` instance and a `ModelRegistry` to `create_app` when you need to override either one. The default registry is empty. Construct a provider with `from_settings` when a call needs stored credentials.
