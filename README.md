# RouteLLM AI

Phase 1 is the FastAPI backend: configuration, shared models, the provider interface, and an in-memory model catalog.

Phase 2 adds non-streaming text clients for OpenAI, Anthropic, and Ollama. Each client implements `LLMProvider` and returns `LLMResponse`. There is no automatic retry.

Phase 3 adds a rule-based complexity heuristic, deterministic model selection, and `POST /api/v1/chat`.

Phase 4 adds a pricing service, PostgreSQL persistence, request history, and a same-token-volume savings estimate. Quality evaluation, escalation, runtime provider fallback, dashboard endpoints, the frontend, and ML are later phases.

The API still starts and serves `/health` when provider credentials and `DATABASE_URL` are missing. A provider reports missing configuration only when that provider is used. If generation fails, the request stops. The router does not call another provider. Phase 4 still makes one provider call per request. `/health` does not touch the database. `/ready` does.

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

Provider clients still leave `estimated_cost` empty. Token counts are taken from the provider response. A missing count is an error, not a zero. Optional cache counts are kept when the provider reports them so the pricing service can price those categories without adding them twice.

## Routing

The classifier scores the user prompt and the optional system prompt together, then sends those strings to the provider in their original roles. The score is a keyword and shape heuristic from 0.0 to 1.0. It does not measure true difficulty or answer quality. Patterns are English-oriented, use character length rather than tokens, and can miss paraphrases or flag incidental words.

Configured thresholds use exact boundaries:

- score `<= LOW_COMPLEXITY_THRESHOLD` (default 0.30): `low`
- score `<= HIGH_COMPLEXITY_THRESHOLD` (default 0.70): `medium`
- otherwise: `high`

The router then picks one enabled model:

- Prefer a model in the requested tier.
- Otherwise prefer the nearest higher enabled tier. That selection is not degraded.
- Otherwise select the highest available lower tier and set `degraded` to true. A lower-tier model is not described as matching the requested tier.
- Inside a tier, `ROUTING_PREFERENCE` names model ids in order. Models not listed keep registration order: the OpenAI model, then the Anthropic model, then the Ollama model.
- If nothing is enabled, the chat endpoint returns 503.

Selection does not claim to be the cheapest or the highest quality. It does not contact a provider to see if it is healthy. Quality tiers on models are policy labels (`OPENAI_QUALITY_TIER`, `ANTHROPIC_QUALITY_TIER`, `OLLAMA_QUALITY_TIER`), not benchmark results.

A model is eligible only when its routing flag is on and its required configuration is present. For the documented local setup, `OPENAI_ROUTING_ENABLED` defaults to true, but `gpt-5-nano` stays out of routing until `OPENAI_API_KEY` is set. Anthropic and Ollama stay disabled until their routing flags are turned on. Context windows and list prices in the catalog are verified metadata for `gpt-5-nano`, `claude-sonnet-4-6`, and `llama3.2`. Changing a model id to something else fails startup instead of inventing metadata. Chat responses leave `quality_score` null and `escalated` false. `metrics.cost` is filled from the price book when the selected model id is in that book, and stays null when it is not.

`MAX_INPUT_CHARACTERS` (default 8000) limits the combined user prompt and system prompt. It is an application character limit, not a model context-window check. `MIN_QUALITY_SCORE` and `MAX_MODEL_ATTEMPTS` are unused until later phases.

## Pricing

Cost is calculated outside the provider clients:

```text
input_cost = input_tokens / 1_000_000 * input_price
output_cost = output_tokens / 1_000_000 * output_price
total_cost = input_cost + output_cost
```

Rates come from the verified catalog, checked on 2026-09-28.

| Model | Input | Cached or cache read | Cache write | Output | Source |
| --- | --- | --- | --- | --- | --- |
| `gpt-5-nano` | $0.05 | cached input $0.005 | no separate rate | $0.40 | [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5-nano) |
| `claude-sonnet-4-6` | $3 | cache read $0.30 | $3.75 for 5 minutes, $6 for 1 hour | $15 | [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) |
| `llama3.2` | $0 | n/a | n/a | $0 | [Ollama library](https://ollama.com/library/llama3.2) |

Prices are USD per 1,000,000 tokens. OpenAI cached tokens are a subset of `input_tokens`, so they are priced at the cached rate and the remainder at the standard input rate. They are not billed twice. OpenAI cache writes have no separate rate and stay in the uncached input bucket. `output_tokens` already includes reasoning tokens, so those are not added again. Anthropic `input_tokens` excludes cache writes and cache reads, so those categories are priced on their own. If only the aggregate cache-write count is present, it is priced at the 5-minute write rate and the result is marked `estimated`.

A missing price is not zero. An unknown model id does not inherit another model's rates, and `metrics.cost` stays null. `llama3.2` has a real provider API price of zero. That zero excludes hardware and electricity.

When the provider does not report cache counts, the service prices the reported input and output tokens at standard rates and sets `cost_completeness` to `estimated`. A full cache breakdown that the price book can price is `complete`.

Money in JSON is a string, not a number. Values are quantized to 12 decimal places, half up, then trailing zeros are removed. Twelve input tokens at $0.05 per million is `"0.0000006"`.

`PREMIUM_BASELINE_MODEL` defaults to `claude-sonnet-4-6`. It needs pricing metadata only. The API does not call that model. The baseline applies the premium model's standard input and output rates to the successful response's observed token volume. For Anthropic responses, reported cache-write and cache-read tokens are added to that volume because they are not inside `input_tokens`. For OpenAI they are already included, so they are not added again. Savings are `baseline_cost - total_cost`. They may be negative. Either side being unavailable makes `estimated_savings` null. There is no ratio, so a zero baseline does not divide. `savings_basis` is `same_token_volume`: another model may tokenize differently, write a different amount of text, and this comparison does not establish equal quality.

## Persistence

PostgreSQL stores the prompt, optional system prompt, routing decision, response, token usage, cost, and a pricing snapshot on each attempt. The list endpoint returns summaries only. A later price change does not rewrite a stored snapshot.

The service inserts a `pending` row and commits it before the provider call. If that write fails, the API returns 503 and does not call the provider. The provider call is not wrapped in a database transaction. After the call, a second short transaction stores success or failure. Reported token counts and a known cost are kept when a failed provider response still included them. An unknown cost stays null.

If the provider succeeds and the final write fails, the API still returns the generated text, the request id, `persistence_status` of `failed`, and `persistence_warning`. It does not call the provider again. The log line names the request id and the exception type. It does not include the prompt, the response, or the database URL.

`metrics.latency_ms` is the provider-call latency. `end_to_end_latency_ms` is the service time around routing, the provider call, and the final write.

An interrupted process can leave a row `pending`. Version 1 does not sweep those rows. Startup does not create tables. Apply the Alembic migration first. Without `DATABASE_URL`, chat still runs and sets `persistence_status` to `not_configured`. History and `/ready` then report that the database is unavailable. `/health` stays a liveness check.

Prompts and responses are stored in the local database. This API has no application authentication. Keep it on `127.0.0.1`.

## Run PostgreSQL and the API

Development should bind to loopback. Do not expose the API on a public interface.

From the repository root, start the database. The Compose file uses a named volume, `routellm_postgres`, and development defaults that you can override with `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, and `POSTGRES_PORT`. It does not delete an existing volume.

```powershell
docker compose up -d postgres
```

If port 5432 is already in use, pick another host port:

```powershell
$env:POSTGRES_PORT = "5433"
docker compose up -d --force-recreate postgres
```

Copy `.env.example` to `.env` if you have not already, and set `DATABASE_URL` to the host address. The backend runs on the host, so the host is `127.0.0.1`, not the Compose service name. The example URL matches the Compose defaults:

```text
postgresql+asyncpg://routellm:routellm@127.0.0.1:5432/routellm
```

From `backend`, with the virtual environment activated, apply migrations and start the API:

```powershell
Set-Location backend
.\.venv\Scripts\Activate.ps1
python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Readiness check: [http://127.0.0.1:8000/ready](http://127.0.0.1:8000/ready)

`/health` returns `{"status":"ok"}` without connecting to PostgreSQL or a provider. `/ready` returns 200 only when the database accepts a query. A failure uses a fixed message and does not include the connection string.

## Call chat and history from Insomnia

Insomnia calls the local backend. Do not add an OpenAI `Authorization` header. The backend loads its provider key from the repository-root `.env`.

`POST http://127.0.0.1:8000/api/v1/chat`

Header: `Content-Type: application/json`

```json
{
  "prompt": "Explain what an API is in two sentences"
}
```

An optional system prompt is separate from the user prompt:

```json
{
  "prompt": "Explain what an API is in two sentences",
  "system_prompt": "Use plain language."
}
```

With only `gpt-5-nano` enabled, a short prompt selects that low-tier model. A higher complexity score still uses it when no higher tier is enabled, and the response sets `degraded` to true. `metrics.latency_ms` is the provider-call latency. `end_to_end_latency_ms` is separate. `metrics.cost` is a decimal string when `gpt-5-nano` is in the price book. `quality_score` is null and `escalated` is false. `baseline_model`, `baseline_cost`, `estimated_savings`, and `savings_basis` describe the premium estimate. `persistence_status` is `stored` after a successful write.

History does not require a body:

`GET http://127.0.0.1:8000/api/v1/requests?limit=20&offset=0`

`GET http://127.0.0.1:8000/api/v1/requests/{request_id}`

Use the `request_id` from the chat response. A missing id returns 404. The list is newest first and does not include the prompt or the response text. `limit` must be from 1 to 100 and `offset` must be 0 or greater.

## Run the tests

From `backend`, with the virtual environment activated.

Offline tests, including pricing and the in-memory lifecycle. They do not need PostgreSQL or provider credentials:

```powershell
python -m pytest
```

PostgreSQL integration tests. They drop and recreate only a database whose name ends with `_test`. The default name is `routellm_test` on `127.0.0.1:5432`. They do not migrate the `routellm` development database. Point `TEST_DATABASE_URL` at another disposable database if the host port is not 5432:

```powershell
$env:TEST_DATABASE_URL = "postgresql+asyncpg://routellm:routellm@127.0.0.1:5432/routellm_test"
python -m pytest integration_tests
```

The offline suite uses FastAPI's `TestClient` and `httpx.MockTransport`. Fictional keys in the provider tests never leave the process. Do not point `TEST_DATABASE_URL` at a database you want to keep.

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
    main.py            create_app factory, /health, /ready, chat and history
    config.py          pydantic-settings configuration
    api/               chat and history routes
    chat/              classify, select, price, and call one provider
    db/                SQLAlchemy models and the request store
    pricing/           Decimal cost, baseline, and savings
    providers/         LLMProvider and OpenAI, Anthropic, Ollama clients
    routing/           catalog, complexity heuristic, and model router
  alembic/             PostgreSQL migrations
  scripts/
    verify_openai.py   one manual OpenAI request
  tests/               offline pytest suite
  integration_tests/   disposable PostgreSQL tests
.env.example           placeholder environment variables
docker-compose.yml     local PostgreSQL only
```

Pass a `Settings` instance, `ModelRegistry`, classifier, router, provider factory, or request store to `create_app` when a test needs to override them. The default catalog registers the documented models and enables `gpt-5-nano` only when its routing flag is on and `OPENAI_API_KEY` is set.
