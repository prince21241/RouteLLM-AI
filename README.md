# RouteLLM AI

A FastAPI service that routes each prompt to an appropriate LLM based on estimated complexity, records every request in PostgreSQL, and tracks cost and estimated savings against a premium baseline model. It includes a browser chat client and a local observability dashboard.

Local-use only. There is no application authentication. Stored prompts and answers are visible to anyone who can reach the API. Keep it bound to 127.0.0.1.

<!-- Add a screenshot or short GIF of the chat page and dashboard here. -->
Features
Multi-provider clients for OpenAI (Responses API), Anthropic (Messages API), and Ollama (local or Ollama Cloud). Each client makes a single non-streaming attempt with no automatic retry.
Complexity-based routing using a rule-based keyword and shape heuristic, with deterministic model selection and configurable tier thresholds.
Cost tracking with a Decimal-based pricing service, per-attempt pricing snapshots, and a same-token-volume savings estimate against a premium baseline.
Request history in PostgreSQL, written before the provider call so a failed write never triggers a model call.
Optional quality checks and escalation that can call one stronger model after an explicit failed check.
Optional provider fallback that can call one model on another provider after a timeout, connection failure, rate limit, or transient server error.
Evaluation harness with a versioned dataset, offline (--mock) and paid (--execute) runners, and exact, numeric, JSON, and rubric grading.
Experimental ML router (TF-IDF + logistic regression) with a data-collection pipeline. Disabled by default.
Chat page and dashboard served as static files by the API, with no frontend build step.

## Screenshots

These are the local pages, from one machine’s stored chat history. They show the interface. They are not a benchmark, and a reload of Chat clears the on-screen thread.

![Chat, with example prompts that only fill the composer](docs/screenshots/chat.png)

![A stored answer, with code shown in labeled blocks](docs/screenshots/chat-answer.png)

![Overview of stored requests, recorded cost, error rate, and latency](docs/screenshots/overview.png)

![Request history, newest first](docs/screenshots/requests.png)

![Recorded cost and daily totals](docs/screenshots/costs.png)

![Enabled models and the attempt breakdown](docs/screenshots/providers.png)

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
| `OPENAI_MAX_OUTPUT_TOKENS` | `2048` | Upper bound for visible output and reasoning tokens. |
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

Uses `POST {OLLAMA_BASE_URL}/api/chat` with `stream` set to `false`. A local server receives no credential. When `OLLAMA_API_KEY` is set, the client sends it as a bearer token. Ollama Cloud uses base URL `https://ollama.com` and a cloud model name such as `gemma4:31b`. That cloud model has no published per-token price, so its recorded cost stays unknown.

| Variable | Default | Role |
| --- | --- | --- |
| `OLLAMA_API_KEY` | unset | Bearer credential for Ollama Cloud. Omit it for a local server. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local server, or `https://ollama.com` for Ollama Cloud. Startup does not connect to it. |
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

`MAX_INPUT_CHARACTERS` (default 8000) limits the combined user prompt and system prompt. It is an application character limit, not a model context-window check. `MIN_QUALITY_SCORE` is the default quality threshold for optional checks and for ML labels. `MAX_MODEL_ATTEMPTS` bounds generation attempts.

## Optional ML router

Rule-based routing stays the default (`ROUTING_STRATEGY=rule_based`). The ML path is an optional scikit-learn pipeline: TF-IDF features of the user prompt and logistic regression. Predicted probabilities are confidence estimates for the chosen class. They are not proof of answer quality.

The 30-case evaluation file is a smoke dataset. It is not training data, and this repository does not contain a measured multi-model training set. No production artifact is shipped. Do not enable ML routing until `compare` on held-out measured data supports a configured quality tolerance and cost objective. A small result is exploratory even when the commands succeed.

Training rows come from comparable live evaluations of the same prompt. The label is the cheapest model that passed the quality threshold with complete or estimated cost. If every candidate has a definitive fail, the label is `__no_acceptable_model__`. Missing evaluations, unknown grades, provider failures, and incomplete costs are excluded and counted. They are not treated as quality failures or as wins for another model. Mock answers are rejected. Reference answers and judge metadata are not features.

Splits are by `group_id`, so duplicates and near-duplicates stay together. Train and validation choose the confidence threshold. The test split is only for the comparison. Each class needs at least three groups. Fewer than 40 labeled prompts is marked exploratory.

From `backend`, with the project virtual environment. Collection does not call a provider until `--execute`. Keep `ROUTING_STRATEGY=rule_based`.

The smoke file stays `app/evaluation/datasets/v1.json` (`2026-09-29.1`). It is not training evidence. The collection dataset is `app/evaluation/datasets/families-v1.json`: 48 families, eight in each of the six categories, two paraphrases in each family, and one shared `group_id` per family. Answers are exact, numeric, or JSON checks, except the ice-density pair, which has an explicit rubric. Character-based ceilings from `--estimate` are approximate allowances, not invoices.

```powershell
..\.venv\Scripts\python.exe -m app.ml collect --estimate --dataset app\evaluation\datasets\families-v1.json
..\.venv\Scripts\python.exe -m app.ml collect --estimate --judge --dataset app\evaluation\datasets\families-v1.json
$out = Join-Path $env:LOCALAPPDATA "RouteLLM-AI\collections\families-2026-09-29.json"
..\.venv\Scripts\python.exe -m app.ml collect --execute --dataset app\evaluation\datasets\families-v1.json --output $out --max-spend 1.56196225 --judge
..\.venv\Scripts\python.exe -m app.ml validate --dataset $out
```

`--max-spend` is USD for generation plus judge calls. With `--judge`, the approximate recommended cap printed for this file is `1.56196225` (generation `1.55852545`, judge `0.0034368`). The collector saves the training file and a sibling `.audit.json` after every recorded outcome. The audit stores judge token usage and judge cost separately; that cost is not added to a candidate's training cost. Resume refuses a different dataset, candidate list, or judge configuration. A call that would exceed the cap is not started. An incurred generation or judge cost that cannot be priced stops the run and is not treated as zero. Rubric rows stay unknown without `--judge`. `gpt-5-nano` and `claude-sonnet-4-6` are the default candidates. Models without a published token price are not called. Write the output under `%LOCALAPPDATA%\RouteLLM-AI\collections`, outside the repository.

```powershell
..\.venv\Scripts\python.exe -m app.ml validate --dataset path\to\training.json
..\.venv\Scripts\python.exe -m app.ml label --dataset path\to\training.json --output path\to\labels.json
..\.venv\Scripts\python.exe -m app.ml train --dataset path\to\training.json --output-dir ..\backend\var\ml --seed 17
..\.venv\Scripts\python.exe -m app.ml compare --dataset path\to\training.json --artifact ..\backend\var\ml\model.joblib --trusted-root ..\backend\var\ml
```

Add `--quality-tolerance 0.05 --cost-objective min_mean_cost` only when you want a promotion decision. Without both, the comparison reports metrics and does not name a winner. Recorded cost and latency are single-model outcomes. They omit escalation, fallback, judge calls, and end-to-end wall-clock time.

Apply the nullable metadata migration before serving traffic that writes the new column:

```powershell
..\.venv\Scripts\python.exe -m alembic upgrade head
```

To try ML routing, set these in `.env` and restart the API. The artifact path must stay inside the trusted directory. The API does not accept an uploaded model or a path from a request.

```text
ROUTING_STRATEGY=ml
ML_ARTIFACT_PATH=C:\Credit-Card-Fraud-Detector\RouteLLM-AI\backend\var\ml\model.joblib
ML_TRUSTED_ROOT=C:\Credit-Card-Fraud-Detector\RouteLLM-AI\backend\var\ml
```

An absent, invalid, or incompatible artifact leaves rule-based routing in effect and records the diagnostic. Return to the default with `ROUTING_STRATEGY=rule_based` and restart. Low confidence, an ineligible model, or a no-acceptable-model prediction also uses the existing rules. That fallback does not guarantee answer quality. Explicit provider limits, credentials, escalation, fallback, and attempt budgets still apply. Request details include `routing_metadata`.

Collect more real data in stages:

1. Run the evaluation dataset with `--execute` separately for each candidate model, only when you authorize the spend. Keep the JSON outside the repository.
2. Convert those runs into the training format: one prompt record, a shared `group_id` for paraphrases, live provenance, and measured evidence. Do not convert `--mock` output.
3. Add families where the cheap model fails and the stronger model passes, and families where no model passes.
4. Hold out entire families before fitting. Check coverage and class balance on the comparison, not a fixed sample count.
5. Enable `ROUTING_STRATEGY=ml` only after that held-out comparison supports the tolerance you configured.

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

`python -m alembic upgrade head` applies `20260929_0002` after the Phase 4 tables. That migration adds quality columns on `requests` and an `evaluations` table. Existing rows stay valid because the new columns are nullable. Alembic loads `DATABASE_URL` through the same settings object as the API, using the absolute repository-root `.env`. A blank or whitespace-only process value does not hide the file. A non-blank process value, or an explicit Alembic `sqlalchemy.url`, still wins. The missing-URL error names the env file path and does not include the URL.

Chat page: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

Dashboard: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)

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

## Quality evaluation and escalation

Quality checks and escalation stay off until `QUALITY_EVALUATION_ENABLED` and `ESCALATION_ENABLED` are true. Restart the API after changing `.env`. Uvicorn does not reload environment files.

| Variable | Default | Role |
| --- | --- | --- |
| `QUALITY_EVALUATION_ENABLED` | `false` | Score the initial answer. Live checks have no reference answer. |
| `QUALITY_JUDGE_ENABLED` | `false` | Ask OpenAI to judge open-ended live answers. Anthropic is not required. |
| `QUALITY_JUDGE_MODEL` | `gpt-5-nano` | Judge model sent to OpenAI. |
| `MIN_QUALITY_SCORE` | `0.75` | A passing judge score below this becomes a failure. A missing score does not. |
| `ESCALATION_ENABLED` | `false` | After an explicit `fail`, allow one more generation. |
| `ESCALATION_MODEL` | empty | Model id to call. Empty uses the next higher configured quality tier. Price is not used. |
| `EVALUATION_BASELINE_MODEL` | `gpt-5-nano` | Model the evaluation CLI compares with the router. This call is separate from the same-token-volume savings estimate. |

`ESCALATION_MODEL` must differ from the model that just answered. An id in the registry is used only when that model is enabled. An id that is not in the registry is sent to OpenAI when `OPENAI_API_KEY` is set, and its cost stays unknown. If no eligible model exists, the original answer is returned and `escalation_error` explains why. A failed second call also keeps the original answer.

Chat `metrics.cost` adds every generation attempt and any judge call. An unknown component makes the total unknown; it is not stored as zero. `estimated_savings` stays `same_token_volume`: the configured premium model's price applied to the returned answer's tokens, minus that answer's own generation cost. It is not a measured comparison against a baseline run. Negative savings remain possible.

The dataset is `backend/app/evaluation/datasets/v1.json`, version `2026-09-29.1`. Cases are `calibration` or `held_out` and cover factual, reasoning, math, coding, structured, and instruction prompts. `exact`, `numeric`, and `json_fields` use reference answers. `--mock` marks rubric cases failed with an offline placeholder judge. That failure means no model graded the rubric. It is not a quality score. Answer length, keyword overlap, and self-reported confidence are not grades. Dataset grading can use reference answers. Live chat grading cannot.

From `backend`:

```powershell
python -m pytest
python -m app.evaluation --mock --smoke
python -m app.evaluation --mock
python -m app.evaluation --execute --smoke
python -m app.evaluation --execute --output evaluation-results.json
```

`--mock` does not call a provider. `--execute` does, and it can spend money. The command prints a summary and JSON. Failures and unknowns stay in the case list. `measured_cost_difference` is the baseline run's actual cost minus the router run's actual cost, or unknown when either side is unknown. Evaluation runs are not written to chat history.

Alembic still reads `DATABASE_URL` through the application settings loader. A blank process value does not hide the repository-root `.env`.

## Provider fallback

Fallback stays off until `FALLBACK_ENABLED` is true. Restart the API after changing `.env`.

| Variable | Default | Role |
| --- | --- | --- |
| `FALLBACK_ENABLED` | `false` | Allow one recovery call after a fallback-eligible provider failure. |
| `FALLBACK_MODELS` | empty | Comma-separated documented model ids, in call order. Duplicates and unknown ids are rejected. |
| `MAX_FALLBACK_ATTEMPTS` | `1` | How many fallback generations one request may start. |
| `MAX_MODEL_ATTEMPTS` | `3` | Shared cap for the original call, a quality escalation, and fallback. |
| `REQUEST_DEADLINE_SECONDS` | `90` | Overall deadline. A new generation or judge call is not started after it. |
| `MAX_JUDGE_CALLS` | `1` | Judge calls are separate from generation attempts and still stop at the deadline. |

Each provider client still makes one attempt. The chat service decides whether another model is allowed. A candidate is used only when it is enabled, has its provider configuration, is a different provider from the attempt that just failed, and has not already been called. `POST /api/v1/chat` accepts an optional `provider`. When that field is set, routing and fallback stay on that provider, so another provider is not called.

Fallback runs for timeouts, connection failures, rate limits, and transient server errors. It does not run for invalid requests, authentication failures, permission failures, safety refusals, quality-check failures, database failures, or client cancellation. Cancelling the request does not start another attempt. A failed database write does not start another model call, and it does not replace the provider error when every generation failed.

Quality escalation and fallback share `MAX_MODEL_ATTEMPTS`. A successful earlier answer is kept when a later escalation and its fallback both fail. The response fields `fallback_used`, `fallback_reason`, `fallback_skips`, and `final_provider` describe that recovery. `routing.escalated` still means the returned answer came from a quality escalation. History attempts include `purpose` (`routing`, `fallback`, or `escalation`) and `error_category`.

`metrics.cost` includes every known generation and judge cost. Unknown usage, including a timeout that did not report tokens, stays unknown and is not stored as zero. `estimated_savings` stays `same_token_volume` and is omitted when the request total cannot be priced. `end_to_end_latency_ms` includes unsuccessful attempts. `metrics.latency_ms` remains the returned answer's provider latency.

Apply the new migration after the Phase 5 revision:

```powershell
python -m alembic upgrade head
```

That applies `20260929_0003`. Existing rows stay valid because the new columns are nullable.

From `backend`, with the virtual environment activated:

```powershell
python -m pytest
python -m alembic upgrade head
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
python -m app.demo_fallback
```

`python -m app.demo_fallback` uses in-memory providers. It does not call OpenAI, Anthropic, or Ollama. The pytest suite is mocked too. Neither one verifies a live provider fallback.

## Chat page

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The same page shell links to Overview, Requests, Costs & Usage, and Providers & Models. There is no login. Use it only on your machine.

Each send is one independent request. The page does not resend earlier messages, and the model is not given the conversation. The character limit covers the user prompt and system prompt together. It is not a model context window. `GET /api/v1/chat/options` returns that limit, the catalog models, and whether each model is enabled. It does not return credentials.

Routing stays automatic. The provider menu is filled from that options response. A provider with no enabled model cannot be selected. The API does not accept a model id on a chat request, so the page does not offer one. Quality checks, escalation, and fallback are server settings. The page shows whether they are on. It does not add switches for them. A provider limit still applies to fallback and escalation, because the chat service already restricts the registry to that provider.

Answers are rendered as readable Markdown: headings, lists, quotes, rules, hard line breaks, and fenced code stay structured. Untrusted text is inserted as text, not HTML. Earlier prompts stay on the page in an in-memory thread so navigation inside the app can show them again. That display is not conversation memory. The persistent note on the page is “Each prompt is routed independently; previous messages aren’t sent.” A reload clears the thread. The page does not store full prompts in the browser.

Each answer has Copy and collapsed routing details. A missing cost, token count, or latency stays unknown. A recorded zero stays zero. Savings stay labeled as a same-token-volume estimate. If saving the request fails, the answer remains on the page and the details say it was not saved. A stored request links to `/dashboard#request={id}`, which opens that request on the Requests page. The page does not pretend to stream tokens; a loading indicator sits beside the pending answer until the response arrives.

History stays in the chat header. Its previews come from `GET /api/v1/dashboard/requests` because `GET /api/v1/requests` does not include prompt text. Opening a row loads the stored request into the thread and does not send it. Reuse copies its prompt into the editor and does not send it. Example prompts only fill the composer. Enter sends. Shift+Enter inserts a newline. Send is disabled while a request is in progress, and a failed request leaves the editor text in place. The optional system prompt and provider restriction stay behind Options. The page cannot change server settings.

The page is the same static frontend as the dashboard. There is no frontend package and no production bundle.

Manual live check, after the API is running and a provider key is configured: open the chat page, leave routing on automatic, and send one short prompt. That call can spend provider credit. Choose a provider only when you intend to spend that account’s usage. This implementation was checked with mocked responses, not a live provider call.

From `backend`:

```powershell
..\.venv\Scripts\python.exe -m pytest tests\test_chat.py
node --test tests\chat_view.test.mjs
node --check app\static\app.js
node --check app\static\chat_view.mjs
```

There is no TypeScript project and no frontend production build.

## Dashboard

The dashboard reads stored chat requests. It does not call a provider, and it does not include offline evaluation runs. There is no login. Open it only on your machine. Do not publish port 8000 or put the history API on a public host: stored prompts and answers are visible to anyone who can reach it.

The page is static files served by the API, so there is no frontend package to install and no production bundle to build. The same origin is the API. `window.ROUTELLM_API_BASE` can point the page at another base URL, but cross-origin requests are not enabled.

No new migration is added for the dashboard. Apply the existing revisions if this database has not already been upgraded. The fallback columns from `20260929_0003` are part of the read queries.

From the repository root:

```powershell
docker compose up -d postgres
```

From `backend`, with the virtual environment that already has the project dependencies. This repository has been run with the virtual environment one level above `backend`. If you created it inside `backend` instead, use `.\.venv\Scripts\python.exe` in the commands below.

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m alembic upgrade head
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open [http://127.0.0.1:8000/overview](http://127.0.0.1:8000/overview). [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) is the same Overview page, so older links keep working. Requests are at `/requests`, cost totals at `/costs`, and provider breakdowns at `/providers`. Sending a chat from `/` can spend money. These pages only read history.

The header status comes from `GET /health` and `GET /ready`. Overview shows four primary figures: request count, recorded cost, error rate, and median end-to-end latency. Attempts, fallback rate, escalation rate, judge cost, the savings estimate, and attempt latency stay in a secondary group. Requests are newest first. The history API has no other sort. Timestamps are labeled UTC, and each row also shows the exact UTC timestamp. Providers & Models shows Enabled or Not enabled from `GET /api/v1/chat/options`. That is the server catalog, not a live provider health check. Historical attempt rows do not get availability badges. Filters are shared across the analytics pages. Overview shows the date range, and a note when a provider, model, or status filter set on another page is still applied.

`from` is inclusive and `to` is exclusive, both in UTC. A date with no time is UTC midnight. The page treats the chosen end date as inclusive and sends the following midnight as `to`. The range cannot exceed 366 days. Filters select requests by the initial routed model, provider, and status. Attempt rows are then aggregated for those requests.

Request counts are logical chat requests. Generation attempts are counted separately, so one fallback does not add a second request. Fallback rate uses rows where `fallback_used` was recorded. A missing flag is not treated as false. Escalation rate counts quality escalations, not provider fallback. Quality pass, fail, unknown, and error stay separate. A missing verdict is unknown.

Request latency is `end_to_end_latency_ms`. Attempt latency is the provider call. Missing samples are left out of the average and median. Complete and estimated costs are summed separately. Unknown amounts are counted and are not added as zero. Judge costs are the recorded judge component and are not added again on top of the request total. Savings stay labeled `same_token_volume`. Attempt costs are attributed to the provider and model that ran the attempt. A recorded zero, such as local Ollama, stays zero.

The Requests table shows an 80-character prompt preview, status, final model, cost, and latency. Initial route, attempts, quality, escalation, fallback, and the full prompt stay in the request detail drawer. Prompts and answers are rendered as text. Daily totals and judge costs are on Costs & Usage. Judge cost is not added again on top of the request total.

`?demo=1`, `?demo=empty`, and `?demo=error` are labeled samples. They do not read the database. The normal page does.

Check the dashboard view module and the API:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest
node --test tests\dashboard_view.test.mjs
node --check app\static\dashboard.js
node --check app\static\dashboard_view.mjs
```

There is no TypeScript project and no frontend build script. `node --check` only confirms the dashboard scripts parse.

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

The script sends one Responses API request through `OpenAIProvider` and does not retry. The call uses `gpt-5-nano`, minimal reasoning effort, and the configured `OPENAI_MAX_OUTPUT_TOKENS`. It prints the model, text, token counts, and latency.

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
    main.py            create_app factory, /health, /ready, chat, history, dashboard
    config.py          pydantic-settings configuration
    api/               chat, history, and dashboard routes
    chat/              classify, select, price, and call one provider
    dashboard/         filter validation and metric formatting
    static/            chat page and local dashboard
    db/                SQLAlchemy models and the request store
    pricing/           Decimal cost, baseline, and savings
    providers/         LLMProvider and OpenAI, Anthropic, Ollama clients
    routing/           catalog, complexity heuristic, and model router
    ml/                optional TF-IDF logistic regression router
    evaluation/        versioned smoke dataset and offline grading
  alembic/             PostgreSQL migrations
  scripts/
    verify_openai.py   one manual OpenAI request
  tests/               offline pytest suite
  integration_tests/   disposable PostgreSQL tests
.env.example           placeholder environment variables
docker-compose.yml     local PostgreSQL only
```

Pass a `Settings` instance, `ModelRegistry`, classifier, router, provider factory, or request store to `create_app` when a test needs to override them. The default catalog registers the documented models and enables `gpt-5-nano` only when its routing flag is on and `OPENAI_API_KEY` is set.
