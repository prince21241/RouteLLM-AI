# RouteLLM AI

Phase 1 is a small FastAPI backend. It loads configuration, exposes shared models, defines the provider interface, and keeps an in-memory model catalog.

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

## Run the API

From `backend`, with the virtual environment activated:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Startup uses default settings when API keys, PostgreSQL, and Ollama are absent.

## Run the tests

From `backend`, with the virtual environment activated:

```powershell
python -m pytest
```

The tests use FastAPI's `TestClient` and run offline.

## Layout

```text
backend/
  app/                 application package
    main.py            create_app factory and GET /health
    config.py          pydantic-settings configuration
    api/schemas.py     shared enums and models
    providers/base.py  LLMProvider interface
    routing/           in-memory model registry
  tests/               offline pytest suite
.env.example           placeholder environment variables
```

Pass a `Settings` instance and a `ModelRegistry` to `create_app` when you need to override either one. The default registry is empty.
