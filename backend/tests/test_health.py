"""Health endpoint tests. These do not call providers or a database."""

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_home_page_is_served(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="chat-form"' in response.text


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_starts_without_credentials_or_services(settings: Settings) -> None:
    app = create_app(settings=settings)

    assert app.state.settings.openai_api_key is None
    assert app.state.settings.anthropic_api_key is None
    assert app.state.registry.list_enabled() == []

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
