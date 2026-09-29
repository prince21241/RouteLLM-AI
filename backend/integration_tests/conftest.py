"""PostgreSQL fixtures for integration tests.

These tests migrate only a database whose name ends with ``_test``.
They do not run as part of the default offline suite.
"""

import asyncio
import os
import re

import pytest
from alembic.config import Config
from sqlalchemy import text

from alembic import command
from app.db.session import build_store

DEFAULT_TEST_URL = "postgresql+asyncpg://routellm:routellm@127.0.0.1:5432/routellm_test"
_NAME = re.compile(r"^[a-z][a-z0-9_]*_test$")


def _database_name(url: str) -> str:
    name = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    if not _NAME.fullmatch(name):
        raise RuntimeError("Refusing to migrate a database whose name does not end with _test")
    return name


def _admin_dsn(url: str) -> str:
    _database_name(url)
    prefix = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[0]
    return (prefix + "/postgres").replace("postgresql+asyncpg://", "postgresql://", 1)


async def _recreate(url: str) -> None:
    import asyncpg

    name = _database_name(url)
    connection = await asyncpg.connect(_admin_dsn(url))
    try:
        await connection.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        await connection.execute(f'CREATE DATABASE "{name}"')
    finally:
        await connection.close()


def _upgrade(url: str) -> None:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", url)
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_URL)
    _database_name(url)
    try:
        asyncio.run(_recreate(url))
    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL test database setup failed "
            f"({type(exc).__name__}). Start it with: docker compose up -d postgres"
        ) from None
    _upgrade(url)
    return url


@pytest.fixture
def store(database_url: str):
    request_store, engine = build_store(database_url, null_pool=True)

    async def clean() -> None:
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM evaluations"))
            await connection.execute(text("DELETE FROM attempts"))
            await connection.execute(text("DELETE FROM requests"))

    asyncio.run(clean())
    yield request_store, engine
    asyncio.run(engine.dispose())
