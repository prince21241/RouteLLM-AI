"""Async engine and session factory.

The application does not create tables here. Run Alembic migrations first.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.db.store import RequestStore


def create_db_engine(url: str, *, null_pool: bool = False) -> AsyncEngine:
    """Create an async PostgreSQL engine. The URL is not logged."""
    options: dict[str, object] = {"pool_pre_ping": True}
    if null_pool:
        options["poolclass"] = NullPool
    return create_async_engine(url, **options)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory that does not expire objects on commit."""
    return async_sessionmaker(engine, expire_on_commit=False)


def build_store(url: str, *, null_pool: bool = False) -> tuple[RequestStore, AsyncEngine]:
    """Build a store and the engine the caller must dispose."""
    engine = create_db_engine(url, null_pool=null_pool)
    return RequestStore(create_session_factory(engine)), engine
