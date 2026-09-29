"""Database session and engine management using SQLAlchemy and aiosqlite."""

import os
from typing import AsyncGenerator
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from backend.app.core.config import get_settings
from backend.app.db.models import Base

settings = get_settings()

# Ensure parent directory for SQLite database exists
db_url = settings.DATABASE_URL
if db_url.startswith("sqlite+aiosqlite:///"):
    db_path = db_url.replace("sqlite+aiosqlite:///", "")
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

# Create asynchronous engine
async_engine = create_async_engine(
    db_url,
    echo=False,
    future=True,
)


# Enforce foreign key constraints for SQLite connections
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key checks in SQLite."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


# Async session factory
async_session_factory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initialize database tables asynchronously."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing an async database session per request."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
