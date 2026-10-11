"""Database engine and session handling."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        # FastAPI runs sync endpoints in a thread pool, so the connection
        # must be usable from more than one thread.
        return create_engine(url, connect_args={"check_same_thread": False})
    # Serverless Postgres closes idle connections; check before each use.
    return create_engine(url, pool_pre_ping=True)


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session(request: Request) -> Iterator[Session]:
    """FastAPI dependency: one session per request, from the app's own factory."""
    with request.app.state.session_factory() as session:
        yield session
