"""FastAPI application: ingestion, storage, metrics and insights for health data."""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.analytics.frames import records_to_frame
from app.api.errors import register_error_handlers
from app.api.routes import router
from app.config import get_settings
from app.db import SessionLocal
from app.services import datasets as store
from app.services.insights import build_prediction

DESCRIPTION = """
Unifies sleep, activity, nutrition, mood, heart-rate, training and study data and
turns it into explainable insights: correlations, anomalies, rolling load,
recovery trend, a burnout-risk score and a next-day readiness prediction.

The `demo` dataset is synthetic and read-only. **Nothing here is medical advice.**
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    with app.state.session_factory() as session:
        store.purge_expired_uploads(session, settings.upload_ttl_days)
        if settings.seed_demo_data:
            demo = records_to_frame(store.seed_demo(session).metrics)
            # Train the demo model in the background so the first visitor does not
            # wait for it and startup is not delayed.
            threading.Thread(target=build_prediction, args=(demo,), daemon=True).start()
    yield


def create_app(session_factory: sessionmaker[Session] = SessionLocal) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Health & Performance Aggregator API",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.session_factory = session_factory
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handlers(app)
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": app.title, "docs": "/docs", "health": "/health"}

    @app.get("/health", tags=["system"])
    def health(request: Request) -> dict[str, str]:
        """Liveness check that also confirms the database is reachable."""
        try:
            with request.app.state.session_factory() as session:
                session.execute(text("SELECT 1"))
        except SQLAlchemyError as exc:
            raise HTTPException(503, "The database is not reachable.") from exc
        return {"status": "ok"}

    return app


app = create_app()
