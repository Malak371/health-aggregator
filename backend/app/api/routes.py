"""HTTP API. All dataset routes live under /api/v1/datasets/{slug}."""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.analytics.frames import records_to_frame
from app.config import Settings, get_settings
from app.db import get_session
from app.ingestion.apple_health import parse_apple_health
from app.ingestion.csv_daily import parse_daily_csv
from app.ingestion.result import ParseResult
from app.models import DailyMetric, Dataset
from app.schemas import (
    DailyRecord,
    DatasetCreate,
    DatasetOut,
    DerivedDay,
    ErrorResponse,
    IngestResult,
    InsightsOut,
    PredictionOut,
    RecordsIn,
)
from app.services import datasets as store
from app.services.insights import build_insights, build_prediction, derived_series

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
Responses = dict[int | str, dict[str, Any]]

NOT_FOUND: Responses = {404: {"model": ErrorResponse, "description": "Unknown dataset"}}
UPLOAD_ERRORS: Responses = {
    **NOT_FOUND,
    403: {"model": ErrorResponse, "description": "The demo dataset is read-only"},
    413: {"model": ErrorResponse, "description": "File too large"},
    422: {"model": ErrorResponse, "description": "The file failed validation"},
}

router = APIRouter(prefix="/api/v1", responses={422: {"model": ErrorResponse}})


def dataset_from_path(slug: str, session: SessionDep) -> Dataset:
    dataset = store.get_dataset(session, slug)
    if dataset is None:
        raise HTTPException(
            404, f"Dataset '{slug}' does not exist. Datasets expire when they are not used."
        )
    store.record_activity(session, dataset)
    return dataset


def writable_dataset(dataset: Annotated[Dataset, Depends(dataset_from_path)]) -> Dataset:
    if dataset.is_demo:
        raise HTTPException(
            403, "The demo dataset is read-only. Create your own with POST /api/v1/datasets."
        )
    return dataset


def frame_for(dataset: Annotated[Dataset, Depends(dataset_from_path)]) -> pd.DataFrame:
    if not dataset.metrics:
        raise HTTPException(409, "This dataset has no data yet. Upload a file first.")
    return records_to_frame(dataset.metrics)


DatasetDep = Annotated[Dataset, Depends(dataset_from_path)]
WritableDep = Annotated[Dataset, Depends(writable_dataset)]
FrameDep = Annotated[pd.DataFrame, Depends(frame_for)]


# --- Datasets ------------------------------------------------------------------


@router.post("/datasets", status_code=201, response_model=DatasetOut, tags=["datasets"])
def create_dataset(
    session: SessionDep, settings: SettingsDep, body: DatasetCreate | None = None
) -> DatasetOut:
    """Create an empty dataset for uploads. Keep the returned `slug`: it is the only key."""
    store.purge_expired_uploads(session, settings.upload_ttl_days)
    if store.count_uploads(session) >= settings.max_upload_datasets:
        raise HTTPException(429, "The demo is at capacity for uploads. Please try again later.")
    dataset = store.create_upload_dataset(session, (body or DatasetCreate()).name)
    return store.describe(session, dataset)


@router.get("/datasets/{slug}", response_model=DatasetOut, responses=NOT_FOUND, tags=["datasets"])
def read_dataset(dataset: DatasetDep, session: SessionDep) -> DatasetOut:
    return store.describe(session, dataset)


@router.delete("/datasets/{slug}", status_code=204, responses=UPLOAD_ERRORS, tags=["datasets"])
def delete_dataset(dataset: WritableDep, session: SessionDep) -> Response:
    """Delete an uploaded dataset and all of its records."""
    store.delete_dataset(session, dataset)
    return Response(status_code=204)


# --- Ingestion -----------------------------------------------------------------


def _store(session: Session, dataset: Dataset, parsed: ParseResult) -> IngestResult:
    added, updated = store.upsert_records(session, dataset, parsed.records)
    return IngestResult(
        dataset=dataset.slug,
        days_received=len(parsed.records),
        days_added=added,
        days_updated=updated,
        warnings=parsed.warnings,
    )


def _import_file(
    file: UploadFile,
    parser: Callable[[bytes], ParseResult],
    session: Session,
    dataset: Dataset,
    settings: Settings,
) -> IngestResult:
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, f"The file is larger than {settings.max_upload_mb} MB.")
    if not content:
        raise HTTPException(422, "The uploaded file is empty.")
    return _store(session, dataset, parser(content))


@router.post(
    "/datasets/{slug}/records",
    response_model=IngestResult,
    responses=UPLOAD_ERRORS,
    tags=["ingestion"],
)
def add_records(body: RecordsIn, dataset: WritableDep, session: SessionDep) -> IngestResult:
    """Add or update days from JSON. Values for an existing date are merged."""
    dates = [record.date for record in body.records]
    if len(set(dates)) != len(dates):
        raise HTTPException(422, "Each date may appear only once per request.")
    return _store(session, dataset, ParseResult(records=body.records))


@router.post(
    "/datasets/{slug}/import/csv",
    response_model=IngestResult,
    responses=UPLOAD_ERRORS,
    tags=["ingestion"],
)
def import_csv(
    dataset: WritableDep,
    session: SessionDep,
    settings: SettingsDep,
    file: UploadFile = File(description="Daily-metrics CSV with a `date` column"),
) -> IngestResult:
    """Import a daily-metrics CSV (see data/samples/demo_daily_metrics.csv)."""
    return _import_file(file, parse_daily_csv, session, dataset, settings)


@router.post(
    "/datasets/{slug}/import/apple-health",
    response_model=IngestResult,
    responses=UPLOAD_ERRORS,
    tags=["ingestion"],
)
def import_apple_health(
    dataset: WritableDep,
    session: SessionDep,
    settings: SettingsDep,
    file: UploadFile = File(description="export.xml from an Apple Health export"),
) -> IngestResult:
    """Import the export.xml file from an Apple Health export, aggregated per day."""
    return _import_file(file, parse_apple_health, session, dataset, settings)


# --- Metrics and insights ------------------------------------------------------


@router.get(
    "/datasets/{slug}/metrics",
    response_model=list[DailyRecord],
    responses=NOT_FOUND,
    tags=["metrics"],
)
def list_metrics(
    dataset: DatasetDep,
    session: SessionDep,
    start: dt.date | None = Query(default=None, description="First date to include"),
    end: dt.date | None = Query(default=None, description="Last date to include"),
) -> list[DailyMetric]:
    """Raw daily records, oldest first."""
    if start and end and start > end:
        raise HTTPException(422, "'start' must not be after 'end'.")
    query = select(DailyMetric).where(DailyMetric.dataset_id == dataset.id)
    if start:
        query = query.where(DailyMetric.date >= start)
    if end:
        query = query.where(DailyMetric.date <= end)
    return list(session.scalars(query.order_by(DailyMetric.date)))


@router.get(
    "/datasets/{slug}/metrics/derived",
    response_model=list[DerivedDay],
    responses=NOT_FOUND,
    tags=["metrics"],
)
def list_derived_metrics(frame: FrameDep) -> list[DerivedDay]:
    """Per-day rolling load, workload ratio, readiness and burnout risk."""
    return derived_series(frame)


@router.get(
    "/datasets/{slug}/insights",
    response_model=InsightsOut,
    responses=NOT_FOUND,
    tags=["insights"],
)
def read_insights(dataset: DatasetDep, frame: FrameDep) -> InsightsOut:
    """Correlations, anomalies, current load, recovery trend, burnout risk and next steps."""
    return build_insights(dataset.slug, frame)


@router.get(
    "/datasets/{slug}/prediction",
    response_model=PredictionOut,
    responses=NOT_FOUND,
    tags=["insights"],
)
def read_prediction(frame: FrameDep) -> PredictionOut:
    """Probability that tomorrow is a low-readiness day, with evaluation and explanation."""
    return build_prediction(frame)
