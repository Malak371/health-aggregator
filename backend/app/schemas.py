"""Pydantic schemas: validated input records and API response shapes."""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

METRIC_FIELDS: tuple[str, ...] = (
    "sleep_hours",
    "resting_heart_rate",
    "steps",
    "mood_score",
    "sugar_grams",
    "training_minutes",
    "training_rpe",
    "study_hours",
)


class DailyRecord(BaseModel):
    """One day of signals. Ranges reject typos and unit mix-ups, not unusual days."""

    model_config = ConfigDict(from_attributes=True, extra="forbid")

    date: dt.date
    sleep_hours: float | None = Field(default=None, ge=0, le=24)
    resting_heart_rate: float | None = Field(default=None, ge=25, le=220)
    steps: int | None = Field(default=None, ge=0, le=200_000)
    mood_score: float | None = Field(default=None, ge=1, le=5)
    sugar_grams: float | None = Field(default=None, ge=0, le=1000)
    training_minutes: float | None = Field(default=None, ge=0, le=1440)
    training_rpe: float | None = Field(default=None, ge=1, le=10)
    study_hours: float | None = Field(default=None, ge=0, le=24)

    @model_validator(mode="after")
    def _has_a_metric(self) -> DailyRecord:
        if all(getattr(self, name) is None for name in METRIC_FIELDS):
            raise ValueError("at least one metric must be provided for the day")
        return self


class RecordsIn(BaseModel):
    records: list[DailyRecord] = Field(min_length=1, max_length=5000)


class DatasetOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    slug: str
    name: str
    source: str
    created_at: dt.datetime
    days: int = 0
    first_date: dt.date | None = None
    last_date: dt.date | None = None


class IngestResult(BaseModel):
    dataset: str
    days_received: int
    days_added: int
    days_updated: int
    warnings: list[str] = []


class ErrorDetail(BaseModel):
    message: str
    errors: list[str] = []


class ErrorResponse(BaseModel):
    detail: ErrorDetail


JsonDict = dict[str, Any]
