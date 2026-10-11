"""Pydantic schemas: validated input records and API response shapes."""

from __future__ import annotations

import datetime as dt

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
    """Every 4xx response has this shape."""

    detail: ErrorDetail


class DatasetCreate(BaseModel):
    name: str = Field(default="My data", min_length=1, max_length=120)


class DerivedDay(BaseModel):
    """Computed series for one day; a value is null until enough history exists."""

    date: dt.date
    training_load: float | None
    training_acute: float | None
    training_chronic: float | None
    acwr: float | None
    study_acute: float | None
    study_chronic: float | None
    readiness: float | None
    burnout_risk: float | None


class CorrelationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    x: str
    y: str
    lag_days: int
    r: float
    p_value: float
    n: int
    strength: str
    insight: str


class AnomalyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    date: dt.date
    kind: str
    value: float
    baseline: float | None
    message: str


class LoadSummary(BaseModel):
    as_of: dt.date
    training_acute: float | None
    training_chronic: float | None
    acwr: float | None
    acwr_zone: str | None
    study_acute: float | None
    study_chronic: float | None


class RecoverySummary(BaseModel):
    readiness_today: float | None
    direction: str
    last_7_days: float | None
    previous_7_days: float | None
    change: float | None


class BurnoutComponent(BaseModel):
    name: str
    label: str
    # 0 (no strain) .. 1 (maximal); null when the signal is not recorded.
    value: float | None
    weight: float


class BurnoutSummary(BaseModel):
    score: float | None
    level: str | None
    components: list[BurnoutComponent]


class RecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    message: str
    reason: str


class InsightsOut(BaseModel):
    dataset: str
    days: int
    first_date: dt.date
    last_date: dt.date
    correlations: list[CorrelationOut]
    anomalies: list[AnomalyOut]
    load: LoadSummary
    recovery: RecoverySummary
    burnout: BurnoutSummary
    recommendations: list[RecommendationOut]
    disclaimer: str


class EvaluationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    train_days: int
    test_days: int
    test_start: dt.date
    test_end: dt.date
    positive_rate_test: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    baseline_majority_accuracy: float
    baseline_persistence_roc_auc: float | None


class FeatureEffectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feature: str
    label: str
    coefficient: float


class ContributionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    feature: str
    label: str
    value: float | None
    contribution: float


class PredictionDetail(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    based_on: dt.date
    predicts: dt.date
    probability_low_readiness: float
    contributions: list[ContributionOut]


class PredictionOut(BaseModel):
    """`available` is false, with a `reason`, when the data cannot support a model."""

    model_config = ConfigDict(from_attributes=True)

    available: bool
    reason: str | None = None
    model: str | None = None
    threshold: float | None = None
    regularisation_c: float | None = None
    evaluation: EvaluationOut | None = None
    feature_effects: list[FeatureEffectOut] = []
    prediction: PredictionDetail | None = None
    disclaimer: str
