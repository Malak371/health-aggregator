"""Assemble API responses from the analytics and ML modules."""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, cast

import pandas as pd

from app.analytics import anomalies as anomaly_rules
from app.analytics.burnout import LABELS, WEIGHTS, burnout_components, burnout_risk, risk_level
from app.analytics.correlations import correlation_insights
from app.analytics.load import acwr_zone, load_metrics
from app.analytics.recommendations import Snapshot, generate_recommendations
from app.analytics.recovery import readiness_score, recovery_trend
from app.analytics.weekly import weekly_changes
from app.ml.model import InsufficientDataError, train_and_explain
from app.schemas import (
    AnomalyOut,
    BurnoutComponent,
    BurnoutSummary,
    CorrelationOut,
    DerivedDay,
    InsightsOut,
    LoadSummary,
    PredictionOut,
    RecommendationOut,
    RecoverySummary,
    WeeklyChangeOut,
)
from app.services.cache import ResultCache, frame_fingerprint

DISCLAIMER = (
    "Insights describe patterns in the data you provided. They are not medical advice "
    "and must not be used to diagnose, treat or prevent any condition."
)
prediction_cache: ResultCache[PredictionOut] = ResultCache()
MODEL_NAME = "Logistic regression (L2, standardised features)"
RECENT_DAYS = 14


def _number(value: Any, digits: int = 2) -> float | None:
    """NaN-safe rounding: JSON has no NaN, so missing values become null."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def derived_series(frame: pd.DataFrame) -> list[DerivedDay]:
    load = load_metrics(frame)
    table = load.assign(
        readiness=readiness_score(frame),
        burnout_risk=burnout_risk(burnout_components(frame)),
    )
    fields = [name for name in DerivedDay.model_fields if name != "date"]
    return [
        DerivedDay(
            date=cast(pd.Timestamp, day).date(), **{name: _number(row[name]) for name in fields}
        )
        for day, row in table.iterrows()
    ]


def build_insights(slug: str, frame: pd.DataFrame) -> InsightsOut:
    load = load_metrics(frame)
    readiness = readiness_score(frame)
    components = burnout_components(frame)
    risk = burnout_risk(components)
    trend = recovery_trend(readiness)
    correlations = correlation_insights(frame.join(load[["training_load"]]))

    found = [
        *anomaly_rules.detect_poor_sleep(frame),
        *anomaly_rules.detect_high_resting_hr(frame),
        *anomaly_rules.detect_load_spikes(load["acwr"]),
    ]
    found.sort(key=lambda anomaly: anomaly.date, reverse=True)

    last_day = frame.index[-1]
    recent_start = last_day.date() - dt.timedelta(days=RECENT_DAYS)
    latest_load = load.iloc[-1]
    score = _number(risk.iloc[-1], 1)
    sleep_sugar = next(
        (c.r for c in correlations if (c.x, c.y) == ("sleep_hours", "sugar_grams")), None
    )
    recommendations = generate_recommendations(
        Snapshot(
            avg_sleep_7d=_number(frame["sleep_hours"].iloc[-7:].mean()),
            sleep_sugar_r=sleep_sugar,
            recent_high_hr_days=sum(
                1 for a in found if a.kind == "high_resting_hr" and a.date > recent_start
            ),
            acwr=_number(latest_load["acwr"]),
            study_acute=_number(latest_load["study_acute"]),
            burnout_score=score,
            recovery_direction=trend.direction,
            days_logged=int(frame.notna().any(axis=1).sum()),
        )
    )

    return InsightsOut(
        dataset=slug,
        days=int(frame.notna().any(axis=1).sum()),
        first_date=frame.index[0].date(),
        last_date=last_day.date(),
        weekly=[WeeklyChangeOut.model_validate(w) for w in weekly_changes(frame)],
        correlations=[CorrelationOut.model_validate(c) for c in correlations],
        anomalies=[AnomalyOut.model_validate(a) for a in found],
        load=LoadSummary(
            as_of=last_day.date(),
            training_acute=_number(latest_load["training_acute"], 1),
            training_chronic=_number(latest_load["training_chronic"], 1),
            acwr=_number(latest_load["acwr"]),
            acwr_zone=acwr_zone(_number(latest_load["acwr"])),
            study_acute=_number(latest_load["study_acute"]),
            study_chronic=_number(latest_load["study_chronic"]),
        ),
        recovery=RecoverySummary(
            readiness_today=_number(readiness.iloc[-1], 1),
            direction=trend.direction,
            last_7_days=trend.last_7_days,
            previous_7_days=trend.previous_7_days,
            change=trend.change,
        ),
        burnout=BurnoutSummary(
            score=score,
            level=risk_level(score),
            components=[
                BurnoutComponent(
                    name=name,
                    label=LABELS[name],
                    value=_number(components[name].iloc[-1]),
                    weight=weight,
                )
                for name, weight in WEIGHTS.items()
            ],
        ),
        recommendations=[RecommendationOut.model_validate(r) for r in recommendations],
        disclaimer=DISCLAIMER,
    )


def build_prediction(frame: pd.DataFrame) -> PredictionOut:
    """Train, evaluate and explain the model, reusing the result for unchanged data.

    Training runs a cross-validated grid search, which takes seconds on a small
    shared CPU, so each distinct dataset state is computed once per process.
    """
    return prediction_cache.get_or_compute(frame_fingerprint(frame), lambda: _predict(frame))


def _predict(frame: pd.DataFrame) -> PredictionOut:
    try:
        report = train_and_explain(frame)
    except InsufficientDataError as exc:
        return PredictionOut(available=False, reason=str(exc), disclaimer=DISCLAIMER)
    return PredictionOut(
        available=True,
        model=MODEL_NAME,
        threshold=report.threshold,
        regularisation_c=report.regularisation_c,
        evaluation=report.evaluation,
        feature_effects=report.feature_effects,
        prediction=report.prediction,
        disclaimer=DISCLAIMER,
    )
