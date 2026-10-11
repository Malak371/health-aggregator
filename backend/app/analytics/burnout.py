"""A rule-based burnout-risk score (0-100).

The score is a weighted average of five strain signals, each scaled to 0..1.
It is deliberately a transparent heuristic: every point of the score can be
traced to a component. It is not a clinical instrument.
"""

from __future__ import annotations

import math

import pandas as pd

from app.analytics.load import load_metrics
from app.analytics.recovery import resting_hr_baseline

SLEEP_TARGET_HOURS = 8.0
MIN_COMPONENTS = 2

WEIGHTS = {
    "sleep_debt": 0.25,
    "training_strain": 0.25,
    "low_mood": 0.20,
    "cognitive_strain": 0.15,
    "elevated_heart_rate": 0.15,
}

LABELS = {
    "sleep_debt": "Sleep debt (7-day average shortfall against 8 h)",
    "training_strain": "Training load rising faster than usual (ACWR)",
    "low_mood": "Low mood over the last week",
    "cognitive_strain": "Sustained study / deep-work hours",
    "elevated_heart_rate": "Resting heart rate above personal baseline",
}


def burnout_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Strain components per day, each 0 (no strain) .. 1 (maximal)."""
    load = load_metrics(frame)

    def weekly(series: pd.Series) -> pd.Series:
        return series.rolling(7, min_periods=4).mean()

    shortfall = weekly((SLEEP_TARGET_HOURS - frame["sleep_hours"]).clip(lower=0))
    hr_deviation = weekly(frame["resting_heart_rate"] - resting_hr_baseline(frame))
    return pd.DataFrame(
        {
            # 2 h short of target every night for a week is the maximum.
            "sleep_debt": (shortfall / 2).clip(0, 1),
            # ACWR 1.0 is neutral; 1.8 or more is the maximum.
            "training_strain": ((load["acwr"] - 1.0) / 0.8).clip(0, 1),
            # Weekly mood of 3.5+ is neutral; 1 is the maximum.
            "low_mood": ((3.5 - weekly(frame["mood_score"])) / 2.5).clip(0, 1),
            # 4 h/day is neutral; 9 h/day averaged over a week is the maximum.
            "cognitive_strain": ((load["study_acute"] - 4) / 5).clip(0, 1),
            # 6 bpm above baseline for a week is the maximum.
            "elevated_heart_rate": (hr_deviation / 6).clip(0, 1),
        }
    )


def burnout_risk(components: pd.DataFrame) -> pd.Series:
    """Weighted average of the available components, scaled to 0-100."""
    weights = pd.Series(WEIGHTS)
    weighted = (components * weights).sum(axis=1)
    available = components.notna().mul(weights).sum(axis=1)
    enough = components.notna().sum(axis=1) >= MIN_COMPONENTS
    return (100 * weighted / available).where(enough).rename("burnout_risk")


def risk_level(score: float | None) -> str | None:
    if score is None or math.isnan(score):
        return None
    if score < 33:
        return "low"
    if score < 66:
        return "moderate"
    return "high"
