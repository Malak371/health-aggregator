"""Feature engineering for the next-day readiness model.

Every feature for day t uses data from day t or earlier only, so nothing from
the day being predicted (t + 1) can leak into the inputs.
"""

from __future__ import annotations

import pandas as pd

from app.analytics.load import load_metrics
from app.analytics.recovery import readiness_score, resting_hr_baseline

FEATURE_LABELS: dict[str, str] = {
    "readiness_today": "Today's readiness",
    "sleep_3d_avg": "Sleep, 3-day average",
    "sleep_shortfall_7d": "Sleep shortfall vs 8 h, 7-day average",
    "rhr_deviation": "Resting heart rate vs baseline",
    "mood_3d_avg": "Mood, 3-day average",
    "training_load_today": "Today's training load",
    "training_acute": "Training load, 7-day average",
    "acwr": "Acute:chronic workload ratio",
    "study_hours_today": "Today's study hours",
    "study_acute": "Study hours, 7-day average",
}


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    load = load_metrics(frame)
    return pd.DataFrame(
        {
            "readiness_today": readiness_score(frame),
            "sleep_3d_avg": frame["sleep_hours"].rolling(3, min_periods=2).mean(),
            "sleep_shortfall_7d": (8 - frame["sleep_hours"])
            .clip(lower=0)
            .rolling(7, min_periods=4)
            .mean(),
            "rhr_deviation": frame["resting_heart_rate"] - resting_hr_baseline(frame),
            "mood_3d_avg": frame["mood_score"].rolling(3, min_periods=2).mean(),
            "training_load_today": load["training_load"],
            "training_acute": load["training_acute"],
            "acwr": load["acwr"],
            "study_hours_today": frame["study_hours"],
            "study_acute": load["study_acute"],
        }
    )


def next_day_readiness(frame: pd.DataFrame) -> pd.Series:
    """The regression target behind the label: tomorrow's readiness score."""
    return readiness_score(frame).shift(-1).rename("readiness_next_day")
