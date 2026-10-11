"""Rolling training and cognitive load.

Training load uses the session-RPE method: minutes x perceived exertion (1-10).
The acute:chronic workload ratio (ACWR) compares the last week with the last
four weeks; a ratio well above 1 means load is rising faster than the body has
adapted to. The zone thresholds follow Gabbett (2016) and are a heuristic, not
a diagnosis.
"""

from __future__ import annotations

import math

import pandas as pd

ACUTE_DAYS = 7
CHRONIC_DAYS = 28
MIN_CHRONIC_DAYS = 21
# Used when a workout has a duration but no exertion rating (e.g. Apple Health).
DEFAULT_RPE = 5.0

ACWR_ZONES: tuple[tuple[float, str], ...] = (
    (0.8, "undertraining"),
    (1.3, "optimal"),
    (1.5, "caution"),
    (math.inf, "high"),
)


def daily_training_load(frame: pd.DataFrame) -> pd.Series:
    """Session-RPE load per day. Days with no workout logged count as rest (0)."""
    minutes = frame["training_minutes"]
    if minutes.notna().sum() == 0:
        return pd.Series(float("nan"), index=frame.index, name="training_load")
    rpe = frame["training_rpe"].fillna(DEFAULT_RPE)
    return (minutes.fillna(0.0) * rpe).rename("training_load")


def acute_chronic(daily: pd.Series, min_acute: int = ACUTE_DAYS) -> pd.DataFrame:
    """Acute (7-day) and chronic (28-day) rolling means and their ratio."""
    acute = daily.rolling(ACUTE_DAYS, min_periods=min_acute).mean()
    # The chronic window is reported from three weeks of history onwards.
    chronic = daily.rolling(CHRONIC_DAYS, min_periods=MIN_CHRONIC_DAYS).mean()
    ratio = (acute / chronic).where(chronic > 0)
    return pd.DataFrame({"acute": acute, "chronic": chronic, "ratio": ratio})


def load_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Training and study load columns for every day in the frame."""
    training = daily_training_load(frame)
    training_rolling = acute_chronic(training)
    # Study hours are logged by hand, so tolerate a few missing days per week.
    study_rolling = acute_chronic(frame["study_hours"], min_acute=4)
    return pd.DataFrame(
        {
            "training_load": training,
            "training_acute": training_rolling["acute"],
            "training_chronic": training_rolling["chronic"],
            "acwr": training_rolling["ratio"],
            "study_acute": study_rolling["acute"],
            "study_chronic": study_rolling["chronic"],
            "study_ratio": study_rolling["ratio"],
        }
    )


def acwr_zone(ratio: float | None) -> str | None:
    if ratio is None or math.isnan(ratio):
        return None
    for upper, zone in ACWR_ZONES:
        if ratio < upper:
            return zone
    return None
