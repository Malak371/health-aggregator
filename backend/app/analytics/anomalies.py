"""Anomaly detection against personal baselines rather than global norms."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

POOR_SLEEP_HOURS = 5.0
HR_WINDOW_DAYS = 7
HR_SIGMA = 1.5
# Ignore statistically "unusual" readings that are within normal day-to-day noise.
HR_MIN_EXCESS_BPM = 3.0
LOAD_SPIKE_ACWR = 1.5


@dataclass(frozen=True)
class Anomaly:
    date: dt.date
    kind: str
    value: float
    baseline: float | None
    message: str


def detect_poor_sleep(frame: pd.DataFrame, threshold: float = POOR_SLEEP_HOURS) -> list[Anomaly]:
    low = frame["sleep_hours"][frame["sleep_hours"] < threshold]
    return [
        Anomaly(
            date=day.date(),
            kind="poor_sleep",
            value=float(hours),
            baseline=threshold,
            message=f"Only {hours:.1f} h of sleep (under {threshold:g} h).",
        )
        for day, hours in low.items()
    ]


def detect_high_resting_hr(
    frame: pd.DataFrame, window: int = HR_WINDOW_DAYS, sigma: float = HR_SIGMA
) -> list[Anomaly]:
    """Days whose resting heart rate is well above the preceding `window` days.

    The baseline is shifted by one day so a spike cannot hide itself by
    inflating its own mean and standard deviation.
    """
    heart_rate = frame["resting_heart_rate"]
    history = heart_rate.rolling(window, min_periods=window).agg(["mean", "std"]).shift(1)
    limit = history["mean"] + sigma * history["std"]
    is_high = (heart_rate > limit) & (heart_rate - history["mean"] >= HR_MIN_EXCESS_BPM)
    return [
        Anomaly(
            date=day.date(),
            kind="high_resting_hr",
            value=float(heart_rate[day]),
            baseline=round(float(history["mean"][day]), 1),
            message=(
                f"Resting heart rate {heart_rate[day]:.0f} bpm, "
                f"well above the previous {window}-day average of {history['mean'][day]:.0f}."
            ),
        )
        for day in heart_rate.index[is_high]
    ]


def detect_load_spikes(acwr: pd.Series, threshold: float = LOAD_SPIKE_ACWR) -> list[Anomaly]:
    """First day of each stretch where the acute:chronic ratio exceeds the threshold."""
    above = acwr > threshold
    starts = above & ~above.shift(1, fill_value=False)
    return [
        Anomaly(
            date=day.date(),
            kind="training_load_spike",
            value=round(float(acwr[day]), 2),
            baseline=threshold,
            message=(
                f"Training load over the last week is {acwr[day]:.1f}x the four-week average."
            ),
        )
        for day in acwr.index[starts]
    ]
