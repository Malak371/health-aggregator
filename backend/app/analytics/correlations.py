"""Pearson correlations between pairs of daily signals.

Correlation is not causation: the insight text always says "tends to go with",
never "causes".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats

MIN_SAMPLES = 14

LABELS = {
    "sleep_hours": "sleep",
    "mood_score": "mood",
    "sugar_grams": "sugar intake",
    "steps": "step count",
    "resting_heart_rate": "resting heart rate",
    "training_load": "training load",
    "study_hours": "study time",
}

# (x, y, lag): y is taken `lag` days after x.
DEFAULT_PAIRS: tuple[tuple[str, str, int], ...] = (
    ("sleep_hours", "mood_score", 0),
    ("sleep_hours", "sugar_grams", 0),
    ("sleep_hours", "steps", 0),
    ("study_hours", "sleep_hours", 0),
    ("training_load", "resting_heart_rate", 1),
)


@dataclass(frozen=True)
class Correlation:
    x: str
    y: str
    lag_days: int
    r: float
    p_value: float
    n: int
    strength: str
    insight: str


def strength_of(r: float) -> str:
    size = abs(r)
    if size < 0.2:
        return "negligible"
    if size < 0.4:
        return "weak"
    if size < 0.6:
        return "moderate"
    return "strong"


def describe(x: str, y: str, lag_days: int, r: float) -> str:
    x_label, y_label = LABELS.get(x, x), LABELS.get(y, y)
    when = " the next day" if lag_days == 1 else f" {lag_days} days later" if lag_days else ""
    strength = strength_of(r)
    if strength == "negligible":
        return f"No clear link between {x_label} and {y_label}{when}."
    direction = "higher" if r > 0 else "lower"
    return f"More {x_label} tends to go with {direction} {y_label}{when} ({strength} link)."


def correlate(
    frame: pd.DataFrame, x: str, y: str, lag_days: int = 0, min_samples: int = MIN_SAMPLES
) -> Correlation | None:
    """Correlate x with y `lag_days` later. None when there is too little data."""
    if x not in frame or y not in frame:
        return None
    pair = pd.DataFrame({"x": frame[x], "y": frame[y].shift(-lag_days)}).dropna()
    if len(pair) < min_samples or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None
    result = stats.pearsonr(pair["x"], pair["y"])
    r = float(result.statistic)
    return Correlation(
        x=x,
        y=y,
        lag_days=lag_days,
        r=round(r, 3),
        p_value=float(result.pvalue),
        n=len(pair),
        strength=strength_of(r),
        insight=describe(x, y, lag_days, r),
    )


def correlation_insights(frame: pd.DataFrame) -> list[Correlation]:
    """Every default pair that has enough overlapping data."""
    found = (correlate(frame, x, y, lag) for x, y, lag in DEFAULT_PAIRS)
    return [item for item in found if item is not None]
