"""Daily readiness score and recovery trend.

Readiness (0-100) blends the signals that respond to recovery: last night's
sleep, resting heart rate relative to the personal baseline, and mood. It is a
transparent weighted average, re-weighted over whichever signals exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BASELINE_DAYS = 28
READINESS_WEIGHTS = {"sleep": 0.4, "heart_rate": 0.3, "mood": 0.3}
TREND_THRESHOLD = 3.0  # readiness points between consecutive weeks


def resting_hr_baseline(frame: pd.DataFrame) -> pd.Series:
    """Mean resting heart rate over the previous four weeks, excluding today."""
    return frame["resting_heart_rate"].rolling(BASELINE_DAYS, min_periods=7).mean().shift(1)


def readiness_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Each component scaled to 0 (poor) .. 1 (good)."""
    deviation = frame["resting_heart_rate"] - resting_hr_baseline(frame)
    return pd.DataFrame(
        {
            # 4 h or less scores 0, 8 h or more scores 1.
            "sleep": ((frame["sleep_hours"] - 4) / 4).clip(0, 1),
            # At baseline scores 0.5; 5 bpm above scores 0, 5 bpm below scores 1.
            "heart_rate": (0.5 - deviation / 10).clip(0, 1),
            "mood": (frame["mood_score"] - 1) / 4,
        }
    )


def readiness_score(frame: pd.DataFrame) -> pd.Series:
    components = readiness_components(frame)
    weights = pd.Series(READINESS_WEIGHTS)
    weighted = (components * weights).sum(axis=1, min_count=1)
    available = components.notna().mul(weights).sum(axis=1)
    return (100 * weighted / available.where(available > 0)).rename("readiness")


@dataclass(frozen=True)
class RecoveryTrend:
    direction: str  # "improving" | "stable" | "declining" | "unknown"
    last_7_days: float | None
    previous_7_days: float | None
    change: float | None


def recovery_trend(readiness: pd.Series) -> RecoveryTrend:
    """Compare mean readiness over the last week with the week before."""
    recent, previous = readiness.iloc[-7:], readiness.iloc[-14:-7]
    if recent.notna().sum() < 4 or previous.notna().sum() < 4:
        return RecoveryTrend("unknown", None, None, None)
    last, before = float(recent.mean()), float(previous.mean())
    change = last - before
    if change > TREND_THRESHOLD:
        direction = "improving"
    elif change < -TREND_THRESHOLD:
        direction = "declining"
    else:
        direction = "stable"
    return RecoveryTrend(direction, round(last, 1), round(before, 1), round(change, 1))
