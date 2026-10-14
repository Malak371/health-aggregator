"""This week against last week, signal by signal."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.analytics.load import daily_training_load
from app.analytics.recovery import readiness_score

WEEK = 7
# A week with fewer logged days than this is not summarised.
MIN_DAYS_PER_WEEK = 3


@dataclass(frozen=True)
class Signal:
    key: str
    label: str
    unit: str
    # None when neither direction is simply "good" (e.g. training load).
    higher_is_better: bool | None
    digits: int = 1


SIGNALS: tuple[Signal, ...] = (
    Signal("readiness", "Readiness", "/ 100", True),
    Signal("sleep_hours", "Sleep", "h", True),
    Signal("resting_heart_rate", "Resting heart rate", "bpm", False),
    Signal("mood_score", "Mood", "/ 5", True),
    Signal("training_load", "Training load per day", "", None, 0),
    Signal("study_hours", "Study per day", "h", None),
    Signal("steps", "Steps per day", "", True, 0),
)


@dataclass(frozen=True)
class WeeklyChange:
    key: str
    label: str
    unit: str
    this_week: float
    last_week: float
    change: float
    higher_is_better: bool | None


def weekly_changes(frame: pd.DataFrame) -> list[WeeklyChange]:
    """Daily averages over the last 7 days and the 7 before, where both have data."""
    table = frame.assign(readiness=readiness_score(frame), training_load=daily_training_load(frame))
    recent, previous = table.iloc[-WEEK:], table.iloc[-2 * WEEK : -WEEK]
    changes = []
    for signal in SIGNALS:
        now, before = recent[signal.key], previous[signal.key]
        if now.notna().sum() < MIN_DAYS_PER_WEEK or before.notna().sum() < MIN_DAYS_PER_WEEK:
            continue
        this_week, last_week = float(now.mean()), float(before.mean())
        changes.append(
            WeeklyChange(
                key=signal.key,
                label=signal.label,
                unit=signal.unit,
                this_week=round(this_week, signal.digits),
                last_week=round(last_week, signal.digits),
                # Rounded after subtracting, so it matches the recovery trend exactly.
                change=round(this_week - last_week, signal.digits),
                higher_is_better=signal.higher_is_better,
            )
        )
    return changes
