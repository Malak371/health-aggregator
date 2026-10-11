"""Small builders for hand-made frames in the analytics tests."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from app.schemas import METRIC_FIELDS


def make_frame(
    days: int = 0, start: str = "2026-01-01", **columns: Sequence[float]
) -> pd.DataFrame:
    """A daily frame with every metric column; unspecified metrics are NaN."""
    length = days or len(next(iter(columns.values())))
    index = pd.date_range(start, periods=length, freq="D", name="date")
    frame = pd.DataFrame(float("nan"), index=index, columns=list(METRIC_FIELDS))
    for name, values in columns.items():
        frame[name] = list(values)
    return frame
