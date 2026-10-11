"""Turn stored records into the DataFrame shape the analytics expect."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from app.schemas import METRIC_FIELDS


def records_to_frame(records: Iterable[Any]) -> pd.DataFrame:
    """Build a frame with one row per calendar day and one column per metric.

    Days missing from the input are inserted as empty rows so that rolling
    windows always span real time, not just the days that were logged.
    """
    rows = [
        {"date": record.date, **{name: getattr(record, name) for name in METRIC_FIELDS}}
        for record in records
    ]
    frame = pd.DataFrame(rows, columns=["date", *METRIC_FIELDS])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index().astype("float64")
    if frame.empty:
        return frame
    return frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="D", name="date"))
