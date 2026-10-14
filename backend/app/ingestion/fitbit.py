"""Parser for Fitbit data exports (the zip from Fitbit's "Export your account
archive", or the Fitbit folder of a Google Takeout).

The archive holds many JSON files, split by month or by a running number, with
names that start with the kind of data:

- steps-2019-11-01.json                per-minute step counts
- sleep-2019-11-01.json                one entry per sleep log
- resting_heart_rate-2019-11-01.json   one entry per day
- exercise-0.json                      one entry per logged or detected activity

Files are matched by name wherever they sit in the archive, because the folder
layout has changed between export versions. Each kind is reduced to one value
per day: steps summed, minutes asleep summed per date of sleep, resting heart
rate as recorded, and active minutes of workouts summed.

Fitbit also logs every walk it detects on its own (more than half of all
entries in real data). Those are everyday movement, already counted in steps,
so they are not treated as training; other detected activities (runs, rides)
and everything the wearer started or entered are. Fitbit has no perceived
exertion, so training load falls back to the default rating.

Not imported: anything Fitbit does not record, such as mood or study time. Step
timestamps in the export are in UTC, so a day's total can differ slightly from
what the Fitbit app shows for people far from that time zone.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterable
from typing import Any

import pandas as pd
from pydantic import ValidationError

from app.ingestion.errors import IngestionError
from app.ingestion.result import ParseResult
from app.schemas import DailyRecord

Rows = list[dict[str, Any]]

# "steps-2019-11-01.json", "sleep-3.json", or the bare "steps.json" some tools write.
KINDS = ("steps", "sleep", "resting_heart_rate", "exercise")
FILE_PATTERN = re.compile(rf"(?:^|/)({'|'.join(KINDS)})(?:-[^/]*)?\.json$", re.IGNORECASE)


def _days(values: pd.Series) -> pd.Series:
    """Calendar days from Fitbit timestamps, which come as "11/01/19 00:00:00" or ISO."""
    text = values.astype(str)
    american = text.str.contains("/", regex=False)
    parsed = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    if american.any():
        parsed[american] = pd.to_datetime(
            text[american], format="%m/%d/%y %H:%M:%S", errors="coerce"
        )
    if (~american).any():
        parsed[~american] = pd.to_datetime(text[~american], format="ISO8601", errors="coerce")
    days: pd.Series = parsed.dt.normalize()
    return days


def daily_steps(rows: Rows) -> pd.Series:
    table = pd.DataFrame(rows, columns=["dateTime", "value"])
    counts = pd.to_numeric(table["value"], errors="coerce")
    daily = counts.groupby(_days(table["dateTime"])).sum()
    # A day with no steps at all means the tracker was not worn.
    return daily.where(daily > 0).clip(upper=200_000)


def daily_sleep_hours(rows: Rows) -> pd.Series:
    table = pd.DataFrame(rows, columns=["dateOfSleep", "minutesAsleep"])
    minutes = pd.to_numeric(table["minutesAsleep"], errors="coerce")
    return (minutes.groupby(_days(table["dateOfSleep"])).sum() / 60).clip(upper=24)


def daily_resting_heart_rate(rows: Rows) -> pd.Series:
    table = pd.DataFrame(
        {
            "day": _days(pd.Series([row.get("dateTime") for row in rows], dtype="object")),
            "bpm": [
                row["value"].get("value") if isinstance(row.get("value"), dict) else None
                for row in rows
            ],
        }
    )
    series = pd.to_numeric(table["bpm"], errors="coerce").groupby(table["day"]).mean()
    # The export writes 0 for days without a reading.
    return series.where(series.between(25, 220))


def is_training(row: dict[str, Any]) -> bool:
    """Whether an exercise entry counts as training (see the module docstring)."""
    auto_walk = row.get("logType") == "auto_detected" and row.get("activityName") == "Walk"
    return not auto_walk


def daily_training_minutes(rows: Rows) -> pd.Series:
    sessions = [row for row in rows if is_training(row)]
    table = pd.DataFrame(sessions, columns=["startTime", "originalStartTime", "activeDuration"])
    start = table["startTime"].where(table["startTime"].notna(), table["originalStartTime"])
    minutes = pd.to_numeric(table["activeDuration"], errors="coerce") / 60_000
    daily = minutes.groupby(_days(start)).sum()
    return daily.where(daily > 0).clip(upper=1440)


READERS = {
    "steps": ("steps", daily_steps),
    "sleep": ("sleep_hours", daily_sleep_hours),
    "resting_heart_rate": ("resting_heart_rate", daily_resting_heart_rate),
    "exercise": ("training_minutes", daily_training_minutes),
}


DIGITS = {"steps": 0, "sleep_hours": 2, "resting_heart_rate": 1, "training_minutes": 1}


def _members(archive: zipfile.ZipFile) -> Iterable[tuple[str, zipfile.ZipInfo]]:
    for info in archive.infolist():
        match = FILE_PATTERN.search(info.filename)
        if match and not info.is_dir():
            yield match.group(1).lower(), info


def parse_fitbit_export(content: bytes, max_unzipped_bytes: int) -> ParseResult:
    """Aggregate a Fitbit export zip into daily records.

    `max_unzipped_bytes` bounds the total size of the files that are read,
    counted on the bytes actually decompressed.
    """
    buffer = io.BytesIO(content)
    if not zipfile.is_zipfile(buffer):
        raise IngestionError(
            "A Fitbit export is a zip file.", ["Upload the archive as Fitbit or Google sent it."]
        )

    rows: dict[str, Rows] = {kind: [] for kind in KINDS}
    unreadable = 0
    remaining = max_unzipped_bytes
    try:
        with zipfile.ZipFile(buffer) as archive:
            for kind, info in _members(archive):
                with archive.open(info) as stream:
                    data = stream.read(remaining + 1)
                remaining -= len(data)
                if remaining < 0:
                    raise IngestionError(
                        "The export is too large once unzipped.",
                        [f"The limit is {max_unzipped_bytes // (1024 * 1024)} MB of data files."],
                    )
                try:
                    parsed = json.loads(data)
                except ValueError:
                    unreadable += 1
                    continue
                if isinstance(parsed, list):
                    rows[kind].extend(row for row in parsed if isinstance(row, dict))
                else:
                    unreadable += 1
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise IngestionError("The zip file could not be read.", [str(exc)]) from exc

    columns = {
        column: reader(rows[kind]) for kind, (column, reader) in READERS.items() if rows[kind]
    }
    if not columns:
        raise IngestionError(
            "No Fitbit data files were found in the zip.",
            ["Expected files such as steps-2024-01-01.json, sleep-..., exercise-... "],
        )
    frame = pd.DataFrame(columns)
    frame = frame[frame.index.notna()].dropna(how="all").sort_index()

    records: list[DailyRecord] = []
    rejected = 0
    for day, row in frame.iterrows():
        values: dict[str, Any] = {
            str(name): round(float(value), DIGITS[str(name)])
            for name, value in row.items()
            if pd.notna(value)
        }
        if "steps" in values:
            values["steps"] = int(values["steps"])
        try:
            records.append(DailyRecord(date=pd.Timestamp(str(day)).date(), **values))
        except ValidationError:
            rejected += 1

    warnings = [
        "Imported steps, sleep, resting heart rate and workout minutes. Fitbit has no effort "
        "rating, mood or study data; use the daily check-in to add those."
    ]
    missing = [kind.replace("_", " ") for kind in KINDS if not rows[kind]]
    if missing:
        warnings.append(f"No files found for: {', '.join(missing)}.")
    if unreadable:
        warnings.append(f"Skipped {unreadable} file(s) that were not valid JSON lists.")
    if rejected:
        warnings.append(f"Dropped {rejected} day(s) whose totals were outside plausible ranges.")
    if not records:
        raise IngestionError("The export contains no usable days.", warnings)
    return ParseResult(records=records, warnings=warnings)
