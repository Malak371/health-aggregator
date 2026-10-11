"""Parser for Apple Health exports (the export.xml inside export.zip).

The export is a flat list of timestamped <Record> and <Workout> elements.
This module reduces them to one row per day:

- steps: summed per source, then the largest source is used, because an
  iPhone and an Apple Watch both count the same steps
- sleep_hours: time in any "asleep" stage, assigned to the day you woke up,
  again using the source that recorded the most
- resting_heart_rate: mean of the day's readings
- sugar_grams: sum of dietary sugar entries
- training_minutes: total workout duration

Apple Health has no mood score, perceived exertion or study time, so those
stay empty.
"""

from __future__ import annotations

import datetime as dt
import io
from collections import defaultdict

from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pydantic import ValidationError

from app.ingestion.errors import IngestionError
from app.ingestion.result import ParseResult
from app.schemas import DailyRecord

STEP_TYPE = "HKQuantityTypeIdentifierStepCount"
RESTING_HR_TYPE = "HKQuantityTypeIdentifierRestingHeartRate"
SUGAR_TYPE = "HKQuantityTypeIdentifierDietarySugar"
SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
# "InBed" and "Awake" are deliberately not counted as sleep.
ASLEEP_PREFIX = "HKCategoryValueSleepAnalysisAsleep"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DURATION_TO_MINUTES = {"min": 1.0, "hr": 60.0, "s": 1 / 60}


def _timestamp(value: str | None) -> dt.datetime:
    if not value:
        raise ValueError("missing timestamp")
    return dt.datetime.strptime(value, TIMESTAMP_FORMAT)


def parse_apple_health(content: bytes) -> ParseResult:
    """Aggregate an Apple Health export.xml into daily records."""
    steps: defaultdict[dt.date, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    sleep: defaultdict[dt.date, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    resting_hr: defaultdict[dt.date, list[float]] = defaultdict(list)
    sugar: defaultdict[dt.date, float] = defaultdict(float)
    workouts: defaultdict[dt.date, float] = defaultdict(float)
    skipped = 0
    root_tag: str | None = None

    try:
        # iterparse keeps memory flat: exports hold millions of records.
        for event, element in ElementTree.iterparse(io.BytesIO(content), events=("start", "end")):
            if event == "start":
                if root_tag is None:
                    root_tag = element.tag
                continue
            try:
                if element.tag == "Record":
                    kind = element.get("type")
                    source = element.get("sourceName", "unknown")
                    if kind == STEP_TYPE:
                        day = _timestamp(element.get("startDate")).date()
                        steps[day][source] += float(element.get("value", ""))
                    elif kind == RESTING_HR_TYPE:
                        day = _timestamp(element.get("startDate")).date()
                        resting_hr[day].append(float(element.get("value", "")))
                    elif kind == SUGAR_TYPE:
                        day = _timestamp(element.get("startDate")).date()
                        sugar[day] += float(element.get("value", ""))
                    elif kind == SLEEP_TYPE and element.get("value", "").startswith(ASLEEP_PREFIX):
                        start = _timestamp(element.get("startDate"))
                        end = _timestamp(element.get("endDate"))
                        sleep[end.date()][source] += (end - start).total_seconds() / 3600
                elif element.tag == "Workout":
                    day = _timestamp(element.get("startDate")).date()
                    unit = DURATION_TO_MINUTES[element.get("durationUnit", "min")]
                    workouts[day] += float(element.get("duration", "")) * unit
            except (ValueError, KeyError):
                skipped += 1
            if element.tag in ("Record", "Workout"):
                element.clear()
    except (ElementTree.ParseError, DefusedXmlException) as exc:
        raise IngestionError("The file is not valid Apple Health XML.", [str(exc)]) from exc

    if root_tag != "HealthData":
        raise IngestionError(
            "This does not look like an Apple Health export.",
            [f"Expected a <HealthData> root element, found <{root_tag}>."],
        )

    days = sorted(set(steps) | set(sleep) | set(resting_hr) | set(sugar) | set(workouts))
    if not days:
        raise IngestionError(
            "No supported records were found in the export.",
            ["Supported: step count, sleep analysis, resting heart rate, dietary sugar, workouts."],
        )

    records: list[DailyRecord] = []
    rejected = 0
    for day in days:
        hr = resting_hr.get(day)
        try:
            records.append(
                DailyRecord(
                    date=day,
                    steps=round(max(steps[day].values())) if day in steps else None,
                    sleep_hours=round(max(sleep[day].values()), 2) if day in sleep else None,
                    resting_heart_rate=round(sum(hr) / len(hr), 1) if hr else None,
                    sugar_grams=round(sugar[day], 1) if day in sugar else None,
                    training_minutes=round(workouts[day], 1) if day in workouts else None,
                )
            )
        except ValidationError:
            rejected += 1

    warnings = [
        "Apple Health has no mood, perceived exertion or study data; "
        "insights that need them are skipped."
    ]
    if skipped:
        warnings.append(f"Skipped {skipped} record(s) with unreadable values or timestamps.")
    if rejected:
        warnings.append(f"Dropped {rejected} day(s) whose totals were outside plausible ranges.")
    if not records:
        raise IngestionError("Every day in the export failed validation.", warnings)
    return ParseResult(records=records, warnings=warnings)
