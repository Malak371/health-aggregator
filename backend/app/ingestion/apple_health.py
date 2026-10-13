"""Parser for Apple Health exports: export.zip as produced by the Health app, or
the export.xml inside it.

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
import zipfile
from collections import defaultdict
from typing import IO, Protocol

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


class SupportsRead(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S %z"
DURATION_TO_MINUTES = {"min": 1.0, "hr": 60.0, "s": 1 / 60}


def _day(value: str | None) -> dt.date:
    """The local calendar date of a timestamp such as "2026-04-01 08:00:00 -0500".

    Only the date part is parsed: an export holds hundreds of thousands of
    records, and full timestamp parsing was half of the total import time.
    """
    if not value:
        raise ValueError("missing timestamp")
    return dt.date.fromisoformat(value[:10])


def _timestamp(value: str | None) -> dt.datetime:
    if not value:
        raise ValueError("missing timestamp")
    return dt.datetime.strptime(value, TIMESTAMP_FORMAT)


class _LimitedReader:
    """Wraps a binary stream and refuses to yield more than `limit` bytes.

    A zip member's declared size can be forged, so the limit is enforced on what
    is actually decompressed.
    """

    def __init__(self, stream: IO[bytes], limit: int) -> None:
        self._stream = stream
        self._remaining = limit
        self._limit = limit

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size if size and size > 0 else 1024 * 1024)
        self._remaining -= len(chunk)
        if self._remaining < 0:
            raise IngestionError(
                "The export is too large once unzipped.",
                [f"The limit is {self._limit // (1024 * 1024)} MB of XML."],
            )
        return chunk


def _export_member(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    """The health records file inside an export.zip.

    Apple names it export.xml (the name is translated in some languages) and puts
    a clinical summary, export_cda.xml, next to it, plus GPX routes in subfolders.
    """
    candidates = [
        info
        for info in archive.infolist()
        if info.filename.lower().endswith(".xml")
        and not info.filename.lower().endswith("_cda.xml")
        and info.filename.count("/") <= 1
    ]
    if not candidates:
        raise IngestionError(
            "The zip file does not contain an Apple Health export.",
            ["Expected a file such as apple_health_export/export.xml."],
        )
    named = [info for info in candidates if info.filename.lower().endswith("export.xml")]
    return max(named or candidates, key=lambda info: info.file_size)


def parse_apple_health_upload(content: bytes, max_xml_bytes: int) -> ParseResult:
    """Parse an upload that is either export.xml itself or Apple's export.zip.

    The XML inside a zip is decompressed as a stream, so it never has to fit in
    memory; `max_xml_bytes` bounds how much is read.
    """
    buffer = io.BytesIO(content)
    if not zipfile.is_zipfile(buffer):
        return parse_apple_health(content)
    try:
        with zipfile.ZipFile(buffer) as archive:
            member = _export_member(archive)
            if member.file_size > max_xml_bytes:
                raise IngestionError(
                    "The export is too large once unzipped.",
                    [
                        f"{member.filename} is {member.file_size // (1024 * 1024)} MB; "
                        f"the limit is {max_xml_bytes // (1024 * 1024)} MB."
                    ],
                )
            with archive.open(member) as stream:
                return parse_apple_health(_LimitedReader(stream, max_xml_bytes))
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        # RuntimeError: password-protected; NotImplementedError: unsupported compression.
        raise IngestionError("The zip file could not be read.", [str(exc)]) from exc


def parse_apple_health(export: bytes | SupportsRead) -> ParseResult:
    """Aggregate an Apple Health export.xml (bytes or a binary stream) into daily records."""
    steps: defaultdict[dt.date, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    sleep: defaultdict[dt.date, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    resting_hr: defaultdict[dt.date, list[float]] = defaultdict(list)
    sugar: defaultdict[dt.date, float] = defaultdict(float)
    workouts: defaultdict[dt.date, float] = defaultdict(float)
    skipped = 0
    root_tag: str | None = None

    try:
        # iterparse keeps memory flat: exports hold millions of records.
        stream = io.BytesIO(export) if isinstance(export, bytes) else export
        for event, element in ElementTree.iterparse(stream, events=("start", "end")):
            if event == "start":
                if root_tag is None:
                    root_tag = element.tag
                continue
            try:
                if element.tag == "Record":
                    kind = element.get("type")
                    source = element.get("sourceName", "unknown")
                    if kind == STEP_TYPE:
                        day = _day(element.get("startDate"))
                        steps[day][source] += float(element.get("value", ""))
                    elif kind == RESTING_HR_TYPE:
                        day = _day(element.get("startDate"))
                        resting_hr[day].append(float(element.get("value", "")))
                    elif kind == SUGAR_TYPE:
                        day = _day(element.get("startDate"))
                        sugar[day] += float(element.get("value", ""))
                    elif kind == SLEEP_TYPE and element.get("value", "").startswith(ASLEEP_PREFIX):
                        start = _timestamp(element.get("startDate"))
                        end = _timestamp(element.get("endDate"))
                        sleep[end.date()][source] += (end - start).total_seconds() / 3600
                elif element.tag == "Workout":
                    day = _day(element.get("startDate"))
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
