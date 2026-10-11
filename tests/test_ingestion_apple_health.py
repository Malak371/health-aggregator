import datetime as dt

import pytest

from app.ingestion.apple_health import parse_apple_health
from app.ingestion.errors import IngestionError
from tests.conftest import APPLE_HEALTH_XML


def export(*elements: str) -> bytes:
    return f"<HealthData locale='en_US'>{''.join(elements)}</HealthData>".encode()


def record(kind: str, start: str, end: str, value: str, source: str = "Watch") -> str:
    return (
        f"<Record type='{kind}' sourceName='{source}' value='{value}' "
        f"startDate='{start} -0500' endDate='{end} -0500'/>"
    )


STEPS = "HKQuantityTypeIdentifierStepCount"
SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"
RESTING_HR = "HKQuantityTypeIdentifierRestingHeartRate"
ASLEEP = "HKCategoryValueSleepAnalysisAsleepCore"


def test_parses_the_bundled_sample() -> None:
    result = parse_apple_health(APPLE_HEALTH_XML.read_bytes())

    assert len(result.records) == 90
    first = result.records[0]
    assert first.date == dt.date(2026, 4, 1)
    assert first.sleep_hours == pytest.approx(5.9, abs=0.05)
    assert first.resting_heart_rate is not None
    assert first.mood_score is None
    assert "no mood" in result.warnings[0]


def test_steps_use_the_largest_source_instead_of_double_counting() -> None:
    content = export(
        record(STEPS, "2026-01-01 08:00:00", "2026-01-01 09:00:00", "4000", "Watch"),
        record(STEPS, "2026-01-01 12:00:00", "2026-01-01 13:00:00", "2000", "Watch"),
        record(STEPS, "2026-01-01 08:00:00", "2026-01-01 09:00:00", "3500", "iPhone"),
    )

    assert parse_apple_health(content).records[0].steps == 6000


def test_sleep_counts_asleep_stages_on_the_wake_up_day() -> None:
    content = export(
        record(
            SLEEP, "2026-01-01 22:30:00", "2026-01-02 06:30:00", "HKCategoryValueSleepAnalysisInBed"
        ),
        record(SLEEP, "2026-01-01 23:00:00", "2026-01-02 03:00:00", ASLEEP),
        record(
            SLEEP, "2026-01-02 03:00:00", "2026-01-02 03:30:00", "HKCategoryValueSleepAnalysisAwake"
        ),
        record(
            SLEEP,
            "2026-01-02 03:30:00",
            "2026-01-02 06:30:00",
            "HKCategoryValueSleepAnalysisAsleepREM",
        ),
    )

    records = parse_apple_health(content).records

    assert len(records) == 1
    assert records[0].date == dt.date(2026, 1, 2)
    assert records[0].sleep_hours == 7.0


def test_resting_heart_rate_is_averaged_and_workouts_are_summed() -> None:
    content = export(
        record(RESTING_HR, "2026-01-01 07:00:00", "2026-01-01 07:00:00", "58"),
        record(RESTING_HR, "2026-01-01 19:00:00", "2026-01-01 19:00:00", "62"),
        "<Workout duration='30' durationUnit='min' startDate='2026-01-01 17:00:00 -0500'/>",
        "<Workout duration='0.5' durationUnit='hr' startDate='2026-01-01 19:00:00 -0500'/>",
    )

    day = parse_apple_health(content).records[0]

    assert day.resting_heart_rate == 60.0
    assert day.training_minutes == 60.0


def test_unreadable_records_are_skipped_with_a_warning() -> None:
    content = export(
        record(STEPS, "2026-01-01 08:00:00", "2026-01-01 09:00:00", "4000"),
        record(STEPS, "yesterday", "today", "100"),
        record(STEPS, "2026-01-01 10:00:00", "2026-01-01 11:00:00", "lots"),
    )

    result = parse_apple_health(content)

    assert result.records[0].steps == 4000
    assert "Skipped 2 record(s)" in result.warnings[1]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"this is not xml", "not valid Apple Health XML"),
        (b"<gpx><trk/></gpx>", "does not look like an Apple Health export"),
        (export("<Record type='HKQuantityTypeIdentifierHeight' value='1'/>"), "No supported"),
    ],
)
def test_unusable_files_are_rejected(content: bytes, expected: str) -> None:
    with pytest.raises(IngestionError, match=expected):
        parse_apple_health(content)


def test_entity_expansion_attacks_are_refused() -> None:
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE HealthData [<!ENTITY a "aaaa">'
        b'<!ENTITY b "&a;&a;&a;">]><HealthData>&b;</HealthData>'
    )

    with pytest.raises(IngestionError, match="not valid Apple Health XML"):
        parse_apple_health(bomb)
