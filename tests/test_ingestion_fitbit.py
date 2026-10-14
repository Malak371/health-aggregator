import datetime as dt
import io
import json
import zipfile
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.ingestion.errors import IngestionError
from app.ingestion.fitbit import (
    daily_resting_heart_rate,
    daily_sleep_hours,
    daily_steps,
    parse_fitbit_export,
)
from tests.conftest import APPLE_HEALTH_XML, FITBIT_ZIP

MB = 1024 * 1024


def export(files: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content if isinstance(content, str) else json.dumps(content))
    return buffer.getvalue()


STEPS = [
    {"dateTime": "01/01/24 08:00:00", "value": "1200"},
    {"dateTime": "01/01/24 18:30:00", "value": "800"},
    {"dateTime": "01/02/24 09:00:00", "value": "0"},
]
SLEEP = [
    {"dateOfSleep": "2024-01-01", "minutesAsleep": 420, "mainSleep": True},
    {"dateOfSleep": "2024-01-01", "minutesAsleep": 30, "mainSleep": False},
]
RESTING = [
    {"dateTime": "01/01/24 00:00:00", "value": {"date": "01/01/24", "value": 58.26, "error": 6}},
    {"dateTime": "01/02/24 00:00:00", "value": {"date": "01/02/24", "value": 0.0, "error": 0}},
]


def test_steps_are_summed_per_day_in_either_timestamp_format() -> None:
    american = daily_steps(STEPS)
    iso = daily_steps(
        [
            {"dateTime": "2024-01-01 08:00:00", "value": "1200"},
            {"dateTime": "2024-01-01 18:30:00", "value": "800"},
        ]
    )

    assert american.loc["2024-01-01"] == 2000
    assert american.isna().loc["2024-01-02"]  # no steps at all: tracker not worn
    assert iso.loc["2024-01-01"] == 2000


def test_sleep_adds_naps_and_resting_heart_rate_ignores_placeholders() -> None:
    assert daily_sleep_hours(SLEEP).loc["2024-01-01"] == 7.5
    heart_rate = daily_resting_heart_rate(RESTING)
    assert heart_rate.loc["2024-01-01"] == pytest.approx(58.26)
    assert heart_rate.isna().loc["2024-01-02"]


def test_export_files_are_found_by_name_in_any_folder_and_merged() -> None:
    archive = export(
        {
            "Takeout/Fitbit/Global Export Data/steps-2024-01-01.json": STEPS[:2],
            "Takeout/Fitbit/Global Export Data/steps-2024-01-31.json": [
                {"dateTime": "02/01/24 10:00:00", "value": "500"}
            ],
            "MyFitbitData/User/Sleep/sleep-2024-01-01.json": SLEEP,
            "resting_heart_rate-2024-01-01.json": RESTING,
            "Takeout/Fitbit/Global Export Data/heart_rate-2024-01-01.json": [{"bpm": 70}],
            "Takeout/Fitbit/Global Export Data/sleep_score.csv": "ignored",
        }
    )

    result = parse_fitbit_export(archive, MB)

    assert [record.date for record in result.records] == [
        dt.date(2024, 1, 1),
        dt.date(2024, 2, 1),
    ]
    first = result.records[0]
    assert (first.steps, first.sleep_hours, first.resting_heart_rate) == (2000, 7.5, 58.3)
    assert result.records[1].steps == 500
    assert len(result.warnings) == 1


def test_missing_kinds_and_broken_files_are_reported() -> None:
    archive = export({"steps-1.json": STEPS, "sleep-1.json": "{not json", "steps-2.json": {}})

    result = parse_fitbit_export(archive, MB)

    assert result.records[0].steps == 2000
    assert "No files found for: sleep, resting heart rate." in result.warnings
    assert "Skipped 2 file(s)" in result.warnings[2]


def test_bundled_sample_export() -> None:
    result = parse_fitbit_export(FITBIT_ZIP.read_bytes(), 200 * MB)

    assert len(result.records) == 150
    first = result.records[0]
    assert first.date == dt.date(2026, 4, 1)
    assert first.sleep_hours == 5.9
    assert first.mood_score is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"not a zip", "is a zip file"),
        (export({"notes.txt": "hello"}), "No Fitbit data files"),
        (export({"steps-1.json": [{"dateTime": "garbage", "value": "x"}]}), "no usable days"),
    ],
)
def test_unusable_uploads_are_rejected(content: bytes, expected: str) -> None:
    with pytest.raises(IngestionError, match=expected):
        parse_fitbit_export(content, MB)


def test_exports_beyond_the_unzipped_limit_are_refused() -> None:
    many = [{"dateTime": "01/01/24 08:00:00", "value": "1"}] * 30_000
    archive = export({"steps-1.json": many, "steps-2.json": many})

    with pytest.raises(IngestionError, match="too large once unzipped"):
        parse_fitbit_export(archive, MB)


def test_fitbit_import_endpoint(client: TestClient, upload_slug: str) -> None:
    base = f"/api/v1/datasets/{upload_slug}"

    result = client.post(
        f"{base}/import/fitbit",
        files={"file": ("MyFitbitData.zip", FITBIT_ZIP.read_bytes(), "application/zip")},
    )

    assert result.status_code == 200
    assert result.json()["days_added"] == 150
    assert "daily check-in" in result.json()["warnings"][0]
    insights = client.get(f"{base}/insights").json()
    assert {item["key"] for item in insights["weekly"]} >= {"sleep_hours", "steps"}


def test_fitbit_endpoint_rejects_other_files_and_the_demo_dataset(
    client: TestClient, upload_slug: str
) -> None:
    wrong = client.post(
        f"/api/v1/datasets/{upload_slug}/import/fitbit",
        files={"file": ("export.xml", APPLE_HEALTH_XML.read_bytes(), "application/xml")},
    )
    demo = client.post(
        "/api/v1/datasets/demo/import/fitbit",
        files={"file": ("x.zip", FITBIT_ZIP.read_bytes(), "application/zip")},
    )

    assert wrong.status_code == 422
    assert "is a zip file" in wrong.json()["detail"]["message"]
    assert demo.status_code == 403
