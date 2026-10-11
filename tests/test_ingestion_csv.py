import datetime as dt

import pytest

from app.ingestion.csv_daily import parse_daily_csv
from app.ingestion.errors import IngestionError
from tests.conftest import DEMO_CSV


def test_parses_the_bundled_sample() -> None:
    result = parse_daily_csv(DEMO_CSV.read_bytes())

    assert len(result.records) == 180
    assert result.warnings == []
    first = result.records[0]
    assert first.date == dt.date(2026, 4, 1)
    assert first.sleep_hours == 5.9


def test_blank_cells_become_missing_values() -> None:
    result = parse_daily_csv(b"date,sleep_hours,mood_score\n2026-01-01,7.5,\n")

    assert result.records[0].sleep_hours == 7.5
    assert result.records[0].mood_score is None


def test_headers_are_case_insensitive_and_rows_are_sorted() -> None:
    result = parse_daily_csv(b" Date , Sleep_Hours\n2026-01-02,7\n2026-01-01,8\n")

    assert [r.date.day for r in result.records] == [1, 2]


def test_unknown_columns_are_ignored_with_a_warning() -> None:
    result = parse_daily_csv(b"date,steps,shoe_size\n2026-01-01,9000,42\n")

    assert result.records[0].steps == 9000
    assert "shoe_size" in result.warnings[0]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"sleep_hours\n7\n", "needs a 'date' column"),
        (b"date,shoe_size\n2026-01-01,42\n", "no recognised metric columns"),
        (b"date,sleep_hours\n", "no data rows"),
        (b"", "could not be read"),
    ],
)
def test_unusable_files_are_rejected(content: bytes, expected: str) -> None:
    with pytest.raises(IngestionError, match=expected):
        parse_daily_csv(content)


def test_invalid_rows_are_reported_with_line_numbers() -> None:
    content = (
        b"date,sleep_hours,mood_score\n"
        b"2026-01-01,7,3\n"
        b"2026-01-02,30,3\n"  # more than 24 hours of sleep
        b"not-a-date,7,3\n"
        b"2026-01-01,6,3\n"  # duplicate date
        b"2026-01-05,,\n"  # no metrics at all
    )

    with pytest.raises(IngestionError) as excinfo:
        parse_daily_csv(content)

    errors = excinfo.value.errors
    assert excinfo.value.message == "4 row(s) failed validation."
    assert errors[0].startswith("Line 3: sleep_hours")
    assert errors[1].startswith("Line 4: date")
    assert errors[2] == "Line 5: duplicate date 2026-01-01"
    assert "at least one metric" in errors[3]


def test_long_error_lists_are_truncated() -> None:
    rows = "".join(f"2026-01-{day:02d},99\n" for day in range(1, 21))

    with pytest.raises(IngestionError) as excinfo:
        parse_daily_csv(("date,sleep_hours\n" + rows).encode())

    assert len(excinfo.value.errors) == 11
    assert excinfo.value.errors[-1] == "...and 10 more"
