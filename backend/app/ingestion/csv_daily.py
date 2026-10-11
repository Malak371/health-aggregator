"""Parser for the project's daily-metrics CSV (one row per day)."""

from __future__ import annotations

import io

import pandas as pd
from pydantic import ValidationError

from app.ingestion.errors import IngestionError
from app.ingestion.result import ParseResult
from app.schemas import METRIC_FIELDS, DailyRecord

MAX_REPORTED_ERRORS = 10


def _describe(error: ValidationError) -> str:
    parts = []
    for item in error.errors():
        field = ".".join(str(loc) for loc in item["loc"]) or "row"
        parts.append(f"{field}: {item['msg']}")
    return "; ".join(parts)


def parse_daily_csv(content: bytes) -> ParseResult:
    """Validate a daily-metrics CSV and return one record per date.

    Raises IngestionError with row-level messages if anything is invalid;
    a file is accepted in full or not at all.
    """
    try:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise IngestionError("The file could not be read as CSV.", [str(exc)]) from exc

    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "date" not in frame.columns:
        raise IngestionError(
            "The CSV needs a 'date' column.",
            [f"Columns found: {', '.join(frame.columns) or '(none)'}"],
        )
    known = [column for column in frame.columns if column in METRIC_FIELDS]
    if not known:
        raise IngestionError(
            "The CSV has no recognised metric columns.",
            [f"Expected at least one of: {', '.join(METRIC_FIELDS)}"],
        )
    if frame.empty:
        raise IngestionError("The CSV has a header but no data rows.")

    warnings = []
    ignored = [column for column in frame.columns if column not in (*METRIC_FIELDS, "date")]
    if ignored:
        warnings.append(f"Ignored unknown columns: {', '.join(ignored)}")

    records: list[DailyRecord] = []
    errors: list[str] = []
    seen: set[object] = set()
    for index, row in enumerate(frame.to_dict(orient="records")):
        line = index + 2  # header is line 1
        values = {"date": row["date"].strip()}
        values.update({name: row[name].strip() or None for name in known})
        try:
            record = DailyRecord.model_validate(values)
        except ValidationError as exc:
            errors.append(f"Line {line}: {_describe(exc)}")
            continue
        if record.date in seen:
            errors.append(f"Line {line}: duplicate date {record.date}")
            continue
        seen.add(record.date)
        records.append(record)

    if errors:
        shown = errors[:MAX_REPORTED_ERRORS]
        if len(errors) > len(shown):
            shown.append(f"...and {len(errors) - len(shown)} more")
        raise IngestionError(f"{len(errors)} row(s) failed validation.", shown)

    return ParseResult(records=sorted(records, key=lambda r: r.date), warnings=warnings)
