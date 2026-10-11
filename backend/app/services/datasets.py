"""Dataset storage: creation, record upserts, demo seeding and clean-up."""

from __future__ import annotations

import datetime as dt
import secrets
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.config import DEMO_CSV
from app.ingestion.csv_daily import parse_daily_csv
from app.models import DEMO_SLUG, DailyMetric, Dataset
from app.schemas import METRIC_FIELDS, DailyRecord, DatasetOut


def get_dataset(session: Session, slug: str) -> Dataset | None:
    return session.scalar(select(Dataset).where(Dataset.slug == slug))


def describe(session: Session, dataset: Dataset) -> DatasetOut:
    days, first, last = session.execute(
        select(func.count(), func.min(DailyMetric.date), func.max(DailyMetric.date)).where(
            DailyMetric.dataset_id == dataset.id
        )
    ).one()
    return DatasetOut(
        slug=dataset.slug,
        name=dataset.name,
        source=dataset.source,
        created_at=dataset.created_at,
        days=days,
        first_date=first,
        last_date=last,
    )


def count_uploads(session: Session) -> int:
    return session.scalar(select(func.count()).where(Dataset.slug != DEMO_SLUG)) or 0


def create_upload_dataset(session: Session, name: str) -> Dataset:
    """Create an empty dataset addressed by an unguessable slug.

    There are no accounts: knowing the slug is what grants access, so it is
    generated with a CSPRNG and never listed by the API.
    """
    dataset = Dataset(slug=secrets.token_urlsafe(16), name=name, source="upload")
    session.add(dataset)
    session.commit()
    return dataset


def upsert_records(
    session: Session, dataset: Dataset, records: Sequence[DailyRecord]
) -> tuple[int, int]:
    """Insert new days and merge into existing ones. Returns (added, updated).

    Merging keeps values the new record does not carry, so a mood log can be
    layered on top of a wearable export for the same dates.
    """
    existing = {
        row.date: row
        for row in session.scalars(
            select(DailyMetric).where(
                DailyMetric.dataset_id == dataset.id,
                DailyMetric.date.in_([record.date for record in records]),
            )
        )
    }
    added = updated = 0
    for record in records:
        values = record.model_dump(include=set(METRIC_FIELDS), exclude_none=True)
        row = existing.get(record.date)
        if row is None:
            session.add(DailyMetric(dataset_id=dataset.id, date=record.date, **values))
            added += 1
        else:
            for name, value in values.items():
                setattr(row, name, value)
            updated += 1
    session.commit()
    return added, updated


def delete_dataset(session: Session, dataset: Dataset) -> None:
    session.delete(dataset)
    session.commit()


def purge_expired_uploads(session: Session, ttl_hours: int) -> int:
    """Delete visitor datasets older than the retention window."""
    cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(hours=ttl_hours)
    expired = session.scalars(
        select(Dataset).where(Dataset.slug != DEMO_SLUG, Dataset.created_at < cutoff)
    ).all()
    for dataset in expired:
        session.delete(dataset)
    session.commit()
    return len(expired)


def seed_demo(session: Session) -> Dataset:
    """(Re)load the bundled synthetic dataset so it always matches the repo."""
    records = parse_daily_csv(DEMO_CSV.read_bytes()).records
    dataset = get_dataset(session, DEMO_SLUG)
    if dataset is None:
        dataset = Dataset(slug=DEMO_SLUG, name="Demo (synthetic)", source="synthetic")
        session.add(dataset)
        session.flush()
    else:
        session.execute(delete(DailyMetric).where(DailyMetric.dataset_id == dataset.id))
    session.add_all(DailyMetric(dataset_id=dataset.id, **record.model_dump()) for record in records)
    session.commit()
    return dataset
