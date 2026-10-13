"""SQLAlchemy models: a dataset owns one row of metrics per calendar day."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

DEMO_SLUG = "demo"


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Public identifier. "demo" for the bundled data, a random token for uploads.
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    source: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    # Last read or write. Visitor datasets expire after a period of inactivity.
    last_active_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    metrics: Mapped[list[DailyMetric]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", order_by="DailyMetric.date"
    )

    @property
    def is_demo(self) -> bool:
        return self.slug == DEMO_SLUG


class DailyMetric(Base):
    """One day of signals. Every signal is optional because sources differ."""

    __tablename__ = "daily_metrics"
    __table_args__ = (UniqueConstraint("dataset_id", "date", name="uq_daily_metrics_dataset_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), index=True
    )
    date: Mapped[dt.date] = mapped_column(Date)

    sleep_hours: Mapped[float | None] = mapped_column(Float)
    resting_heart_rate: Mapped[float | None] = mapped_column(Float)
    steps: Mapped[int | None] = mapped_column(Integer)
    mood_score: Mapped[float | None] = mapped_column(Float)
    sugar_grams: Mapped[float | None] = mapped_column(Float)
    training_minutes: Mapped[float | None] = mapped_column(Float)
    training_rpe: Mapped[float | None] = mapped_column(Float)
    study_hours: Mapped[float | None] = mapped_column(Float)

    dataset: Mapped[Dataset] = relationship(back_populates="metrics")
