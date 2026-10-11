"""Write a small synthetic file in Apple Health's export.xml format.

It replays the first 90 days of the synthetic demo dataset as <Record> and
<Workout> elements, including the quirks a real export has (a DTD header, the
same steps counted by both a phone and a watch, sleep split into stages, time
in bed that is not sleep). No real person's data is involved.

Run from the repository root:

    python backend/scripts/generate_apple_health_sample.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
import pandas as pd

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
SOURCE_CSV = SAMPLES / "demo_daily_metrics.csv"
OUTPUT = SAMPLES / "apple_health_export_sample.xml"
DAYS = 90
TZ = dt.timezone(dt.timedelta(hours=-5))

HEADER = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE HealthData [
<!ELEMENT HealthData (ExportDate,Me,(Record|Workout)*)>
<!ATTLIST HealthData locale CDATA #REQUIRED>
<!ELEMENT ExportDate EMPTY>
<!ATTLIST ExportDate value CDATA #REQUIRED>
<!ELEMENT Me EMPTY>
<!ELEMENT Record EMPTY>
<!ELEMENT Workout EMPTY>
]>
<HealthData locale="en_US">
 <ExportDate value="2026-07-01 09:00:00 -0500"/>
 <Me HKCharacteristicTypeIdentifierDateOfBirth=""/>
"""


def _stamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S %z")


def _element(tag: str, **attributes: object) -> str:
    rendered = " ".join(f"{name}={quoteattr(str(value))}" for name, value in attributes.items())
    return f" <{tag} {rendered}/>\n"


def _record(kind: str, source: str, start: dt.datetime, end: dt.datetime, **extra: object) -> str:
    return _element(
        "Record",
        type=kind,
        sourceName=source,
        **extra,
        creationDate=_stamp(end),
        startDate=_stamp(start),
        endDate=_stamp(end),
    )


def build() -> str:
    rng = np.random.default_rng(7)
    frame = pd.read_csv(SOURCE_CSV, parse_dates=["date"]).head(DAYS)
    parts = [HEADER]

    for row in frame.itertuples():
        midnight = dt.datetime.combine(row.date.date(), dt.time(), tzinfo=TZ)

        # Steps: the watch sees everything, the phone misses some of it.
        shares = rng.dirichlet(np.ones(3))
        for hour, share in zip((8, 13, 18), shares, strict=True):
            start = midnight + dt.timedelta(hours=hour)
            end = start + dt.timedelta(minutes=50)
            watch = round(row.steps * share)
            for source, value in (("Apple Watch", watch), ("iPhone", round(watch * 0.7))):
                parts.append(
                    _record(
                        "HKQuantityTypeIdentifierStepCount",
                        source,
                        start,
                        end,
                        unit="count",
                        value=value,
                    )
                )

        if not pd.isna(row.resting_heart_rate):
            moment = midnight + dt.timedelta(hours=7)
            parts.append(
                _record(
                    "HKQuantityTypeIdentifierRestingHeartRate",
                    "Apple Watch",
                    moment,
                    moment,
                    unit="count/min",
                    value=row.resting_heart_rate,
                )
            )

        # Sleep: in bed from the previous evening, three stages, one awakening.
        if not pd.isna(row.sleep_hours):
            wake = midnight + dt.timedelta(hours=6.5 + float(rng.uniform(0, 1)))
            asleep_minutes = round(row.sleep_hours * 60)
            awake_minutes = 12
            bedtime = wake - dt.timedelta(minutes=asleep_minutes + awake_minutes + 15)
            prefix = "HKCategoryValueSleepAnalysis"
            parts.append(
                _record(
                    "HKCategoryTypeIdentifierSleepAnalysis",
                    "iPhone",
                    bedtime,
                    wake,
                    value=prefix + "InBed",
                )
            )
            cursor = bedtime + dt.timedelta(minutes=15)
            stages = (
                ("AsleepCore", round(asleep_minutes * 0.55)),
                ("Awake", awake_minutes),
                ("AsleepDeep", round(asleep_minutes * 0.2)),
            )
            for stage, minutes in stages:
                end = cursor + dt.timedelta(minutes=minutes)
                parts.append(
                    _record(
                        "HKCategoryTypeIdentifierSleepAnalysis",
                        "Apple Watch",
                        cursor,
                        end,
                        value=prefix + stage,
                    )
                )
                cursor = end
            parts.append(
                _record(
                    "HKCategoryTypeIdentifierSleepAnalysis",
                    "Apple Watch",
                    cursor,
                    wake,
                    value=prefix + "AsleepREM",
                )
            )

        if row.training_minutes > 0:
            start = midnight + dt.timedelta(hours=17, minutes=30)
            end = start + dt.timedelta(minutes=row.training_minutes)
            parts.append(
                _element(
                    "Workout",
                    workoutActivityType="HKWorkoutActivityTypeRunning",
                    duration=row.training_minutes,
                    durationUnit="min",
                    sourceName="Apple Watch",
                    creationDate=_stamp(end),
                    startDate=_stamp(start),
                    endDate=_stamp(end),
                )
            )

    parts.append("</HealthData>\n")
    return "".join(parts)


if __name__ == "__main__":
    OUTPUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
