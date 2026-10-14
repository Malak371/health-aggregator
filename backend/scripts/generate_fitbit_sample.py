"""Write a small synthetic zip in the layout of a Fitbit account export.

It replays the first 150 days of the synthetic demo dataset as monthly
steps-, sleep-, resting_heart_rate- and exercise- JSON files, using the field names and
timestamp formats of a real export. No real person's data is involved.

Run from the repository root:

    python backend/scripts/generate_fitbit_sample.py
"""

from __future__ import annotations

import datetime as dt
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"
SOURCE_CSV = SAMPLES / "demo_daily_metrics.csv"
OUTPUT = SAMPLES / "fitbit_export_sample.zip"
FOLDER = "Fitbit/Global Export Data"
DAYS = 150
STAMP = "%m/%d/%y %H:%M:%S"


def build() -> dict[str, list[dict[str, Any]]]:
    rng = np.random.default_rng(11)
    frame = pd.read_csv(SOURCE_CSV, parse_dates=["date"]).head(DAYS)
    files: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in frame.itertuples():
        day: dt.date = row.date.date()
        month = day.replace(day=1).isoformat()
        midnight = dt.datetime.combine(day, dt.time())

        # Steps: a real export has one entry per active minute; twelve bursts are
        # enough to exercise the parser while keeping the sample small.
        shares = rng.dirichlet(np.ones(12))
        counts = np.floor(row.steps * shares).astype(int)
        counts[0] += row.steps - counts.sum()
        for hour, count in zip(range(8, 20), counts, strict=True):
            moment = midnight + dt.timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            files[f"steps-{month}.json"].append(
                {"dateTime": moment.strftime(STAMP), "value": str(int(count))}
            )

        if not pd.isna(row.resting_heart_rate):
            files[f"resting_heart_rate-{month}.json"].append(
                {
                    "dateTime": midnight.strftime(STAMP),
                    "value": {
                        "date": day.strftime("%m/%d/%y"),
                        "value": float(row.resting_heart_rate),
                        "error": 6.5,
                    },
                }
            )

        if not pd.isna(row.sleep_hours):
            asleep = round(row.sleep_hours * 60)
            awake = int(rng.integers(15, 50))
            end = midnight + dt.timedelta(hours=7, minutes=int(rng.integers(0, 45)))
            start = end - dt.timedelta(minutes=asleep + awake)
            files[f"sleep-{month}.json"].append(
                {
                    "logId": int(rng.integers(10**10, 10**11)),
                    "dateOfSleep": day.isoformat(),
                    "startTime": start.isoformat(timespec="milliseconds"),
                    "endTime": end.isoformat(timespec="milliseconds"),
                    "duration": (asleep + awake) * 60_000,
                    "minutesAsleep": asleep,
                    "minutesAwake": awake,
                    "timeInBed": asleep + awake,
                    "type": "stages",
                    "mainSleep": True,
                }
            )
        if row.training_minutes > 0:
            started = midnight + dt.timedelta(hours=17, minutes=30)
            files["exercise-0.json"].append(
                {
                    "logId": int(rng.integers(10**10, 10**11)),
                    "activityName": "Run",
                    "logType": "tracker",
                    "startTime": started.strftime(STAMP),
                    "originalStartTime": started.strftime(STAMP),
                    "duration": int(row.training_minutes * 60_000) + 90_000,
                    "activeDuration": int(row.training_minutes * 60_000),
                    "averageHeartRate": int(rng.integers(120, 165)),
                }
            )
        # The tracker also logs everyday walks by itself; they are not training.
        if rng.random() < 0.5:
            started = midnight + dt.timedelta(hours=12, minutes=int(rng.integers(0, 50)))
            files["exercise-0.json"].append(
                {
                    "logId": int(rng.integers(10**10, 10**11)),
                    "activityName": "Walk",
                    "logType": "auto_detected",
                    "startTime": started.strftime(STAMP),
                    "originalStartTime": started.strftime(STAMP),
                    "duration": 1_200_000,
                    "activeDuration": 1_200_000,
                }
            )
    return files


if __name__ == "__main__":
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, rows in sorted(build().items()):
            # A fixed timestamp keeps the committed file identical between runs.
            info = zipfile.ZipInfo(f"{FOLDER}/{name}", date_time=(2026, 9, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, json.dumps(rows, indent=1))
    print(f"Wrote {OUTPUT.name} ({OUTPUT.stat().st_size / 1024:.0f} KB)")
