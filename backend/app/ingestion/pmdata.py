"""Reader for one participant folder of the PMData dataset.

PMData (Thambawita et al., MMSys 2020, CC BY 4.0, https://datasets.simula.no/pmdata)
combines a Fitbit Versa 2 with daily self-reports. This module maps one
participant onto the project's daily schema:

- sleep_hours: Fitbit minutes asleep, summed per date of sleep; the self-reported
  whole hours are used only for nights the watch did not record
- resting_heart_rate: Fitbit daily value
- steps: Fitbit per-minute counts, summed per day
- mood_score: self-reported, 1-5
- training_minutes / training_rpe: logged sessions, summed per day, with exertion
  averaged by duration so that minutes x RPE equals the day's session-RPE load

It also returns the participant's self-reported readiness (0-10), which is not
part of the schema but serves as an independently measured target for the model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.schemas import METRIC_FIELDS


@dataclass(frozen=True)
class PMDataParticipant:
    name: str
    frame: pd.DataFrame  # daily metrics, indexed by date
    reported_readiness: pd.Series  # 0-10, indexed like frame


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, format="ISO8601").dt.tz_localize(None).dt.normalize()


def _wellness(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table["date"] = _dates(table["effective_time_frame"])
    # A few days have two reports; keep the last one submitted.
    return table.sort_values("effective_time_frame").groupby("date").last()


def _training(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table = table[(table["duration_min"] > 0) & table["perceived_exertion"].between(1, 10)]
    table["date"] = _dates(table["end_date_time"])
    table["load"] = table["duration_min"] * table["perceived_exertion"]
    daily = table.groupby("date")[["duration_min", "load"]].sum()
    return pd.DataFrame(
        {
            "training_minutes": daily["duration_min"].clip(upper=1440),
            "training_rpe": daily["load"] / daily["duration_min"],
        }
    )


def _fitbit_sleep(path: Path) -> pd.Series:
    nights = pd.DataFrame(json.loads(path.read_text()))
    hours = nights.groupby(pd.to_datetime(nights["dateOfSleep"]))["minutesAsleep"].sum() / 60
    return hours.clip(upper=24)


def _fitbit_resting_hr(path: Path) -> pd.Series:
    rows = json.loads(path.read_text())
    series = pd.Series(
        {pd.Timestamp(row["dateTime"]).normalize(): row["value"]["value"] for row in rows},
        dtype="float64",
    )
    # The export writes 0 for days without a reading.
    return series.where(series.between(25, 220))


def _fitbit_steps(path: Path) -> pd.Series:
    minutes = pd.DataFrame(json.loads(path.read_text()))
    counts = pd.to_numeric(minutes["value"], errors="coerce")
    daily = counts.groupby(pd.to_datetime(minutes["dateTime"]).dt.normalize()).sum()
    # A day with no steps at all means the watch was not worn.
    return daily.where(daily > 0).clip(upper=200_000)


def load_participant(folder: Path) -> PMDataParticipant:
    """Read `<folder>/pmsys` and `<folder>/fitbit` into a daily frame."""
    wellness = _wellness(folder / "pmsys" / "wellness.csv")
    columns: dict[str, pd.Series] = {
        "mood_score": wellness["mood"].where(lambda m: m.between(1, 5))
    }

    training_file = folder / "pmsys" / "srpe.csv"
    if training_file.exists():
        training = _training(training_file)
        columns["training_minutes"] = training["training_minutes"]
        columns["training_rpe"] = training["training_rpe"]

    fitbit = folder / "fitbit"
    reported_sleep = wellness["sleep_duration_h"].where(lambda h: h.between(0, 24))
    if (fitbit / "sleep.json").exists():
        columns["sleep_hours"] = _fitbit_sleep(fitbit / "sleep.json").combine_first(reported_sleep)
    else:
        columns["sleep_hours"] = reported_sleep
    if (fitbit / "resting_heart_rate.json").exists():
        columns["resting_heart_rate"] = _fitbit_resting_hr(fitbit / "resting_heart_rate.json")
    if (fitbit / "steps.json").exists():
        columns["steps"] = _fitbit_steps(fitbit / "steps.json")

    frame = pd.DataFrame(columns).reindex(columns=list(METRIC_FIELDS)).astype("float64")
    frame = frame.dropna(how="all").sort_index()
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="D", name="date"))
    readiness = wellness["readiness"].where(lambda r: r.between(0, 10)).astype("float64")
    return PMDataParticipant(
        name=folder.name,
        frame=frame,
        reported_readiness=readiness.reindex(frame.index).rename("reported_readiness"),
    )
