import json
from pathlib import Path

import pandas as pd
import pytest

from app.ingestion.pmdata import load_participant
from app.ml.model import train_and_explain


@pytest.fixture
def participant(tmp_path: Path) -> Path:
    """A three-day participant folder in PMData's layout."""
    folder = tmp_path / "p99"
    (folder / "pmsys").mkdir(parents=True)
    (folder / "fitbit").mkdir()
    (folder / "pmsys" / "wellness.csv").write_text(
        "effective_time_frame,fatigue,mood,readiness,sleep_duration_h,sleep_quality,"
        "soreness,soreness_area,stress\n"
        "2020-01-01T07:30:00.000Z,3,4,7,8,3,2,[],3\n"
        "2020-01-02T07:10:00.000Z,3,2,3,5,3,2,[],3\n"
        "2020-01-02T09:00:00.000Z,3,3,4,5,3,2,[],3\n"  # corrected report, same day
        "2020-01-03T08:00:00.000Z,3,5,9,9,3,2,[],3\n"
    )
    (folder / "pmsys" / "srpe.csv").write_text(
        "end_date_time,activity_names,perceived_exertion,duration_min\n"
        "2020-01-01T18:00:00.000Z,\"['running']\",8,30\n"
        "2020-01-01T20:00:00.000Z,\"['strength']\",4,60\n"
        "2020-01-03T18:00:00.000Z,\"['running']\",0,45\n"  # no exertion rating
    )
    (folder / "fitbit" / "sleep.json").write_text(
        json.dumps(
            [
                {"dateOfSleep": "2020-01-01", "minutesAsleep": 420},
                {"dateOfSleep": "2020-01-01", "minutesAsleep": 30},  # nap
                {"dateOfSleep": "2020-01-03", "minutesAsleep": 480},
            ]
        )
    )
    (folder / "fitbit" / "resting_heart_rate.json").write_text(
        json.dumps(
            [
                {"dateTime": "2020-01-01 00:00:00", "value": {"value": 55.4, "error": 6.0}},
                {"dateTime": "2020-01-02 00:00:00", "value": {"value": 0.0, "error": 0.0}},
            ]
        )
    )
    (folder / "fitbit" / "steps.json").write_text(
        json.dumps(
            [
                {"dateTime": "2020-01-01 08:00:00", "value": "120"},
                {"dateTime": "2020-01-01 08:01:00", "value": "80"},
                {"dateTime": "2020-01-02 08:00:00", "value": "0"},
            ]
        )
    )
    return folder


def test_participant_is_mapped_onto_the_daily_schema(participant: Path) -> None:
    person = load_participant(participant)
    frame = person.frame

    assert person.name == "p99"
    assert [day.day for day in frame.index] == [1, 2, 3]
    first = frame.iloc[0]
    assert first["sleep_hours"] == 7.5  # main sleep plus nap, from the watch
    assert first["resting_heart_rate"] == 55.4
    assert first["steps"] == 200
    assert first["mood_score"] == 4
    assert first["training_minutes"] == 90
    assert first["training_rpe"] == pytest.approx((30 * 8 + 60 * 4) / 90)


def test_gaps_and_placeholder_values_become_missing(participant: Path) -> None:
    frame = load_participant(participant).frame

    assert frame.loc["2020-01-02", "sleep_hours"] == 5  # watch gap: self-report is used
    assert pd.isna(frame.loc["2020-01-02", "resting_heart_rate"])  # exported as 0
    assert pd.isna(frame.loc["2020-01-02", "steps"])  # watch not worn
    assert pd.isna(frame.loc["2020-01-03", "training_minutes"])  # session without a rating
    assert frame["sugar_grams"].isna().all()


def test_the_last_report_of_a_day_wins_and_readiness_is_returned(participant: Path) -> None:
    person = load_participant(participant)

    assert person.frame.loc["2020-01-02", "mood_score"] == 3
    assert person.reported_readiness.tolist() == [7, 4, 9]


def test_only_the_wellness_log_is_required(participant: Path) -> None:
    for name in ("sleep.json", "resting_heart_rate.json", "steps.json"):
        (participant / "fitbit" / name).unlink()
    (participant / "pmsys" / "srpe.csv").unlink()

    frame = load_participant(participant).frame

    assert frame["sleep_hours"].tolist() == [8, 5, 9]
    assert frame["steps"].isna().all()


def test_model_can_be_scored_against_an_external_target(demo_frame: pd.DataFrame) -> None:
    # An independent score that simply lags sleep: the model should find it.
    external = (demo_frame["sleep_hours"].shift(1) * 10).rename("external")

    report = train_and_explain(
        demo_frame, score=external, extra_features=pd.DataFrame({"External today": external})
    )

    assert report.evaluation.roc_auc > 0.6
    assert "External today" in {effect.label for effect in report.feature_effects}
    assert report.evaluation.baseline_persistence_roc_auc is not None
