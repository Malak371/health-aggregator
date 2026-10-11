import numpy as np
import pandas as pd
import pytest

from app.analytics.load import (
    DEFAULT_RPE,
    acute_chronic,
    acwr_zone,
    daily_training_load,
    load_metrics,
)
from tests.helpers import make_frame


def test_training_load_is_minutes_times_rpe() -> None:
    frame = make_frame(training_minutes=[60, 30, 0], training_rpe=[7, 4, float("nan")])

    assert daily_training_load(frame).tolist() == [420, 120, 0]


def test_days_without_a_workout_count_as_rest() -> None:
    frame = make_frame(training_minutes=[60, float("nan")], training_rpe=[5, float("nan")])

    assert daily_training_load(frame).tolist() == [300, 0]


def test_missing_rpe_falls_back_to_a_moderate_default() -> None:
    frame = make_frame(training_minutes=[40.0], training_rpe=[float("nan")])

    assert daily_training_load(frame).iloc[0] == 40 * DEFAULT_RPE


def test_training_load_is_unknown_when_no_training_was_ever_logged() -> None:
    assert daily_training_load(make_frame(sleep_hours=[7.0, 8.0])).isna().all()


def test_constant_load_gives_a_ratio_of_one() -> None:
    daily = pd.Series(300.0, index=pd.date_range("2026-01-01", periods=40, freq="D"))

    rolling = acute_chronic(daily)

    assert rolling["ratio"].iloc[-1] == pytest.approx(1.0)
    assert rolling["acute"].iloc[:6].isna().all()  # needs a full week
    assert rolling["chronic"].iloc[:20].isna().all()  # needs three weeks
    assert rolling["chronic"].iloc[20] == pytest.approx(300)


def test_doubling_load_for_a_week_raises_the_ratio() -> None:
    daily = pd.Series(100.0, index=pd.date_range("2026-01-01", periods=35, freq="D"))
    daily.iloc[-7:] = 200.0

    last = acute_chronic(daily).iloc[-1]

    assert last["acute"] == pytest.approx(200)
    assert last["chronic"] == pytest.approx(125)  # (21 * 100 + 7 * 200) / 28
    assert last["ratio"] == pytest.approx(1.6)


def test_ratio_is_undefined_when_chronic_load_is_zero() -> None:
    daily = pd.Series(0.0, index=pd.date_range("2026-01-01", periods=30, freq="D"))

    assert np.isnan(acute_chronic(daily)["ratio"].iloc[-1])


def test_study_load_tolerates_a_few_unlogged_days() -> None:
    study = [4.0, float("nan"), 4, 4, float("nan"), 4, float("nan")]

    metrics = load_metrics(make_frame(study_hours=study))

    assert metrics["study_acute"].iloc[-1] == pytest.approx(4.0)
    assert metrics["training_load"].isna().all()


@pytest.mark.parametrize(
    ("ratio", "zone"),
    [
        (0.5, "undertraining"),
        (0.8, "optimal"),
        (1.29, "optimal"),
        (1.3, "caution"),
        (1.5, "high"),
        (2.4, "high"),
        (float("nan"), None),
        (None, None),
    ],
)
def test_acwr_zones(ratio: float | None, zone: str | None) -> None:
    assert acwr_zone(ratio) == zone
