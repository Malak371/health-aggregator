import pandas as pd
import pytest

from app.analytics.burnout import WEIGHTS, burnout_components, burnout_risk, risk_level
from app.analytics.recommendations import Snapshot, generate_recommendations
from app.analytics.recovery import readiness_score, recovery_trend
from tests.helpers import make_frame


def test_readiness_is_high_for_a_rested_day_and_low_for_a_rough_one() -> None:
    frame = make_frame(
        sleep_hours=[7.0] * 10 + [8.5, 4.0],
        resting_heart_rate=[60.0] * 10 + [55, 66],
        mood_score=[3.0] * 10 + [5, 1],
    )

    readiness = readiness_score(frame)

    assert readiness.iloc[-2] == pytest.approx(100)
    assert readiness.iloc[-1] < 10


def test_readiness_reweights_over_the_signals_that_exist() -> None:
    # Sleep only: 6 h is halfway between 4 h (0) and 8 h (1).
    readiness = readiness_score(make_frame(sleep_hours=[6.0, 8.0, 3.0]))

    assert readiness.tolist() == [50, 100, 0]


def test_readiness_is_missing_when_nothing_was_recorded() -> None:
    assert readiness_score(make_frame(steps=[9000, 8000])).isna().all()


def test_recovery_trend_compares_the_last_two_weeks() -> None:
    index = pd.date_range("2026-01-01", periods=14, freq="D")

    declining = recovery_trend(pd.Series([80.0] * 7 + [60.0] * 7, index=index))
    stable = recovery_trend(pd.Series([70.0] * 7 + [71.0] * 7, index=index))
    short = recovery_trend(pd.Series([70.0] * 9, index=index[:9]))

    assert (declining.direction, declining.change) == ("declining", -20.0)
    assert stable.direction == "stable"
    assert short.direction == "unknown"


def test_burnout_risk_is_low_for_a_calm_month_and_high_for_a_crunch() -> None:
    calm = make_frame(
        sleep_hours=[8.0] * 42,
        resting_heart_rate=[58.0] * 42,
        mood_score=[4.0] * 42,
        study_hours=[3.0] * 42,
        training_minutes=[45.0] * 42,
        training_rpe=[5.0] * 42,
    )
    crunch = calm.copy()
    crunch.iloc[-10:] = [5.5, 66.0, 5000, 1.5, 80, 110, 8, 10]

    assert burnout_risk(burnout_components(calm)).iloc[-1] == pytest.approx(0)
    risk = burnout_risk(burnout_components(crunch)).iloc[-1]
    assert risk > 66
    assert risk_level(risk) == "high"


def test_burnout_components_stay_between_zero_and_one(demo_frame: pd.DataFrame) -> None:
    components = burnout_components(demo_frame)

    assert set(components.columns) == set(WEIGHTS)
    assert components.min().min() >= 0
    assert components.max().max() <= 1
    assert burnout_risk(components).dropna().between(0, 100).all()


def test_burnout_risk_needs_at_least_two_components() -> None:
    sleep_only = make_frame(sleep_hours=[6.0] * 14)

    assert burnout_risk(burnout_components(sleep_only)).isna().all()


@pytest.mark.parametrize(
    ("score", "level"),
    [(10, "low"), (33, "moderate"), (65.9, "moderate"), (66, "high"), (None, None)],
)
def test_risk_levels(score: float | None, level: str | None) -> None:
    assert risk_level(score) == level


def test_recommendations_explain_what_triggered_them() -> None:
    found = generate_recommendations(
        Snapshot(avg_sleep_7d=5.8, acwr=1.62, burnout_score=71, recent_high_hr_days=2)
    )

    reasons = " ".join(item.reason for item in found)
    assert len(found) == 4
    assert "71/100" in reasons
    assert "5.8 h" in reasons
    assert "1.62" in reasons


def test_recommendations_fall_back_to_a_stable_message() -> None:
    found = generate_recommendations(Snapshot(avg_sleep_7d=7.6, acwr=1.0, burnout_score=12))

    assert len(found) == 1
    assert "stable" in found[0].message


def test_recommendations_do_not_call_a_few_days_of_data_stable() -> None:
    found = generate_recommendations(Snapshot(days_logged=3))

    assert "Keep logging" in found[0].message
    assert "3 day(s)" in found[0].reason
