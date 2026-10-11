import numpy as np
import pandas as pd
import pytest

from app.analytics.correlations import correlate, correlation_insights, strength_of
from app.analytics.load import load_metrics
from tests.helpers import make_frame


def test_perfect_positive_and_negative_correlation() -> None:
    sleep = np.linspace(5, 9, 20)
    frame = make_frame(sleep_hours=sleep, mood_score=sleep / 2, sugar_grams=100 - sleep)

    up = correlate(frame, "sleep_hours", "mood_score")
    down = correlate(frame, "sleep_hours", "sugar_grams")

    assert up is not None and down is not None
    assert up.r == pytest.approx(1.0)
    assert up.strength == "strong"
    assert "higher mood" in up.insight
    assert down.r == pytest.approx(-1.0)
    assert "lower sugar intake" in down.insight


def test_lag_pairs_x_with_a_later_y() -> None:
    rng = np.random.default_rng(0)
    load = rng.uniform(0, 500, 40)
    # Heart rate follows yesterday's load exactly.
    heart_rate = 55 + np.roll(load, 1) / 50
    frame = make_frame(sleep_hours=load, resting_heart_rate=heart_rate)

    same_day = correlate(frame, "sleep_hours", "resting_heart_rate")
    next_day = correlate(frame, "sleep_hours", "resting_heart_rate", lag_days=1)

    assert next_day is not None and same_day is not None
    assert next_day.r == pytest.approx(1.0)
    assert next_day.n == 39
    assert "the next day" in next_day.insight
    assert abs(same_day.r) < 0.5


def test_missing_values_are_dropped_pairwise() -> None:
    sleep = list(np.linspace(5, 9, 20))
    mood = [value / 2 for value in sleep]
    mood[3] = float("nan")
    result = correlate(make_frame(sleep_hours=sleep, mood_score=mood), "sleep_hours", "mood_score")

    assert result is not None
    assert result.n == 19


@pytest.mark.parametrize(
    "frame",
    [
        make_frame(sleep_hours=[7.0] * 5, mood_score=[3.0, 4, 3, 4, 3]),  # too few days
        make_frame(sleep_hours=[7.0] * 20, mood_score=np.linspace(1, 5, 20)),  # constant x
        make_frame(sleep_hours=np.linspace(5, 9, 20)),  # y never recorded
    ],
)
def test_returns_none_when_a_correlation_is_not_meaningful(frame: pd.DataFrame) -> None:
    assert correlate(frame, "sleep_hours", "mood_score") is None


@pytest.mark.parametrize(
    ("r", "expected"),
    [(0.1, "negligible"), (-0.25, "weak"), (0.45, "moderate"), (-0.8, "strong")],
)
def test_strength_labels(r: float, expected: str) -> None:
    assert strength_of(r) == expected


def test_negligible_correlations_do_not_claim_a_direction() -> None:
    rng = np.random.default_rng(1)
    frame = make_frame(sleep_hours=rng.normal(7, 1, 200), steps=rng.normal(8000, 1000, 200))

    result = correlate(frame, "sleep_hours", "steps")

    assert result is not None
    assert result.insight.startswith("No clear link")


def test_demo_data_reproduces_the_relationships_it_was_generated_with(
    demo_frame: pd.DataFrame,
) -> None:
    insights = {
        (c.x, c.y): c for c in correlation_insights(demo_frame.join(load_metrics(demo_frame)))
    }

    assert insights[("sleep_hours", "mood_score")].r > 0.4
    assert insights[("sleep_hours", "sugar_grams")].r < -0.4
    assert insights[("training_load", "resting_heart_rate")].lag_days == 1
