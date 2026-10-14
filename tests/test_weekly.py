import pytest
from fastapi.testclient import TestClient

from app.analytics.weekly import weekly_changes
from tests.helpers import make_frame


def test_each_signal_is_averaged_over_the_last_two_weeks() -> None:
    frame = make_frame(
        sleep_hours=[8.0] * 7 + [6.5] * 7,
        resting_heart_rate=[58.0] * 7 + [61.0] * 7,
        training_minutes=[30.0] * 7 + [60.0] * 7,
        training_rpe=[5.0] * 14,
    )

    changes = {change.key: change for change in weekly_changes(frame)}

    sleep = changes["sleep_hours"]
    assert (sleep.last_week, sleep.this_week, sleep.change) == (8.0, 6.5, -1.5)
    assert sleep.higher_is_better is True
    assert changes["resting_heart_rate"].change == 3.0
    assert changes["resting_heart_rate"].higher_is_better is False
    load = changes["training_load"]
    assert (load.last_week, load.this_week) == (150, 300)
    assert load.higher_is_better is None


def test_only_the_last_fourteen_days_count() -> None:
    frame = make_frame(sleep_hours=[3.0] * 30 + [7.0] * 7 + [8.0] * 7)

    sleep = next(c for c in weekly_changes(frame) if c.key == "sleep_hours")

    assert (sleep.last_week, sleep.this_week) == (7.0, 8.0)


def test_signals_without_enough_days_in_both_weeks_are_left_out() -> None:
    nan = float("nan")
    frame = make_frame(
        sleep_hours=[7.0] * 14,
        mood_score=[nan] * 7 + [4.0] * 7,  # nothing last week
        study_hours=[3.0] * 7 + [nan] * 5 + [4.0, 4.0],  # two days this week
    )

    keys = {change.key for change in weekly_changes(frame)}

    assert "sleep_hours" in keys
    assert "mood_score" not in keys
    assert "study_hours" not in keys
    assert "steps" not in keys


def test_less_than_two_weeks_of_data_gives_no_summary() -> None:
    assert weekly_changes(make_frame(sleep_hours=[7.0] * 9)) == []


def test_weekly_summary_is_part_of_the_insights(client: TestClient) -> None:
    weekly = client.get("/api/v1/datasets/demo/insights").json()["weekly"]

    assert [item["key"] for item in weekly][:2] == ["readiness", "sleep_hours"]
    assert len(weekly) == 7
    for item in weekly:
        assert item["change"] == pytest.approx(item["this_week"] - item["last_week"], abs=1.01)


def test_weekly_readiness_agrees_with_the_recovery_trend(client: TestClient) -> None:
    insights = client.get("/api/v1/datasets/demo/insights").json()
    readiness = next(item for item in insights["weekly"] if item["key"] == "readiness")

    assert readiness["change"] == insights["recovery"]["change"]
    assert readiness["this_week"] == insights["recovery"]["last_7_days"]
