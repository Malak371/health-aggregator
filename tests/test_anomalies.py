import datetime as dt

import pandas as pd

from app.analytics.anomalies import detect_high_resting_hr, detect_load_spikes, detect_poor_sleep
from tests.helpers import make_frame


def test_poor_sleep_flags_only_nights_under_the_threshold() -> None:
    frame = make_frame(sleep_hours=[7.5, 4.9, 5.0, float("nan"), 3.0])

    found = detect_poor_sleep(frame)

    assert [a.date for a in found] == [dt.date(2026, 1, 2), dt.date(2026, 1, 5)]
    assert found[0].kind == "poor_sleep"
    assert found[0].value == 4.9


def test_high_resting_hr_flags_a_spike_against_the_previous_week() -> None:
    heart_rate = [60.0, 61, 59, 60, 61, 59, 60, 72, 60, 61]
    found = detect_high_resting_hr(make_frame(resting_heart_rate=heart_rate))

    assert len(found) == 1
    assert found[0].date == dt.date(2026, 1, 8)
    assert found[0].value == 72
    assert found[0].baseline == 60.0
    assert "72 bpm" in found[0].message


def test_high_resting_hr_needs_a_full_window_of_history() -> None:
    # The spike is on day 5, before seven days of baseline exist.
    found = detect_high_resting_hr(make_frame(resting_heart_rate=[60.0, 61, 59, 60, 80, 60]))

    assert found == []


def test_small_wobbles_in_a_very_steady_series_are_not_flagged() -> None:
    # 61 is far outside the standard deviation of a flat series but only 1 bpm up.
    heart_rate = [60.0, 60.1, 59.9, 60, 60.1, 59.9, 60, 61]
    assert detect_high_resting_hr(make_frame(resting_heart_rate=heart_rate)) == []


def test_load_spikes_report_the_first_day_of_each_stretch() -> None:
    index = pd.date_range("2026-01-01", periods=8, freq="D")
    acwr = pd.Series([1.0, 1.6, 1.7, 1.2, float("nan"), 1.55, 1.4, 1.9], index=index)

    found = detect_load_spikes(acwr)

    assert [a.date.day for a in found] == [2, 6, 8]
    assert found[0].value == 1.6
