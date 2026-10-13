import datetime as dt

import numpy as np
import pandas as pd
import pytest

from app.analytics.recovery import readiness_score
from app.ml.features import FEATURE_LABELS, build_features, next_day_readiness
from app.ml.model import (
    REGULARISATION_GRID,
    InsufficientDataError,
    choose_regularisation,
    train_and_explain,
)
from tests.helpers import make_frame


def test_features_never_use_information_from_later_days(demo_frame: pd.DataFrame) -> None:
    cutoff = 120
    tampered = demo_frame.copy()
    tampered.iloc[cutoff:] = tampered.iloc[cutoff:] * 3

    original = build_features(demo_frame).iloc[:cutoff]
    recomputed = build_features(tampered).iloc[:cutoff]

    pd.testing.assert_frame_equal(original, recomputed)


def test_target_is_the_following_days_readiness(demo_frame: pd.DataFrame) -> None:
    readiness = readiness_score(demo_frame)
    target = next_day_readiness(demo_frame)

    assert target.iloc[10] == readiness.iloc[11]
    assert np.isnan(target.iloc[-1])


def test_every_feature_has_a_human_readable_label(demo_frame: pd.DataFrame) -> None:
    assert set(build_features(demo_frame).columns) == set(FEATURE_LABELS)


def test_report_on_the_demo_data(demo_frame: pd.DataFrame) -> None:
    report = train_and_explain(demo_frame)
    evaluation = report.evaluation

    # Chronological split: the test period is the most recent quarter.
    assert evaluation.train_days == 118
    assert evaluation.test_days == 40
    assert evaluation.test_end == dt.date(2026, 9, 26)
    assert evaluation.test_start > demo_frame.index[0].date()

    # Clearly better than chance; the docs discuss how it compares to persistence.
    assert evaluation.roc_auc > 0.7
    assert evaluation.baseline_persistence_roc_auc is not None
    assert report.regularisation_c in REGULARISATION_GRID

    sizes = [abs(effect.coefficient) for effect in report.feature_effects]
    assert sizes == sorted(sizes, reverse=True)
    assert len(report.feature_effects) == len(FEATURE_LABELS)


def test_prediction_is_for_the_day_after_the_last_record(demo_frame: pd.DataFrame) -> None:
    prediction = train_and_explain(demo_frame).prediction

    assert prediction.based_on == dt.date(2026, 9, 27)
    assert prediction.predicts == dt.date(2026, 9, 28)
    assert 0 <= prediction.probability_low_readiness <= 1
    pushes = [abs(item.contribution) for item in prediction.contributions]
    assert pushes == sorted(pushes, reverse=True)


def test_training_is_deterministic(demo_frame: pd.DataFrame) -> None:
    assert train_and_explain(demo_frame) == train_and_explain(demo_frame)


def test_features_that_were_never_recorded_are_left_out(demo_frame: pd.DataFrame) -> None:
    wearable_only = demo_frame.assign(mood_score=np.nan, study_hours=np.nan)

    names = {effect.feature for effect in train_and_explain(wearable_only).feature_effects}

    assert "mood_3d_avg" not in names
    assert "study_acute" not in names
    assert "readiness_today" in names


def test_short_datasets_are_refused_with_an_explanation(demo_frame: pd.DataFrame) -> None:
    with pytest.raises(InsufficientDataError, match="about 81 days"):
        train_and_explain(demo_frame.iloc[:45])


def test_flat_data_is_refused() -> None:
    flat = make_frame(sleep_hours=[7.0] * 120, mood_score=[3.0] * 120)

    with pytest.raises(InsufficientDataError):
        train_and_explain(flat)


def test_regularisation_falls_back_when_no_fold_is_usable() -> None:
    x = pd.DataFrame({"a": np.arange(20.0)})
    y = pd.Series([0] * 19 + [1])

    assert choose_regularisation(x, y) == 0.1
