"""Next-day low-readiness classifier.

Question: given everything known up to tonight, how likely is tomorrow to be a
low-readiness day? "Low" means below the 33rd percentile of the person's own
readiness in the training period, so the label is personal, not a global cutoff.

Model: standardised, L2-regularised logistic regression. With a few hundred
rows at most, a linear model is the largest thing the data can support, and its
coefficients are directly readable, which is the point of the dashboard. The
regularisation strength is chosen by expanding-window cross-validation inside
the training period.

Evaluation: the last 25% of days are held out in time order (never shuffled)
and scored with ROC-AUC, next to two baselines: always predicting the majority
class, and "tomorrow will be like today" (persistence).
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.analytics.recovery import readiness_score
from app.ml.features import FEATURE_LABELS, build_features

TEST_FRACTION = 0.25
LOW_READINESS_QUANTILE = 1 / 3
MIN_LABELLED_DAYS = 60
# Skip the first weeks, where the rolling features are mostly still empty.
WARM_UP_DAYS = 21
MIN_FEATURE_COVERAGE = 0.5
# The hold-out must contain this many days of each class to be worth scoring.
MIN_TEST_DAYS_PER_CLASS = 3
REGULARISATION_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
DEFAULT_REGULARISATION = 0.1
CV_FOLDS = 4


class InsufficientDataError(ValueError):
    """The dataset cannot support training; the message says why."""


@dataclass(frozen=True)
class Evaluation:
    train_days: int
    test_days: int
    test_start: dt.date
    test_end: dt.date
    positive_rate_test: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    baseline_majority_accuracy: float
    # None when today's readiness is missing on too many held-out days.
    baseline_persistence_roc_auc: float | None


@dataclass(frozen=True)
class FeatureEffect:
    feature: str
    label: str
    # Change in log-odds of a low-readiness day per one standard deviation.
    coefficient: float


@dataclass(frozen=True)
class Contribution:
    feature: str
    label: str
    value: float | None
    # coefficient x standardised value: this feature's push on today's prediction.
    contribution: float


@dataclass(frozen=True)
class Prediction:
    based_on: dt.date
    predicts: dt.date
    probability_low_readiness: float
    contributions: list[Contribution]


@dataclass(frozen=True)
class ReadinessReport:
    threshold: float
    regularisation_c: float
    evaluation: Evaluation
    feature_effects: list[FeatureEffect]
    prediction: Prediction


def make_pipeline(c: float = DEFAULT_REGULARISATION) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=1000)),
        ]
    )


def choose_regularisation(x: pd.DataFrame, y: pd.Series) -> float:
    """Pick C by mean ROC-AUC over expanding-window folds of the training data.

    Folds always train on the past and validate on the future. Folds whose
    train or validation part contains a single class are skipped.
    """
    scores: dict[float, list[float]] = {c: [] for c in REGULARISATION_GRID}
    for train, validate in TimeSeriesSplit(n_splits=CV_FOLDS).split(x):
        if y.iloc[train].nunique() < 2 or y.iloc[validate].nunique() < 2:
            continue
        for c in REGULARISATION_GRID:
            fitted = make_pipeline(c).fit(x.iloc[train], y.iloc[train])
            probabilities = fitted.predict_proba(x.iloc[validate])[:, 1]
            scores[c].append(float(roc_auc_score(y.iloc[validate], probabilities)))
    if not any(scores.values()):
        return DEFAULT_REGULARISATION
    return max(REGULARISATION_GRID, key=lambda c: float(np.mean(scores[c])))


def _labelled(
    frame: pd.DataFrame,
    score: pd.Series | None = None,
    extra_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Features for day t, the score on day t + 1, and the score on day t.

    `score` is the daily quantity whose low days are predicted (higher is better).
    It defaults to the readiness score computed from the frame; an evaluation can
    pass an independently measured one instead.
    """
    features = build_features(frame)
    if extra_features is not None:
        features = features.join(extra_features)
    features = features.iloc[WARM_UP_DAYS:]
    features = features.loc[:, features.notna().mean() >= MIN_FEATURE_COVERAGE]
    if features.shape[1] < 2:
        raise InsufficientDataError(
            "Too few signals are recorded to build features (at least two are needed)."
        )
    today = readiness_score(frame) if score is None else score.reindex(frame.index)
    target = today.shift(-1).reindex(features.index)
    return features, target, today.reindex(features.index)


def train_and_explain(
    frame: pd.DataFrame,
    score: pd.Series | None = None,
    extra_features: pd.DataFrame | None = None,
) -> ReadinessReport:
    """Evaluate on a chronological hold-out, then refit on everything to predict tomorrow.

    See `_labelled` for `score` and `extra_features`; the API uses neither.
    """
    features, target, score_today = _labelled(frame, score, extra_features)
    usable = target.notna() & features.notna().any(axis=1)
    x, y_score = features[usable], target[usable]
    if len(x) < MIN_LABELLED_DAYS:
        raise InsufficientDataError(
            f"The model needs {MIN_LABELLED_DAYS} days with a next-day score after a "
            f"{WARM_UP_DAYS}-day warm-up (about {MIN_LABELLED_DAYS + WARM_UP_DAYS} days of data); "
            f"this dataset has {len(x)} such days out of {len(frame)}."
        )

    split = int(len(x) * (1 - TEST_FRACTION))
    x_train, x_test = x.iloc[:split], x.iloc[split:]
    # The cutoff comes from the training period only, so the test set stays unseen.
    threshold = float(y_score.iloc[:split].quantile(LOW_READINESS_QUANTILE))
    y = (y_score < threshold).astype(int)
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if y_train.nunique() < 2:
        raise InsufficientDataError(
            "Readiness barely varies in this dataset, so there is nothing to learn."
        )
    if min(y_test.sum(), len(y_test) - y_test.sum()) < MIN_TEST_DAYS_PER_CLASS:
        raise InsufficientDataError(
            "The most recent quarter of the data has too few low-readiness (or too few "
            "normal) days to evaluate a model honestly. More history is needed."
        )

    with warnings.catch_warnings():
        # Features that are entirely missing in one split are imputed away by design.
        warnings.simplefilter("ignore", category=UserWarning)
        c = choose_regularisation(x_train, y_train)
        pipeline = make_pipeline(c).fit(x_train, y_train)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        persistence: float | None = None
        # Persistence ranks tomorrow by today's value of the same score.
        today = score_today.reindex(x_test.index)
        if today.notna().mean() > 0.9:
            persistence = float(roc_auc_score(y_test, -today.fillna(today.median())))

        evaluation = Evaluation(
            train_days=len(x_train),
            test_days=len(x_test),
            test_start=x_test.index[0].date(),
            test_end=x_test.index[-1].date(),
            positive_rate_test=round(float(y_test.mean()), 3),
            roc_auc=round(float(roc_auc_score(y_test, probabilities)), 3),
            accuracy=round(float(accuracy_score(y_test, predictions)), 3),
            precision=round(float(precision_score(y_test, predictions, zero_division=0)), 3),
            recall=round(float(recall_score(y_test, predictions, zero_division=0)), 3),
            baseline_majority_accuracy=round(float(max(y_test.mean(), 1 - y_test.mean())), 3),
            baseline_persistence_roc_auc=None if persistence is None else round(persistence, 3),
        )

        # Refit on every labelled day: the most recent weeks matter most for tomorrow.
        final = make_pipeline(c).fit(x, y)

    names = list(x.columns)
    coefficients = final.named_steps["model"].coef_[0]
    effects = sorted(
        (
            FeatureEffect(name, FEATURE_LABELS.get(name, name), round(float(coef), 3))
            for name, coef in zip(names, coefficients, strict=True)
        ),
        key=lambda effect: abs(effect.coefficient),
        reverse=True,
    )

    latest = features.iloc[[-1]]
    standardised = final[:-1].transform(latest)[0]
    contributions = sorted(
        (
            Contribution(
                feature=name,
                label=FEATURE_LABELS.get(name, name),
                value=None if np.isnan(raw) else round(float(raw), 2),
                contribution=round(float(coef * z), 3),
            )
            for name, coef, z, raw in zip(
                names, coefficients, standardised, latest.iloc[0], strict=True
            )
        ),
        key=lambda item: abs(item.contribution),
        reverse=True,
    )
    based_on = latest.index[0].date()
    prediction = Prediction(
        based_on=based_on,
        predicts=based_on + dt.timedelta(days=1),
        probability_low_readiness=round(float(final.predict_proba(latest)[0, 1]), 3),
        contributions=contributions,
    )
    return ReadinessReport(round(threshold, 1), c, evaluation, effects, prediction)
