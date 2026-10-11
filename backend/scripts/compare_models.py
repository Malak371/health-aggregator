"""Compare the shipped model with alternatives on the demo dataset.

Uses expanding-window cross-validation (train on the past, test on the next
block) and prints the mean ROC-AUC of:

- the logistic regression used by the API
- a small gradient-boosted tree model
- the persistence baseline ("tomorrow will be like today")

Run from the repository root:

    PYTHONPATH=backend python backend/scripts/compare_models.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from app.ml.model import LOW_READINESS_QUANTILE, _labelled, make_pipeline

DEMO_CSV = Path(__file__).resolve().parents[2] / "data" / "samples" / "demo_daily_metrics.csv"


def main() -> None:
    frame = pd.read_csv(DEMO_CSV, parse_dates=["date"]).set_index("date").astype("float64")
    features, target = _labelled(frame)
    usable = target.notna()
    x, score = features[usable], target[usable]

    results: dict[str, list[float]] = {
        "logistic regression": [],
        "gradient boosting": [],
        "persistence": [],
    }
    for train, test in TimeSeriesSplit(n_splits=4).split(x):
        threshold = score.iloc[train].quantile(LOW_READINESS_QUANTILE)
        y = (score < threshold).astype(int)
        if y.iloc[train].nunique() < 2 or y.iloc[test].nunique() < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            linear = make_pipeline().fit(x.iloc[train], y.iloc[train])
            boosted = HistGradientBoostingClassifier(
                max_depth=2, max_iter=100, learning_rate=0.05, random_state=0
            ).fit(x.iloc[train], y.iloc[train])
            today = x["readiness_today"].iloc[test]
            results["logistic regression"].append(
                roc_auc_score(y.iloc[test], linear.predict_proba(x.iloc[test])[:, 1])
            )
            results["gradient boosting"].append(
                roc_auc_score(y.iloc[test], boosted.predict_proba(x.iloc[test])[:, 1])
            )
            results["persistence"].append(
                roc_auc_score(y.iloc[test], -today.fillna(today.median()))
            )

    print(f"{'model':<22}{'mean ROC-AUC':>14}   per fold")
    for name, scores in results.items():
        folds = "  ".join(f"{value:.2f}" for value in scores)
        print(f"{name:<22}{np.mean(scores):>14.3f}   {folds}")


if __name__ == "__main__":
    main()
