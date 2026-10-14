"""Does one model trained across people do better than one model per person?

The per-person evaluation (validate_on_pmdata.py) found no signal for
self-reported readiness. Each person has only about 100 usable days, so this
script pools them: for every participant, a model is trained on all the *other*
participants and tested on that participant's days (leave-one-person-out). The
held-out person contributes nothing to training.

A day is "low" when the next day's score is below the 33rd percentile of that
person's own scores, so the label stays personal while the model is shared.
Because the held-out person's cutoff uses all of their days, this measures how
well low days can be *ranked*, not a deployable forecast.

Compared on the same days:
- pooled logistic regression (the app's pipeline)
- the same, with every feature standardised within each person first. The label
  is relative to the person, so the inputs arguably should be too: one person's
  "6 out of 10" is another's "8". This uses the held-out person's own feature
  averages (not their labels).
- pooled gradient boosting (there is now enough data to try it)
- persistence: rank tomorrow by today's value of the score

Data: PMData (Thambawita et al., MMSys 2020, https://doi.org/10.1145/3339825.3394926),
CC BY 4.0, downloaded separately from https://osf.io/vx4bk.

    python backend/scripts/pooled_model_on_pmdata.py data/external/pmdata \
        --output docs/pmdata-pooled.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.pmdata import PMDataParticipant, load_participant
from app.ml.model import (
    LOW_READINESS_QUANTILE,
    InsufficientDataError,
    _labelled,
    make_pipeline,
)

MIN_DAYS = 30
MIN_PER_CLASS = 5
TODAY = "Self-reported readiness today"


@dataclass(frozen=True)
class PersonData:
    name: str
    features: pd.DataFrame
    low_tomorrow: pd.Series
    score_today: pd.Series


def prepare(person: PMDataParticipant, target: str) -> PersonData | None:
    """Features, labels and today's score for one person, or None if unusable."""
    try:
        if target == "computed":
            features, tomorrow, today = _labelled(person.frame)
        else:
            features, tomorrow, today = _labelled(
                person.frame,
                score=person.reported_readiness,
                extra_features=pd.DataFrame({TODAY: person.reported_readiness}),
            )
    except InsufficientDataError:
        return None
    usable = tomorrow.notna() & today.notna()
    features, tomorrow, today = features[usable], tomorrow[usable], today[usable]
    if len(features) < MIN_DAYS:
        return None
    low = (tomorrow < tomorrow.quantile(LOW_READINESS_QUANTILE)).astype(int)
    if min(low.sum(), len(low) - low.sum()) < MIN_PER_CLASS:
        return None
    return PersonData(person.name, features, low, today)


def within_person(features: pd.DataFrame) -> pd.DataFrame:
    """Each feature as standard deviations from that person's own mean."""
    spread = features.std().where(lambda s: s > 0)
    return (features - features.mean()) / spread


def evaluate(target: str, people: list[PMDataParticipant]) -> list[str]:
    prepared = [data for person in people if (data := prepare(person, target)) is not None]
    # Every person must present the same columns to a shared model.
    columns = sorted(set().union(*(data.features.columns for data in prepared)))
    rows = []
    for held_out in prepared:
        others = [data for data in prepared if data.name != held_out.name]
        x_train = pd.concat([data.features.reindex(columns=columns) for data in others])
        y_train = pd.concat([data.low_tomorrow for data in others])
        x_test = held_out.features.reindex(columns=columns)
        y_test = held_out.low_tomorrow
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            linear = make_pipeline().fit(x_train, y_train)
            boosted = HistGradientBoostingClassifier(
                max_depth=3, max_iter=200, learning_rate=0.05, random_state=0
            ).fit(x_train, y_train)
            relative = make_pipeline().fit(
                pd.concat([within_person(d.features).reindex(columns=columns) for d in others]),
                y_train,
            )
            x_test_relative = within_person(held_out.features).reindex(columns=columns)
            rows.append(
                (
                    held_out.name,
                    len(y_test),
                    len(y_train),
                    roc_auc_score(y_test, linear.predict_proba(x_test)[:, 1]),
                    roc_auc_score(y_test, relative.predict_proba(x_test_relative)[:, 1]),
                    roc_auc_score(y_test, boosted.predict_proba(x_test)[:, 1]),
                    roc_auc_score(y_test, -held_out.score_today),
                )
            )

    lines = [
        f"### Target: {target} readiness",
        "",
        "| Held-out participant | Test days | Training days (others) | Logistic regression "
        "| Logistic regression, per-person scaling | Gradient boosting | Persistence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {name} | {days} | {train} | {linear:.2f} | {relative:.2f} | {boosted:.2f} "
        f"| {persistence:.2f} |"
        for name, days, train, linear, relative, boosted, persistence in rows
    ]
    lines.append("")
    models = (
        ("Logistic regression", 3),
        ("Logistic regression, per-person scaling", 4),
        ("Gradient boosting", 5),
        ("Persistence", 6),
    )
    for label, index in models:
        scores = [row[index] for row in rows]
        beats = sum(row[index] > row[6] for row in rows)
        versus = "" if index == 6 else f"; ahead of persistence for {beats} of {len(rows)}"
        lines.append(
            f"- {label}: mean ROC-AUC {statistics.mean(scores):.2f}, "
            f"median {statistics.median(scores):.2f}, "
            f"range {min(scores):.2f} to {max(scores):.2f}{versus}."
        )
    lines.append(
        f"- {len(rows)} of {len(people)} participants had at least {MIN_DAYS} usable days and "
        f"{MIN_PER_CLASS} days of each class."
    )
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path, help="folder containing p01 ... p16")
    parser.add_argument("--output", type=Path, help="write the Markdown report here")
    args = parser.parse_args()

    folders = sorted(path for path in args.folder.iterdir() if (path / "pmsys").is_dir())
    if not folders:
        raise SystemExit(f"No participant folders found in {args.folder}")
    people = [load_participant(folder) for folder in folders]

    lines = [
        "# Pooled model on PMData (leave one person out)",
        "",
        "Generated by `backend/scripts/pooled_model_on_pmdata.py`. For each participant a "
        "model is trained on all other participants and tested on that participant. Data: "
        "PMData (Thambawita et al., MMSys 2020, https://doi.org/10.1145/3339825.3394926), "
        "licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); not "
        "redistributed here.",
        "",
    ]
    for target in ("reported", "computed"):
        lines += evaluate(target, people)
    report = "\n".join(lines)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
