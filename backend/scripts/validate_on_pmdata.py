"""Evaluate the readiness model on real people, using the PMData dataset.

PMData (Thambawita et al., "PMData: A Sports Logging Dataset", MMSys 2020,
https://doi.org/10.1145/3339825.3394926) is licensed CC BY 4.0. It is not part
of this repository: download it from https://osf.io/vx4bk and point this script
at the folder that contains p01 ... p16.

For every participant the script runs the same procedure the API uses
(chronological 75/25 split, ROC-AUC, persistence baseline) for two targets:

1. computed  - the app's own readiness score, derived from sleep, resting heart
               rate and mood. This is exactly what the product does.
2. reported  - the participant's self-reported readiness to train (0-10). It is
               measured independently of the features, so it is the fairer test.
               Today's self-report is added as a feature for this target.

Run from the repository root:

    python backend/scripts/validate_on_pmdata.py data/external/pmdata \
        --output docs/pmdata-validation.md
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.pmdata import PMDataParticipant, load_participant
from app.ml.model import InsufficientDataError, train_and_explain

TARGETS = ("computed", "reported")


@dataclass(frozen=True)
class Outcome:
    participant: str
    days: int
    test_days: int | None = None
    model_auc: float | None = None
    persistence_auc: float | None = None
    top_feature: str | None = None
    skipped: str | None = None


def evaluate(person: PMDataParticipant, target: str) -> Outcome:
    days = int(person.frame.notna().any(axis=1).sum())
    try:
        if target == "computed":
            report = train_and_explain(person.frame)
        else:
            report = train_and_explain(
                person.frame,
                score=person.reported_readiness,
                extra_features=pd.DataFrame(
                    {"Self-reported readiness today": person.reported_readiness}
                ),
            )
    except InsufficientDataError as exc:
        return Outcome(person.name, days, skipped=str(exc))
    evaluation = report.evaluation
    return Outcome(
        participant=person.name,
        days=days,
        test_days=evaluation.test_days,
        model_auc=evaluation.roc_auc,
        persistence_auc=evaluation.baseline_persistence_roc_auc,
        top_feature=report.feature_effects[0].label,
    )


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render(target: str, outcomes: list[Outcome]) -> list[str]:
    scored = [o for o in outcomes if o.model_auc is not None]
    both = [o for o in scored if o.persistence_auc is not None]
    lines = [
        f"### Target: {target} readiness",
        "",
        "| Participant | Days | Test days | Model ROC-AUC | Persistence ROC-AUC "
        "| Strongest feature |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for o in outcomes:
        if o.skipped:
            lines.append(f"| {o.participant} | {o.days} | | skipped | | {o.skipped} |")
        else:
            lines.append(
                f"| {o.participant} | {o.days} | {o.test_days} | {_cell(o.model_auc)} | "
                f"{_cell(o.persistence_auc)} | {o.top_feature} |"
            )
    lines.append("")
    if scored:
        aucs = [o.model_auc for o in scored if o.model_auc is not None]
        lines.append(
            f"- Evaluated {len(scored)} of {len(outcomes)} participants. "
            f"Model ROC-AUC: mean {statistics.mean(aucs):.2f}, "
            f"median {statistics.median(aucs):.2f}, range {min(aucs):.2f} to {max(aucs):.2f}."
        )
        lines.append(
            f"- Better than chance (above 0.5) for {sum(a > 0.5 for a in aucs)} of {len(aucs)}."
        )
    if both:
        wins = sum(o.model_auc > o.persistence_auc for o in both)  # type: ignore[operator]
        baseline = [o.persistence_auc for o in both if o.persistence_auc is not None]
        paired = [o.model_auc for o in both if o.model_auc is not None]
        lines.append(
            f"- Against persistence ({len(both)} participants with both scores): model mean "
            f"{statistics.mean(paired):.2f} vs persistence mean {statistics.mean(baseline):.2f}; "
            f"the model is ahead for {wins} of {len(both)}."
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
        "# Validation on PMData",
        "",
        "Generated by `backend/scripts/validate_on_pmdata.py`. Data: PMData "
        "(Thambawita et al., MMSys 2020, https://doi.org/10.1145/3339825.3394926), "
        "licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data was "
        "converted to this project's daily format; it is not redistributed here.",
        "",
    ]
    for target in TARGETS:
        lines += render(target, [evaluate(person, target) for person in people])
    report = "\n".join(lines)
    print(report)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
