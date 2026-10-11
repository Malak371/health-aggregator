"""Generate the synthetic demo dataset.

The data is driven by a hidden "fatigue" state: training and study load push
it up, sleep brings it down, and it leaks into sleep, resting heart rate, mood,
steps and sugar intake with noise. That gives the analytics realistic lagged
relationships to find without using anyone's personal data.

Run from the repository root:

    python backend/scripts/generate_synthetic_data.py
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DAYS = 180
START_DATE = dt.date(2026, 4, 1)
OUTPUT = Path(__file__).resolve().parents[2] / "data" / "samples" / "demo_daily_metrics.csv"

# Stretches of unusually heavy study (deadlines, exams), as (first_day, last_day).
CRUNCH_PERIODS = [(48, 60), (108, 120), (163, 176)]
# Weeks in which training volume jumps well above the plan.
OVERREACH_WEEKS = {7, 17, 23}


def generate(days: int = DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    fatigue = 0.3

    for day in range(days):
        date = START_DATE + dt.timedelta(days=day)
        week, weekday = divmod(day, 7)

        # Training: three build weeks then a lighter week, two rest days a week.
        block_factor = (0.85, 1.0, 1.15, 0.7)[week % 4]
        if week in OVERREACH_WEEKS:
            block_factor *= 1.6
        if weekday in (2, 6) or rng.random() < 0.05:
            minutes, rpe = 0.0, np.nan
        else:
            minutes = float(np.clip(rng.normal(55, 15) * block_factor, 15, 150))
            rpe = float(np.clip(rng.normal(5.5 + 1.5 * (block_factor - 1), 1.2), 2, 10))
        training_load = minutes * (0 if np.isnan(rpe) else rpe)

        # Study: lighter at weekends, much heavier in crunch periods.
        in_crunch = any(first <= day <= last for first, last in CRUNCH_PERIODS)
        study_mean = 8.5 if in_crunch else (1.5 if weekday >= 5 else 4.0)
        study = float(np.clip(rng.normal(study_mean, 1.0), 0, 12))

        # Sleep depends on yesterday's fatigue and on today's late-night study.
        sleep = float(
            np.clip(7.6 - 1.0 * fatigue - 0.18 * max(study - 6, 0) + rng.normal(0, 0.7), 3.5, 10)
        )

        # Fatigue carries over, rises with load and falls with a good night.
        fatigue = float(
            np.clip(
                0.8 * fatigue
                + 0.07 * training_load / 300
                + 0.05 * study / 4
                - 0.10 * (sleep - 7)
                + rng.normal(0, 0.06),
                0,
                2,
            )
        )

        rows.append(
            {
                "date": date.isoformat(),
                "sleep_hours": round(sleep, 1),
                "resting_heart_rate": round(55 + 10 * fatigue + rng.normal(0, 2.4), 1),
                "steps": int(max(rng.normal(6500 + 55 * minutes - 900 * fatigue, 1400), 500)),
                "mood_score": round(
                    float(
                        np.clip(
                            4.6 - 1.8 * fatigue + 0.25 * (sleep - 7) + rng.normal(0, 0.65), 1, 5
                        )
                    ),
                    1,
                ),
                "sugar_grams": int(max(rng.normal(38 + 18 * fatigue - 5 * (sleep - 7), 12), 5)),
                "training_minutes": round(minutes, 0),
                "training_rpe": None if np.isnan(rpe) else round(rpe, 0),
                "study_hours": round(study, 1),
            }
        )

    frame = pd.DataFrame(rows)
    # Real logs have gaps: blank out a few mood entries and one watch-free day.
    frame.loc[rng.choice(days, size=5, replace=False), "mood_score"] = np.nan
    frame.loc[rng.integers(30, days - 30), ["sleep_hours", "resting_heart_rate"]] = np.nan
    return frame


if __name__ == "__main__":
    data = generate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(data)} days to {OUTPUT}")
