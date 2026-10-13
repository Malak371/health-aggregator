"""Rule-based next steps. Each one carries the observation that triggered it."""

from __future__ import annotations

from dataclasses import dataclass

# Below this many days there is too little data to call anything "stable".
MIN_DAYS = 7


@dataclass(frozen=True)
class Recommendation:
    message: str
    reason: str


@dataclass(frozen=True)
class Snapshot:
    """The latest values the rules look at. None means "not available"."""

    avg_sleep_7d: float | None = None
    sleep_sugar_r: float | None = None
    recent_high_hr_days: int = 0
    acwr: float | None = None
    study_acute: float | None = None
    burnout_score: float | None = None
    recovery_direction: str = "unknown"
    days_logged: int | None = None


def generate_recommendations(snapshot: Snapshot) -> list[Recommendation]:
    found: list[Recommendation] = []

    if snapshot.burnout_score is not None and snapshot.burnout_score >= 66:
        found.append(
            Recommendation(
                "Several strain signals are high at once. Consider planning a lighter week: "
                "fewer hard sessions, shorter study blocks and protected sleep.",
                f"Burnout-risk score is {snapshot.burnout_score:.0f}/100.",
            )
        )

    if snapshot.avg_sleep_7d is not None and snapshot.avg_sleep_7d < 6.5:
        found.append(
            Recommendation(
                "Sleep has been short this week. A consistent bedtime or less screen time "
                "before bed may help.",
                f"7-day average sleep is {snapshot.avg_sleep_7d:.1f} h.",
            )
        )

    if snapshot.acwr is not None and snapshot.acwr > 1.3:
        found.append(
            Recommendation(
                "Training load is climbing faster than your recent norm. Holding volume "
                "steady for a few days lets your body catch up.",
                f"Acute:chronic workload ratio is {snapshot.acwr:.2f} (above 1.3).",
            )
        )
    elif snapshot.acwr is not None and snapshot.acwr < 0.8:
        found.append(
            Recommendation(
                "Training load is well below your recent norm. If this is not a planned "
                "rest week, ease back in gradually rather than all at once.",
                f"Acute:chronic workload ratio is {snapshot.acwr:.2f} (below 0.8).",
            )
        )

    if snapshot.study_acute is not None and snapshot.study_acute > 7:
        found.append(
            Recommendation(
                "Study hours have been heavy all week. Short breaks and a hard stop time "
                "in the evening help protect sleep.",
                f"7-day average study time is {snapshot.study_acute:.1f} h/day.",
            )
        )

    if snapshot.recent_high_hr_days:
        found.append(
            Recommendation(
                "Resting heart rate was elevated recently. Lighter activity, hydration or "
                "extra rest are reasonable on days like that.",
                f"{snapshot.recent_high_hr_days} elevated day(s) in the last two weeks.",
            )
        )

    if snapshot.sleep_sugar_r is not None and snapshot.sleep_sugar_r < -0.3:
        found.append(
            Recommendation(
                "Shorter nights tend to come with more sugar. Improving sleep may make "
                "cravings easier to manage.",
                f"Sleep and sugar intake correlate at r = {snapshot.sleep_sugar_r:.2f}.",
            )
        )

    if snapshot.recovery_direction == "declining" and not found:
        found.append(
            Recommendation(
                "Readiness has dipped compared with last week. An easier day or two may help.",
                "Mean readiness fell week over week.",
            )
        )

    if not found and snapshot.days_logged is not None and snapshot.days_logged < MIN_DAYS:
        found.append(
            Recommendation(
                "Keep logging. Trends and recommendations start to appear after about a week.",
                f"Only {snapshot.days_logged} day(s) of data so far.",
            )
        )

    if not found:
        found.append(
            Recommendation(
                "Your recent data looks stable. Keeping current habits should sustain it.",
                "No rule was triggered.",
            )
        )
    return found
