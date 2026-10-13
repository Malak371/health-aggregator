"""Streamlit dashboard. All data and analysis come from the backend API."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import charts
from api_client import ApiError, BackendClient

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO = "demo"
MOOD_LABELS = {1: "1 · very low", 2: "2 · low", 3: "3 · okay", 4: "4 · good", 5: "5 · great"}
RETENTION = "90 days after you last open or update it"
RANGES = {"30 days": 30, "90 days": 90, "All": None}
UPLOAD_KINDS = {
    "Daily metrics CSV": ("csv", ["csv"]),
    "Apple Health export.xml": ("apple-health", ["xml"]),
}
# The uploaded dataset's key is kept in the page address so a reload, or a
# bookmark, comes back to the same data.
LINK_PARAM = "data"
DATASET_PATHS = {
    "insights": "/insights",
    "metrics": "/metrics",
    "derived": "/metrics/derived",
    "prediction": "/prediction",
}
ZONE_HELP = "Under 0.8 undertraining · 0.8-1.3 optimal · 1.3-1.5 caution · over 1.5 high"

st.set_page_config(page_title="Health & Performance Aggregator", page_icon="❤️‍🩹", layout="wide")


def backend_url() -> str:
    if url := os.environ.get("BACKEND_URL"):
        return url
    try:
        return str(st.secrets["BACKEND_URL"])
    except (KeyError, FileNotFoundError):
        return "http://localhost:8000"


@st.cache_data(ttl=300, show_spinner=False)
def load_dataset(url: str, slug: str, version: int) -> dict[str, Any]:
    """Everything the page needs for one dataset, fetched in parallel and cached.

    `version` changes after an upload so stale results are dropped.
    """
    return BackendClient(url).get_many(
        {name: f"/datasets/{slug}{path}" for name, path in DATASET_PATHS.items()}
    )


def show_error(error: ApiError) -> None:
    st.error(error.message)
    for item in error.errors:
        st.markdown(f"- {item}")


def fmt(value: float | None, pattern: str = "{:.0f}") -> str:
    return "n/a" if value is None else pattern.format(value)


# Built on every run on purpose: a client cached across runs survives a redeploy
# as an instance of the old class and breaks as soon as new code calls a new method.
client = BackendClient(backend_url())
state = st.session_state
state.setdefault("upload_slug", st.query_params.get(LINK_PARAM))


def own_dataset() -> str:
    """The visitor's dataset, created (and put in the page address) on first use."""
    if not state["upload_slug"]:
        state["upload_slug"] = client.create_dataset("My data")
        st.query_params[LINK_PARAM] = state["upload_slug"]
    return str(state["upload_slug"])


def forget_upload() -> None:
    state["upload_slug"] = None
    state["version"] += 1
    st.query_params.pop(LINK_PARAM, None)


state.setdefault("version", 0)

if not state.get("backend_ready"):
    with st.spinner(
        "Waking up the backend. On free hosting it sleeps when idle, "
        "so the first load can take up to a minute..."
    ):
        try:
            client.wait_until_ready()
            state["backend_ready"] = True
        except ApiError as error:
            show_error(error)
            st.stop()


# --- Sidebar: choose or upload a dataset ----------------------------------------

with st.sidebar:
    st.header("Data")
    options = {"Demo data (synthetic)": DEMO}
    if state["upload_slug"]:
        options["My data"] = state["upload_slug"]
    choice = st.radio("Dataset", list(options), index=len(options) - 1)
    slug = options[choice]

    st.divider()
    st.subheader("Try your own data")
    kind_label = st.selectbox("File type", list(UPLOAD_KINDS))
    kind, extensions = UPLOAD_KINDS[kind_label]
    uploaded = st.file_uploader("File", type=extensions, label_visibility="collapsed")
    if st.button("Import", disabled=uploaded is None, type="primary") and uploaded is not None:
        try:
            result = client.import_file(own_dataset(), kind, uploaded.name, uploaded.getvalue())
            state["version"] += 1
            state["import_result"] = result
            st.rerun()
        except ApiError as error:
            show_error(error)

    if result := state.pop("import_result", None):
        st.success(f"Imported {result['days_added'] + result['days_updated']} days.")
        for warning in result["warnings"]:
            st.caption(warning)

    if state["upload_slug"] and st.button("Delete my data"):
        try:
            client.delete_dataset(state["upload_slug"])
        except ApiError as error:
            show_error(error)
        forget_upload()
        st.rerun()

    if state["upload_slug"]:
        st.caption(
            "This page's address is now a private link to your data. Bookmark it to come "
            "back later; anyone you share it with can see the data too."
        )
    st.caption(
        "No account needed. Your data is stored under a random key that only appears in "
        f"your page address, and is deleted {RETENTION}."
    )
    st.caption("No file handy? Download a sample and import it:")
    for filename in ("demo_daily_metrics.csv", "apple_health_export_sample.xml"):
        sample = SAMPLES / filename
        if sample.exists():
            st.download_button(filename, sample.read_bytes(), file_name=filename)


# --- Load everything for the selected dataset -----------------------------------

st.title("Health & Performance Aggregator ❤️‍🩹")
st.write(
    "Sleep, activity, nutrition, mood, heart rate, training and study load in one place, "
    "with explainable insights about recovery and burnout risk."
)

try:
    loaded = load_dataset(backend_url(), slug, state["version"])
except ApiError as error:
    if error.status == 404 and slug != DEMO:
        # Shown after the rerun below, which would otherwise wipe it straight away.
        state["notice"] = "Your uploaded data has expired. Showing the demo data again."
        forget_upload()
        st.rerun()
    show_error(error)
    st.stop()

insights, prediction = loaded["insights"], loaded["prediction"]
metrics, derived = pd.DataFrame(loaded["metrics"]), pd.DataFrame(loaded["derived"])

metrics["date"] = pd.to_datetime(metrics["date"])
derived["date"] = pd.to_datetime(derived["date"])
daily = derived.merge(metrics, on="date", how="left")
daily["readiness_7d"] = daily["readiness"].rolling(7, min_periods=4).mean()

if notice := state.pop("notice", None):
    st.warning(notice)
if saved := state.pop("saved", None):
    st.success(saved)
if slug == DEMO:
    st.info(
        "You are looking at **synthetic demo data** (180 days, generated by a script in the "
        "repo). Use the sidebar to import your own CSV or Apple Health export.",
    )
st.caption(f"{insights['days']} days of data, {insights['first_date']} to {insights['last_date']}.")

load, recovery, burnout = insights["load"], insights["recovery"], insights["burnout"]

first, second, third, fourth = st.columns(4)
first.metric(
    "Readiness (latest day)",
    fmt(recovery["readiness_today"], "{:.0f} / 100"),
    delta=None if recovery["change"] is None else f"{recovery['change']:+.1f} week over week",
    help="Blend of sleep, resting heart rate against your baseline, and mood.",
)
second.metric(
    "Burnout risk",
    fmt(burnout["score"], "{:.0f} / 100"),
    delta=burnout["level"],
    delta_color="off",
    delta_arrow="off",
    help="Rule-based score from sleep debt, training strain, mood, study load and heart rate.",
)
third.metric(
    "Training load ratio (ACWR)",
    fmt(load["acwr"], "{:.2f}"),
    delta=load["acwr_zone"],
    delta_color="off",
    delta_arrow="off",
    help=f"Last 7 days of training load against the last 28. {ZONE_HELP}.",
)
fourth.metric("Sleep, 7-day average", fmt(daily["sleep_hours"].tail(7).mean(), "{:.1f} h"))

window = RANGES[st.radio("Show", list(RANGES), index=1, horizontal=True)]
shown = daily if window is None else daily.tail(window)


def plot(columns: dict[str, str], y_title: str, **options: Any) -> None:
    available = {name: label for name, label in columns.items() if shown[name].notna().any()}
    if not available:
        names = " or ".join(label.lower() for label in columns.values())
        st.caption(f"No data for {names} in this dataset yet.")
        return
    st.altair_chart(charts.time_series(shown, available, y_title, **options), width="stretch")


overview, checkin_tab, load_tab, insight_tab, model_tab, about = st.tabs(
    ["Overview", "Daily check-in", "Load & recovery", "Insights", "Prediction", "About"]
)

with overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Sleep")
        plot({"sleep_hours": "Sleep"}, "Hours")
        st.subheader("Resting heart rate")
        plot({"resting_heart_rate": "Resting heart rate"}, "Beats per minute")
    with right:
        st.subheader("Steps")
        plot({"steps": "Steps"}, "Steps", zero=True)
        st.subheader("Mood")
        plot({"mood_score": "Mood"}, "Score (1-5)", y_domain=(1, 5))

with load_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Training load: acute vs chronic")
        st.caption("Daily load is minutes x perceived exertion. Lines are rolling averages.")
        plot(
            {"training_acute": "Acute (7 days)", "training_chronic": "Chronic (28 days)"},
            "Load per day",
            zero=True,
        )
        st.subheader("Study load")
        plot(
            {"study_acute": "Acute (7 days)", "study_chronic": "Chronic (28 days)"},
            "Hours per day",
            zero=True,
        )
    with right:
        st.subheader("Acute:chronic workload ratio")
        st.caption(ZONE_HELP)
        plot(
            {"acwr": "ACWR"},
            "Ratio",
            reference_lines={0.8: "0.8", 1.3: "1.3", 1.5: "1.5"},
        )
        st.subheader("Readiness")
        st.caption(f"Recovery trend: **{recovery['direction']}** compared with the week before.")
        plot({"readiness": "Daily", "readiness_7d": "7-day average"}, "Score (0-100)")

    st.subheader("Burnout risk")
    left, right = st.columns(2)
    with left:
        plot({"burnout_risk": "Burnout risk"}, "Score (0-100)", y_domain=(0, 100))
    with right:
        parts = pd.DataFrame(burnout["components"]).dropna(subset=["value"])
        if parts.empty:
            st.caption("Not enough signals to score burnout risk.")
        else:
            parts["points"] = 100 * parts["value"] * parts["weight"] / parts["weight"].sum()
            parts = parts.sort_values("points", ascending=False)
            st.caption(
                f"What makes up the latest score of {fmt(burnout['score'])}: each bar is "
                "the number of points one signal contributes."
            )
            st.altair_chart(
                charts.magnitude_bars(parts, "points", "label", "Points of the score"),
                width="stretch",
            )

with insight_tab:
    st.subheader("Next steps")
    for item in insights["recommendations"]:
        st.markdown(f"- {item['message']}  \n  :gray[Why: {item['reason']}]")

    st.subheader("Relationships in your data")
    st.caption(
        "Pearson correlation. A link in the data does not show that one thing causes another."
    )
    if not insights["correlations"]:
        st.caption("Not enough overlapping data yet (at least 14 days per pair).")
    for column, item in zip(st.columns(3) * 2, insights["correlations"], strict=False):
        with column:
            st.metric(
                f"{item['x']} / {item['y']}".replace("_", " "),
                f"r = {item['r']:+.2f}",
                delta=f"{item['strength']}, n = {item['n']}",
                delta_color="off",
                delta_arrow="off",
            )
            st.write(item["insight"])

    st.subheader("Unusual days")
    if insights["anomalies"]:
        table = pd.DataFrame(insights["anomalies"])[["date", "kind", "message"]]
        table["kind"] = table["kind"].str.replace("_", " ")
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.success("No anomalies detected in this period.")

with model_tab:
    st.subheader("Will tomorrow be a low-readiness day?")
    if not prediction["available"]:
        st.warning(f"No prediction for this dataset. {prediction['reason']}")
    else:
        detail, evaluation = prediction["prediction"], prediction["evaluation"]
        left, right = st.columns([1, 2])
        with left:
            st.metric(
                f"Chance of low readiness on {detail['predicts']}",
                f"{detail['probability_low_readiness']:.0%}",
                help=(
                    "Low readiness means a score under "
                    f"{prediction['threshold']:.0f}: the bottom third of this "
                    "dataset's training period."
                ),
            )
            st.caption(f"Based on data up to {detail['based_on']}. Model: {prediction['model']}.")
        with right:
            st.markdown("**What is driving this prediction**")
            contributions = pd.DataFrame(detail["contributions"])
            st.altair_chart(
                charts.signed_bars(
                    contributions[["label", "contribution", "value"]],
                    "contribution",
                    "label",
                    "Push on the prediction (log-odds)",
                ),
                width="stretch",
            )

        st.subheader("How good is the model?")
        st.caption(
            f"Trained on the first {evaluation['train_days']} days and tested on the most recent "
            f"{evaluation['test_days']} ({evaluation['test_start']} to {evaluation['test_end']}), "
            "which it never saw. Splitting by time avoids learning from the future."
        )
        persistence = evaluation["baseline_persistence_roc_auc"]
        first, second, third, fourth = st.columns(4)
        first.metric("ROC-AUC (model)", f"{evaluation['roc_auc']:.2f}")
        second.metric(
            "ROC-AUC (tomorrow = today)",
            fmt(persistence, "{:.2f}"),
            help="A baseline that ranks days by today's readiness alone.",
        )
        third.metric("Accuracy (model)", f"{evaluation['accuracy']:.0%}")
        fourth.metric(
            "Accuracy (always guess majority)",
            f"{evaluation['baseline_majority_accuracy']:.0%}",
        )
        if persistence is not None and evaluation["roc_auc"] <= persistence:
            st.info(
                "On this test period the model does not beat the simple "
                "'tomorrow will be like today' baseline. Its value here is the explanation "
                "of which signals matter, not extra accuracy."
            )
        st.markdown("**What the model learned overall**")
        st.caption(
            "Change in the log-odds of a low-readiness day when a feature rises by one "
            "standard deviation."
        )
        st.altair_chart(
            charts.signed_bars(
                pd.DataFrame(prediction["feature_effects"])[["label", "coefficient"]],
                "coefficient",
                "label",
                "Coefficient",
            ),
            width="stretch",
        )
        st.caption(
            f"A test set of {evaluation['test_days']} days is small, so these numbers are "
            "rough. See DOCUMENTATION.md for the method and its limits."
        )

with checkin_tab:
    st.subheader("Log a day")
    st.caption(
        "Add what a wearable cannot measure, or keep a log without one. Every field is "
        "optional, and saving again for the same day only changes the fields you fill in. "
        f"Entries go to your own dataset, which is kept until {RETENTION}."
    )
    today = dt.date.today()
    # One day of slack, because the server's clock may be behind the visitor's time zone.
    day = st.date_input("Day", value=today, max_value=today + dt.timedelta(days=1))

    # Pre-fill with what is already stored for that day when looking at your own data.
    stored: dict[str, Any] = {}
    if slug != DEMO:
        match = metrics[metrics["date"].dt.date == day]
        if not match.empty:
            stored = {k: v for k, v in match.iloc[0].items() if pd.notna(v)}

    def number(label: str, field: str, top: float, step: float, **options: Any) -> float | None:
        value: float | None = st.number_input(
            label,
            min_value=0.0,
            max_value=top,
            step=step,
            value=stored.get(field),
            placeholder="Not logged",
            key=f"checkin_{field}_{day}",
            **options,
        )
        return value

    def choice(label: str, field: str, options: list[int], **extra: Any) -> int | None:
        current = stored.get(field)
        picked: int | None = st.radio(
            label,
            options,
            index=options.index(int(current)) if current in options else None,
            horizontal=True,
            key=f"checkin_{field}_{day}",
            **extra,
        )
        return picked

    with st.form("checkin"):
        left, right = st.columns(2)
        with left:
            mood = choice("Mood", "mood_score", list(MOOD_LABELS), format_func=MOOD_LABELS.get)
            sleep = number("Sleep last night (hours)", "sleep_hours", 24.0, 0.25)
            study = number("Study or deep work (hours)", "study_hours", 24.0, 0.5)
        with right:
            minutes = number("Training (minutes)", "training_minutes", 1440.0, 5.0)
            rpe = choice(
                "How hard was the training? (1 very easy, 10 maximal)",
                "training_rpe",
                list(range(1, 11)),
            )
        with st.expander("More: steps, resting heart rate, sugar"):
            steps = number("Steps", "steps", 200000.0, 500.0, format="%.0f")
            heart_rate = number("Resting heart rate (bpm)", "resting_heart_rate", 220.0, 1.0)
            sugar = number("Sugar (grams)", "sugar_grams", 1000.0, 5.0)
        submitted = st.form_submit_button("Save day", type="primary", key="checkin_save")

    if submitted:
        entry = {
            "mood_score": mood,
            "sleep_hours": sleep,
            "study_hours": study,
            "training_minutes": minutes,
            "training_rpe": rpe,
            "steps": None if steps is None else int(steps),
            "resting_heart_rate": heart_rate,
            "sugar_grams": sugar,
        }
        entry = {name: value for name, value in entry.items() if value is not None}
        if not entry:
            st.warning("Fill in at least one field to save the day.")
        else:
            try:
                is_new = not state["upload_slug"]
                client.add_records(own_dataset(), [{"date": day.isoformat(), **entry}])
                state["version"] += 1
                state["saved"] = f"Saved {day:%A %d %B}." + (
                    " This page's address is now your private link: bookmark it to keep logging."
                    if is_new
                    else ""
                )
                st.rerun()
            except ApiError as error:
                show_error(error)

with about:
    st.markdown(
        """
**How it works.** This dashboard is a Streamlit app that only draws charts. A FastAPI
backend stores the data (PostgreSQL in production), computes every metric and trains
the model. The code, tests and documentation are on
[GitHub](https://github.com/Malak371/health-aggregator).

**Privacy.** The demo data is synthetic. Files you import and days you log are kept under a
random key that only appears in your page address, are never listed anywhere, and are
deleted 90 days after you last open or update them. You can delete them sooner from the sidebar.
"""
    )

st.divider()
st.caption(f"⚠️ {insights['disclaimer']}")
