"""Altair chart builders shared by the dashboard tabs."""

from __future__ import annotations

import altair as alt
import pandas as pd

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
MUTED = "#898781"
SERIES_COLORS = (BLUE, ORANGE, AQUA)
HEIGHT = 240


def time_series(
    frame: pd.DataFrame,
    series: dict[str, str],
    y_title: str,
    reference_lines: dict[float, str] | None = None,
    zero: bool = False,
    y_domain: tuple[float, float] | None = None,
) -> alt.LayerChart:
    """Line chart of one or more columns sharing a single y-axis.

    `series` maps column names to display labels, in a fixed colour order.
    Hovering shows a vertical guide and every series' value for that day.
    """
    labels = list(series.values())
    wide = frame[["date", *series]].rename(columns=series)
    long = wide.melt("date", var_name="series", value_name="value").dropna()
    scale = alt.Scale(zero=zero) if y_domain is None else alt.Scale(domain=list(y_domain))

    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("value:Q", title=y_title, scale=scale),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=labels, range=list(SERIES_COLORS[: len(labels)])),
                legend=alt.Legend(title=None, orient="top") if len(labels) > 1 else None,
            ),
        )
    )

    hover = alt.selection_point(fields=["date"], nearest=True, on="pointerover", empty=False)
    guide = (
        alt.Chart(wide)
        .mark_rule(color=MUTED)
        .encode(
            x="date:T",
            opacity=alt.condition(hover, alt.value(0.7), alt.value(0)),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%a %d %b %Y"),
                *[alt.Tooltip(f"{label}:Q", format=",.2~f") for label in labels],
            ],
        )
        .add_params(hover)
    )

    layers: list[alt.Chart] = [lines, guide]
    if reference_lines:
        marks = pd.DataFrame({"value": list(reference_lines), "label": reference_lines.values()})
        base = alt.Chart(marks).encode(y="value:Q")
        layers.insert(0, base.mark_rule(color=MUTED, strokeDash=[4, 4], strokeWidth=1))
        layers.append(
            base.mark_text(align="left", dx=4, dy=-6, color=MUTED, fontSize=11).encode(
                x=alt.value(0), text="label:N"
            )
        )
    return alt.layer(*layers).properties(height=HEIGHT)


def signed_bars(frame: pd.DataFrame, value: str, label: str, x_title: str) -> alt.Chart:
    """Horizontal bars for effects that push a prediction up (orange) or down (blue)."""
    data = frame.assign(
        direction=frame[value].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    )
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, height=14)
        .encode(
            x=alt.X(f"{value}:Q", title=x_title),
            y=alt.Y(
                f"{label}:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelLimit=260, labelOverlap=False),
            ),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[ORANGE, BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip(f"{label}:N", title="Feature")]
            + [
                alt.Tooltip(f"{column}:Q", format=".2f")
                for column in frame.columns
                if column != label and pd.api.types.is_numeric_dtype(frame[column])
            ],
        )
        .properties(height=alt.Step(28))
    )


def magnitude_bars(frame: pd.DataFrame, value: str, label: str, x_title: str) -> alt.Chart:
    """Horizontal bars for parts of a total, in a single colour."""
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=4, height=14, color=BLUE)
        .encode(
            x=alt.X(f"{value}:Q", title=x_title),
            y=alt.Y(
                f"{label}:N",
                title=None,
                sort=None,
                axis=alt.Axis(labelLimit=320, labelOverlap=False),
            ),
            tooltip=[
                alt.Tooltip(f"{label}:N", title="Signal"),
                alt.Tooltip(f"{value}:Q", format=".1f"),
            ],
        )
        .properties(height=alt.Step(28))
    )
