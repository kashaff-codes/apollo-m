"""Charts for Topic watch: where a topic is being discussed, and when."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import theme as T  # noqa: E402


def where(rows: pd.DataFrame) -> go.Figure:
    """Matches per community, largest first: a single-hue bar per community."""
    c = rows.groupby("subreddit").size().sort_values()
    fig = go.Figure(go.Bar(
        x=c.values, y=c.index, orientation="h",
        marker=dict(color=T.SERIES[0], line=dict(width=0)),
        text=[f"{v:,}" for v in c.values], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} matches<extra></extra>",
    ))
    fig.update_xaxes(title="matches", rangemode="tozero")
    fig.update_layout(height=max(320, 42 * len(c) + 120), margin=dict(l=20, r=60, t=78, b=20),
                      title_text="Where it is being discussed")
    return fig


def when(rows: pd.DataFrame, hours: float) -> go.Figure:
    """Matches per time bucket: an hourly histogram for a day or less, daily beyond."""
    freq = "15min" if hours <= 1 else "h" if hours <= 24 else "6h" if hours <= 72 else "D"
    t = pd.to_datetime(rows["time"], utc=True).dt.floor(freq)
    c = t.value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=c.index, y=c.values, marker=dict(color=T.SERIES[0], line=dict(width=0)),
        hovertemplate="%{x|%d %b %H:%M} UTC: %{y:,}<extra></extra>",
    ))
    fig.update_yaxes(title="matches", rangemode="tozero")
    fig.update_layout(height=360, bargap=0.15,
                      title_text={"15min": "When — per 15 minutes", "h": "When — per hour",
                                  "6h": "When — per 6 hours", "D": "When — per day"}[freq])
    return fig
