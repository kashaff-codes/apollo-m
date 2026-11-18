"""
Figure builders for the console. Each takes plain DataFrames and returns a
plotly Figure, so every chart can be built and checked without Streamlit
(tests/test_dashboard.py does exactly that).

Conventions, applied everywhere:
  * one y-axis per panel — two measures get two panels, never two scales;
  * alert bands use status colour AND marker shape AND their name;
  * categorical colour follows the entity, in the fixed SERIES order;
  * a single hue (SEQ) for magnitude, dark → light on the dark surface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import theme as T  # noqa: E402
from apollo_bands import COLOR, FLOOR, ICON, ORDER, SYMBOL  # noqa: E402

BAND_ZONES = [("CRITICAL", 0, 65), ("HIGH", 65, 75), ("MEDIUM", 75, 85), ("LOW", 85, 100)]


def _base(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, title_text=title or "")
    if not title:
        fig.update_layout(margin=dict(t=40))
    return fig


def _band_zones(fig: go.Figure, y0: float, y1: float, row=None, col=None) -> None:
    """Faint horizontal bands for the alert thresholds, named at the right edge."""
    for band, lo, hi in BAND_ZONES:
        lo_v, hi_v = max(lo, y0), min(hi, y1)
        if lo_v >= hi_v:
            continue
        kw = dict(row=row, col=col) if row else {}
        fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=T.rgba(COLOR[band], .07), line_width=0, layer="below", **kw)
        fig.add_annotation(xref="paper", x=1, xanchor="right", y=(lo_v + hi_v) / 2, yanchor="middle",
                           text=f"{ICON[band]} {band}", showarrow=False,
                           font=dict(size=10, color=T.rgba(COLOR[band], .85)), **({} if not row else {}))


# ═════════════════════════════════════════════════════════════════════════════
# Overview
# ═════════════════════════════════════════════════════════════════════════════

def landscape(meso: pd.DataFrame, label_worst: int = 3) -> go.Figure:
    """
    Every community at once: toxicity across, health up, size = comments
    analysed. The faint bands are the alert thresholds, so a point's band can
    be read from its height as well as its colour and shape.
    """
    fig = go.Figure()
    vol = meso["volume"] if "volume" in meso else meso["total_comments"]
    size = np.sqrt(vol.clip(lower=1))
    size = 9 + 17 * (size - size.min()) / max(size.max() - size.min(), 1e-9)
    for band in ORDER:
        m = meso["band"] == band
        if not m.any():
            continue
        g = meso[m]
        fig.add_trace(go.Scatter(
            x=g["toxicity_rate"], y=g["community_health_index"], mode="markers",
            name=f"{ICON[band]} {band}",
            marker=dict(size=size[m], color=T.rgba(COLOR[band], .72), symbol=SYMBOL[band],
                        line=dict(width=2, color=T.SURFACE)),
            customdata=np.stack([g["subreddit"], g["polarization"], g["churn_rate"],
                                 g["echo_chamber_index"], vol[m], g["total_comments"]], axis=-1),
            hovertemplate=("<b>%{customdata[0]}</b><br>CHI %{y:.1f} · " + band +
                           "<br>toxicity %{x:.3f}<br>polarisation %{customdata[1]:.3f}"
                           "<br>churn %{customdata[2]:.3f} · echo %{customdata[3]:.3f}"
                           "<br>%{customdata[4]:,.0f} comments in the period"
                           "<br>%{customdata[5]:,} analysed<extra></extra>"),
        ))
    y0 = max(0, meso["community_health_index"].min() - 4)
    y1 = min(100, meso["community_health_index"].max() + 4)
    _band_zones(fig, y0, y1)
    for _, r in meso.nsmallest(label_worst, "community_health_index").iterrows():
        fig.add_annotation(x=r["toxicity_rate"], y=r["community_health_index"], text=r["subreddit"],
                           showarrow=True, arrowhead=0, arrowcolor=T.MUTED, ax=28, ay=-22,
                           font=dict(size=11, color=T.INK))
    fig.update_xaxes(title="mean toxicity (toxic-bert)")
    fig.update_yaxes(title="Community Health Index", range=[y0, y1])
    return _base(fig, 470, "Health landscape — every community, sized by its comment volume")


def band_strip(meso: pd.DataFrame) -> go.Figure:
    """Part-to-whole of the four bands as one segmented bar, labelled in place."""
    counts = meso["band"].value_counts()
    total = int(counts.sum())
    fig = go.Figure()
    for band in ORDER:
        n = int(counts.get(band, 0))
        if not n:
            continue
        share = n / total
        fig.add_trace(go.Bar(
            x=[n], y=[""], orientation="h", name=f"{ICON[band]} {band}",
            marker=dict(color=COLOR[band], line=dict(width=2, color=T.SURFACE)),
            text=[f"{ICON[band]} {band} {n}" if share >= .12 else (str(n) if share >= .04 else "")],
            textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#0b0b0b" if band == "MEDIUM" else "#ffffff", size=12),
            hovertemplate=f"{band}: %{{x}} of {total} ({share:.0%})<extra></extra>",
        ))
    fig.update_layout(barmode="stack", showlegend=True, bargap=0)
    fig.update_xaxes(visible=False, range=[0, total])
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(t=70, b=8))
    return _base(fig, 170, f"Alert bands — {total} communities")


def heatmap(daily: pd.DataFrame, order: list[str]) -> go.Figure:
    """Community × day toxicity. Blank cells are days with no data, not zeros."""
    p = daily.pivot_table(index="subreddit", columns="date", values="avg_toxicity")
    p = p.reindex([s for s in order if s in p.index])
    cnt = daily.pivot_table(index="subreddit", columns="date", values="comment_count").reindex(p.index)
    fig = go.Figure(go.Heatmap(
        z=p.values, x=p.columns, y=p.index, colorscale=T.SEQ_SCALE, hoverongaps=False,
        customdata=cnt.values, xgap=1, ygap=1,
        colorbar=dict(title=dict(text="toxicity", side="right"), thickness=10, outlinewidth=0,
                      tickfont=dict(color=T.MUTED)),
        hovertemplate="<b>%{y}</b> · %{x|%b %d}<br>mean toxicity %{z:.3f}<br>%{customdata:,} comments<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return _base(fig, max(360, 16 * len(p) + 90),
                 "Daily toxicity — rows ordered worst CHI first; blank = no data that day")


def phase_space(ps: pd.DataFrame, bands: dict[str, str], max_frames: int = 60) -> go.Figure:
    """
    Animated: each community's toxicity level (x) against its momentum (y),
    day by day. Press play to watch the period unfold. Top-right = toxic and
    worsening. Bands come from the period's CHI, held fixed so a point keeps
    its colour and shape through the animation.
    """
    dates = sorted(ps["date"].unique())
    if len(dates) > max_frames:
        dates = dates[:: int(np.ceil(len(dates) / max_frames))] + [dates[-1]]
    xr = [ps["level"].min() * .92, ps["level"].max() * 1.06]
    mlim = float(np.nanmax(np.abs(ps["momentum"]))) * 1.15 or .01
    yr = [-mlim, mlim]

    def traces(day):
        g = ps[ps["date"] == day]
        out = []
        for band in ORDER:
            h = g[g["subreddit"].map(bands) == band]
            out.append(go.Scatter(
                x=h["level"], y=h["momentum"], mode="markers", name=f"{ICON[band]} {band}",
                text=h["subreddit"], ids=h["subreddit"],
                marker=dict(size=13, color=T.rgba(COLOR[band], .8), symbol=SYMBOL[band],
                            line=dict(width=2, color=T.SURFACE)),
                hovertemplate="<b>%{text}</b><br>level %{x:.3f}<br>5-day change %{y:+.3f}<extra></extra>",
            ))
        return out

    fig = go.Figure(data=traces(dates[0]),
                    frames=[go.Frame(data=traces(d), name=pd.Timestamp(d).strftime("%Y-%m-%d")) for d in dates])
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    fig.add_annotation(xref="paper", yref="paper", x=.99, y=.98, text="toxic and worsening ↗",
                       showarrow=False, font=dict(color=T.INK_2, size=11), xanchor="right")
    fig.add_annotation(xref="paper", yref="paper", x=.01, y=.02, text="↙ calm and improving",
                       showarrow=False, font=dict(color=T.MUTED, size=11), xanchor="left")
    steps = [dict(method="animate", label=pd.Timestamp(d).strftime("%b %d"),
                  args=[[pd.Timestamp(d).strftime("%Y-%m-%d")],
                        dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))])
             for d in dates]
    fig.update_layout(
        xaxis=dict(title="toxicity level (5-day trailing mean)", range=xr),
        yaxis=dict(title="momentum (change over 5 days)", range=yr),
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-.16, xanchor="left", yanchor="top",
                          bgcolor="#1b1c3d", bordercolor=T.AXIS, font=dict(color=T.INK),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=380, redraw=False),
                                                         transition=dict(duration=300, easing="cubic-in-out"),
                                                         fromcurrent=True, mode="immediate")]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=.12, len=.88, y=-.1, yanchor="top", steps=steps,
                      currentvalue=dict(prefix="day ", font=dict(color=T.INK)),
                      font=dict(color=T.MUTED), bgcolor="#1b1c3d", bordercolor=T.AXIS, tickcolor=T.AXIS)],
        margin=dict(b=110),
    )
    return _base(fig, 560, "Instability phase space — level against momentum, animated over the period")


# ═════════════════════════════════════════════════════════════════════════════
# Community drill-down
# ═════════════════════════════════════════════════════════════════════════════

def gauge(chi: float, band: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=chi, number=dict(font=dict(color=T.INK, size=44), valueformat=".1f"),
        title=dict(text=f"CHI · {ICON[band]} {band}", font=dict(color=T.INK_2, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=T.MUTED, tickfont=dict(color=T.MUTED)),
            bar=dict(color=T.INK, thickness=.22), bgcolor=T.SURFACE, borderwidth=0,
            steps=[dict(range=[lo, hi], color=T.rgba(COLOR[b], .32)) for b, lo, hi in BAND_ZONES],
        )))
    return _base(fig, 250)


def waterfall(steps: pd.DataFrame) -> go.Figure:
    """CHI = 100 − 35·toxicity − 30·polarisation − 20·churn − 15·echo, step by step."""
    measure = ["absolute"] + ["relative"] * (len(steps) - 2) + ["total"]
    text = ["100"] + [f"{v:+.1f}" for v in steps["value"].iloc[1:-1]] + [f"{steps['value'].iloc[-1]:.1f}"]
    names = [str(x).replace(" ×", "<br>×") for x in steps["step"]]
    fig = go.Figure(go.Waterfall(
        x=names, y=steps["value"], measure=measure, text=text, textposition="outside",
        textfont=dict(color=T.INK_2), connector=dict(line=dict(color=T.AXIS, width=1)),
        increasing=dict(marker=dict(color=T.SERIES[0])), decreasing=dict(marker=dict(color=T.SERIES[1])),
        totals=dict(marker=dict(color=T.SERIES[0])),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 108], title="points")
    fig.update_xaxes(tickangle=0)
    return _base(fig, 330, "Where the score goes — each component's penalty")


def profile(row: pd.Series, median: pd.Series, components) -> go.Figure:
    """The four CHI inputs for this community against the corpus median."""
    names = [n for _, n, _ in components]
    vals = [float(row[c]) for c, _, _ in components]
    meds = [float(median[c]) for c, _, _ in components]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=meds + meds[:1], theta=names + names[:1], name="corpus median",
                                  line=dict(color=T.MUTED, width=2), fill="toself",
                                  fillcolor=T.rgba(T.MUTED, .08)))
    fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=names + names[:1], name=str(row["subreddit"]),
                                  line=dict(color=T.SERIES[0], width=2), fill="toself",
                                  fillcolor=T.rgba(T.SERIES[0], .22), marker=dict(size=8)))
    fig.update_layout(polar=dict(bgcolor=T.SURFACE, domain=dict(x=[.2, .8], y=[.06, .88]),
                                 radialaxis=dict(range=[0, max(1e-3, max(vals + meds)) * 1.15], gridcolor=T.GRID,
                                                 linecolor=T.AXIS, nticks=4, angle=90, tickangle=90,
                                                 tickfont=dict(color=T.MUTED, size=9)),
                                 angularaxis=dict(gridcolor=T.GRID, linecolor=T.AXIS,
                                                  tickfont=dict(color=T.INK_2, size=12))),
                      legend=dict(x=0, xanchor="left", y=-0.02, yanchor="top"), margin=dict(b=50))
    return _base(fig, 440, "Component profile vs the corpus median")


def history(daily_sub: pd.DataFrame, future_sub: pd.DataFrame | None = None,
            held_sub: pd.DataFrame | None = None, title: str = "") -> go.Figure:
    """
    Two panels on one time axis: toxicity (with the forecast band, when there
    is one) above, comments per day below. Two measures, two panels — never a
    second y-scale.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.72, .28], vertical_spacing=.06)
    d = daily_sub.sort_values("date")
    c = T.SERIES[0]
    fig.add_trace(go.Scatter(x=d["date"], y=d["avg_toxicity"], mode="lines+markers", name="observed",
                             line=dict(color=c, width=2), marker=dict(size=6, line=dict(width=2, color=T.SURFACE)),
                             hovertemplate="%{x|%b %d}: %{y:.3f}<extra>observed</extra>"), row=1, col=1)

    def fan(f: pd.DataFrame, name: str):
        f = f.sort_values("date")
        fig.add_trace(go.Scatter(x=list(f["date"]) + list(f["date"])[::-1],
                                 y=list(f["p90"]) + list(f["p10"])[::-1], fill="toself", mode="lines",
                                 fillcolor=T.rgba(T.VIOLET, .22), line=dict(width=0), name="p10–p90 band",
                                 hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=f["date"], y=f["p50"], mode="lines+markers", name=name,
                                 line=dict(color=T.VIOLET, width=2, dash="dot"), marker=dict(size=7),
                                 customdata=np.stack([f["p10"], f["p90"]], axis=-1),
                                 hovertemplate="%{x|%b %d}: p50 %{y:.3f} (p10 %{customdata[0]:.3f} – "
                                               "p90 %{customdata[1]:.3f})<extra>" + name + "</extra>"),
                      row=1, col=1)

    if future_sub is not None and not future_sub.empty:
        last = d["date"].max()
        fig.add_vline(x=last, line=dict(color=T.AXIS, width=1), row=1, col=1)
        fig.add_annotation(x=last, y=1, yref="y domain", text="last observed day", showarrow=False,
                           xanchor="right", font=dict(color=T.MUTED, size=10), row=1, col=1)
        fan(future_sub, "forecast (p50)")
    if held_sub is not None and not held_sub.empty:
        fan(held_sub, "prediction for held-out day (p50)")
        fig.add_vrect(x0=held_sub["date"].min() - pd.Timedelta(hours=12),
                      x1=held_sub["date"].max() + pd.Timedelta(hours=12),
                      fillcolor=T.rgba("#ffffff", .03), line_width=0, row=1, col=1)
        fig.add_annotation(x=held_sub["date"].min(), y=1, yref="y domain", text="held out of training",
                           showarrow=False, xanchor="left", font=dict(color=T.MUTED, size=10), row=1, col=1)

    if "comment_count" in d:
        fig.add_trace(go.Bar(x=d["date"], y=d["comment_count"], name="comments / day",
                             marker=dict(color=T.rgba(T.SERIES[2], .75), line=dict(width=0)),
                             hovertemplate="%{x|%b %d}: %{y:,} comments<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title="mean toxicity", row=1, col=1)
    fig.update_yaxes(title="comments", row=2, col=1)
    # Four entries are wider than the space beside a title, so the legend
    # goes under the plot instead of colliding with it.
    fig.update_layout(bargap=.25, margin=dict(b=70),
                      legend=dict(x=0, xanchor="left", y=-0.12, yanchor="top"))
    return _base(fig, 500, title or None)


# ═════════════════════════════════════════════════════════════════════════════
# Forecast & evaluation
# ═════════════════════════════════════════════════════════════════════════════

METHODS = [("mae_tft", "TFT (p50)"), ("mae_last_value", "repeat last value"), ("mae_recent_mean", "14-day mean")]


def backtest_dots(bt: pd.DataFrame) -> go.Figure:
    """Per community, the error of TFT and of each naive baseline on held-out days."""
    b = bt.sort_values("mae_tft", ascending=False)
    fig = go.Figure()
    lo = b[[c for c, _ in METHODS]].min(axis=1)
    hi = b[[c for c, _ in METHODS]].max(axis=1)
    for y, a, z in zip(b["subreddit"], lo, hi):
        fig.add_trace(go.Scatter(x=[a, z], y=[y, y], mode="lines", line=dict(color=T.GRID, width=2),
                                 showlegend=False, hoverinfo="skip"))
    for i, (col, name) in enumerate(METHODS):
        fig.add_trace(go.Scatter(x=b[col], y=b["subreddit"], mode="markers", name=name,
                                 marker=dict(size=10, color=T.SERIES[i], symbol=["circle", "square", "diamond"][i],
                                             line=dict(width=2, color=T.SURFACE)),
                                 hovertemplate="%{y}: MAE %{x:.4f}<extra>" + name + "</extra>"))
    fig.update_xaxes(title="mean absolute error on held-out days (lower is better)")
    fig.update_yaxes(tickfont=dict(size=10))
    return _base(fig, max(340, 18 * len(b) + 110), "Forecast error by community — TFT against naive baselines")


def backtest_across(summaries: dict[str, dict], labels: dict[str, str]) -> go.Figure:
    """Weighted MAE of each method, per corpus. Same three methods, same colours."""
    keys = [k for k in summaries if summaries[k]]
    fig = go.Figure()
    for i, (col, name) in enumerate(METHODS):
        fig.add_trace(go.Bar(
            x=[labels[k] for k in keys], y=[summaries[k][col] for k in keys], name=name,
            marker=dict(color=T.SERIES[i], line=dict(width=2, color=T.SURFACE)),
            hovertemplate="%{x}<br>" + name + ": MAE %{y:.4f}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=.28, bargroupgap=.06)
    fig.update_yaxes(title="weighted MAE, held-out days")
    fig.update_xaxes(tickfont=dict(size=11))
    return _base(fig, 380, "The same comparison on every corpus")


def small_multiples(daily: pd.DataFrame, fc: pd.DataFrame, subs: list[str], cols: int = 3,
                    history_days: int = 21) -> go.Figure:
    rows = int(np.ceil(len(subs) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subs, vertical_spacing=.12, horizontal_spacing=.06)
    for i, s in enumerate(subs):
        r, c = i // cols + 1, i % cols + 1
        d = daily[daily["subreddit"] == s].sort_values("date").tail(history_days)
        f = fc[fc["subreddit"] == s].sort_values("date")
        fig.add_trace(go.Scatter(x=d["date"], y=d["avg_toxicity"], mode="lines", line=dict(color=T.SERIES[0], width=2),
                                 showlegend=False, hovertemplate="%{x|%b %d}: %{y:.3f}<extra>" + s + "</extra>"), r, c)
        if not f.empty:
            fig.add_trace(go.Scatter(x=list(f["date"]) + list(f["date"])[::-1], y=list(f["p90"]) + list(f["p10"])[::-1],
                                     fill="toself", mode="lines", fillcolor=T.rgba(T.VIOLET, .25), line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"), r, c)
            fig.add_trace(go.Scatter(x=f["date"], y=f["p50"], mode="lines", line=dict(color=T.VIOLET, width=2, dash="dot"),
                                     showlegend=False, hovertemplate="%{x|%b %d}: p50 %{y:.3f}<extra>" + s + "</extra>"), r, c)
    fig.update_annotations(font=dict(size=12, color=T.INK_2))
    fig.update_xaxes(showticklabels=False)
    return _base(fig, 230 * rows + 40)


# ═════════════════════════════════════════════════════════════════════════════
# Clusters & corpora
# ═════════════════════════════════════════════════════════════════════════════

CLUSTER_SYMBOLS = ["circle", "square", "diamond", "cross", "x", "circle-open", "square-open", "diamond-open"]


def clusters_3d(cl: pd.DataFrame) -> go.Figure:
    """
    PCA position (two axes) plus toxicity as the third. Five clusters exceed the
    three colours that stay distinguishable in a scatter, so each cluster also
    has its own marker shape and a legend entry.
    """
    fig = go.Figure()
    for k in sorted(cl["cluster"].unique()):
        g = cl[cl["cluster"] == k]
        fig.add_trace(go.Scatter3d(
            x=g["pca_x"], y=g["pca_y"], z=g["toxicity_rate"], mode="markers", name=f"cluster {k}",
            text=g["subreddit"],
            marker=dict(size=7, color=T.SERIES[int(k) % len(T.SERIES)],
                        symbol=CLUSTER_SYMBOLS[int(k) % len(CLUSTER_SYMBOLS)], opacity=.9,
                        line=dict(width=1, color=T.SURFACE)),
            hovertemplate="<b>%{text}</b><br>PC1 %{x:.2f} · PC2 %{y:.2f}<br>toxicity %{z:.3f}<extra>cluster "
                          + str(k) + "</extra>"))
    ax = dict(backgroundcolor=T.SURFACE, gridcolor=T.GRID, zerolinecolor=T.AXIS, color=T.MUTED, showbackground=True)
    fig.update_layout(scene=dict(xaxis=dict(title="PC1", **ax), yaxis=dict(title="PC2", **ax),
                                 zaxis=dict(title="toxicity", **ax), bgcolor=T.SURFACE,
                                 camera=dict(eye=dict(x=1.6, y=1.5, z=.9))),
                      margin=dict(l=0, r=0, t=48, b=0))
    return _base(fig, 560, "Communities in health-profile space — drag to rotate")


def corpora_dots(table: pd.DataFrame, order: list[str], labels: dict[str, str]) -> go.Figure:
    """
    Communities that appear in more than one real corpus, with their mean
    toxicity in each. Colour = corpus, in registry order, the same everywhere.
    """
    shared = table.groupby("subreddit")["dataset"].nunique()
    subs = shared[shared >= 2].index
    t = table[table["subreddit"].isin(subs)]
    rank = t.groupby("subreddit")["toxicity"].mean().sort_values().index.tolist()
    fig = go.Figure()
    for s in rank:
        g = t[t["subreddit"] == s]
        fig.add_trace(go.Scatter(x=[g["toxicity"].min(), g["toxicity"].max()], y=[s, s], mode="lines",
                                 line=dict(color=T.GRID, width=3), showlegend=False, hoverinfo="skip"))
    for i, key in enumerate(order):
        g = t[t["dataset"] == key]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(x=g["toxicity"], y=g["subreddit"], mode="markers", name=labels[key],
                                 marker=dict(size=11, color=T.SERIES[i], symbol=["circle", "square", "diamond"][i % 3],
                                             line=dict(width=2, color=T.SURFACE)),
                                 hovertemplate="%{y}: %{x:.3f}<extra>" + labels[key] + "</extra>"))
    fig.update_yaxes(categoryorder="array", categoryarray=rank, tickfont=dict(size=10))
    fig.update_xaxes(title="mean toxicity in that corpus")
    return _base(fig, max(380, 20 * len(rank) + 120), "The same communities, measured three times")


def live_line(x: list, y: list) -> go.Figure:
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=T.SERIES[0], width=2, shape="spline"),
                               fill="tozeroy", fillcolor=T.rgba(T.SERIES[0], .12),
                               hovertemplate="comment %{x:,}: %{y:.1%}<extra>rolling toxic rate</extra>"))
    fig.update_yaxes(tickformat=".0%", title="rolling toxic rate", rangemode="tozero")
    fig.update_xaxes(title="comments processed")
    fig.update_layout(transition=dict(duration=0))
    return _base(fig, 240)
