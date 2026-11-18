"""
Dashboard tests: the chart builders on every corpus, and the real app headless.

The console is what an examiner sees, so a page that raises is a failed demo.
AppTest runs dashboard/app.py exactly as Streamlit would — sign-in, sidebar,
every page — without a browser, and fails on any exception a page raises.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))
warnings.filterwarnings("ignore", category=FutureWarning)

import apollo_datasets as AD  # noqa: E402
import charts as C  # noqa: E402
import data as D  # noqa: E402

KEYS = list(AD.DATASETS)


@pytest.fixture(params=KEYS)
def ds(request):
    return AD.get(request.param)


# ── data layer ──────────────────────────────────────────────────────────────

def test_meso_has_bands_and_volume(ds):
    m = D.meso(ds)
    assert {"band", "action", "volume"} <= set(m.columns)
    assert m["subreddit"].str.startswith("r/").all()
    assert (m["volume"] >= m["total_comments"]).all(), "pre-cap volume cannot be below what was analysed"


def test_waterfall_reconstructs_chi(ds):
    """The chart's steps must add back up to the CHI the pipeline reported."""
    for _, row in D.meso(ds).iterrows():
        steps = D.chi_waterfall(row)
        assert steps["value"].iloc[0] + steps["value"].iloc[1:-1].sum() == pytest.approx(
            steps["value"].iloc[-1], abs=0.05), row["subreddit"]


def test_phase_space_never_zero_fills(ds):
    ps = D.phase_space(D.daily(ds))
    assert ps[["level", "momentum"]].notna().all().all()
    assert not ps.empty


def test_torrent_backtest_matches_rizwans_own_evaluation():
    """Reducing forecast_seeds.csv must reproduce forecast_eval.json, seed 42."""
    import json
    ds = AD.get("real_torrent_2026_08")
    ev = json.loads(ds.file("forecast_eval.json").read_text())["per_seed"]["42"]
    s = D.backtest_summary(ds)
    assert s["mae_tft"] == pytest.approx(ev["mae_tft_p50"], abs=1e-6)
    assert s["mae_last_value"] == pytest.approx(ev["mae_persistence"], abs=1e-6)
    assert s["mae_recent_mean"] == pytest.approx(ev["mae_mean14"], abs=1e-6)
    assert s["beats_last"] == ev["communities_tft_beats_persistence"]
    assert s["beats_mean"] == ev["communities_tft_beats_mean14"]
    assert s["coverage"] == pytest.approx(ev["coverage_p10_p90"], abs=1e-6)


def test_cross_corpus_rank_agreement_reproduces_e10():
    """Rizwan's expectation E10 reported ρ = 0.916 over 14 shared communities."""
    t = D.across_corpora(AD.real())
    rho, n = D.rank_agreement(t, "real_api_2026_07", "real_torrent_2026_08")
    assert n == 14 and rho == pytest.approx(0.916, abs=0.001)


# ── chart builders ──────────────────────────────────────────────────────────

def test_every_chart_builds(ds):
    m, d, f, h = D.meso(ds), D.daily(ds), D.future(ds), D.held_out(ds)
    fc = f if not f.empty else h
    r = m.iloc[0]
    sub = r["subreddit"]
    figs = [
        C.landscape(m), C.band_strip(m), C.heatmap(d, m["subreddit"].tolist()),
        C.phase_space(D.phase_space(d), dict(zip(m["subreddit"], m["band"]))),
        C.gauge(float(r["community_health_index"]), r["band"]),
        C.waterfall(D.chi_waterfall(r)), C.profile(r, m.median(numeric_only=True), D.COMPONENTS),
        C.history(d[d["subreddit"] == sub], f[f["subreddit"] == sub] if not f.empty else None,
                  h[h["subreddit"] == sub] if not h.empty else None),
        C.backtest_dots(D.backtest(ds)), C.clusters_3d(D.clusters(ds)),
        C.small_multiples(d, fc, sorted(fc["subreddit"].unique())[:4]),
    ]
    for fig in figs:
        fig.to_json()


def test_no_chart_leaks_an_undefined_title(ds):
    """An untitled figure must not inherit the template's empty title ("undefined")."""
    m = D.meso(ds)
    fig = C.gauge(float(m.iloc[0]["community_health_index"]), m.iloc[0]["band"])
    assert fig.layout.title.text in ("", None) or fig.layout.title.text.strip()


def test_bands_never_rely_on_colour_alone():
    m = D.meso(AD.get("real_torrent_2026_08"))
    for trace in C.landscape(m).data:
        assert trace.marker.symbol is not None and trace.name.split()[-1] in {"CRITICAL", "HIGH", "MEDIUM", "LOW"}


def test_band_fill_traces_draw_no_markers():
    """Plotly adds markers to short traces unless told not to; a band is a fill."""
    ds = AD.get("real_torrent_2026_08")
    d, h = D.daily(ds), D.held_out(ds)
    sub = h["subreddit"].iloc[0]
    fig = C.history(d[d["subreddit"] == sub], None, h[h["subreddit"] == sub])
    fills = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
    assert fills and all(t.mode == "lines" for t in fills)


def test_live_seed_is_real_reddit_without_authors():
    seed = pd.read_csv(ROOT / "outputs" / "live_seed.csv")
    assert "author" not in seed.columns
    assert not seed["subreddit"].astype(str).str.endswith("auto").any(), "fabricated-corpus subreddits"
    assert seed["body"].astype(str).str.match(r"^RT @").mean() < 0.01, "looks like tweets"
    assert seed["toxicity"].between(0, 1).all()


def test_third_party_text_is_inert():
    """Reddit text must render as written, never as Markdown links or HTML."""
    import theme as T
    out = T.safe('[Message the mods](https://evil.example) <script>alert(1)</script> **bold** <img src=x>')
    assert "<script" not in out and "<img" not in out
    assert "](" not in out and "**" not in out
    assert "Message the mods" in out


# ── the app itself, headless ────────────────────────────────────────────────

PAGES = ["Overview", "Dynamics", "Communities", "Forecast", "🎯 Actions", "Clusters",
         "Across corpora", "Evidence", "🔴 Live", "🌐 Real Reddit", "Monitoring"]


@pytest.fixture(scope="module")
def app(monkeypatch_module):
    from streamlit.testing.v1 import AppTest
    import requests
    # The Monitoring page fetches the live /metrics endpoint; keep the test offline.
    monkeypatch_module.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    [b for b in at.button if "Sign in as demo" in b.label][0].click().run()
    assert not at.exception, at.exception
    return at


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.mark.parametrize("corpus", ["real_torrent_2026_08", "benchmark"])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, corpus, page):
    # Address the sidebar's widgets: several pages add their own selectbox, so
    # an index into the whole page would pick whichever rendered first.
    app.sidebar.selectbox[0].set_value(corpus)
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"{page} on {corpus}: {app.exception}"
    text = " ".join(m.value for m in app.markdown)
    assert "Traceback" not in text
