"""
Loaders for the console. Every file is reached through apollo_datasets, so a
page cannot read one corpus's forecast next to another corpus's meso report.

The three people who produced these corpora wrote slightly different files —
"r/" prefixes on some and not others, timezone-aware dates on some, a backtest
CSV in two datasets and per-seed rows in the third. Those differences are
reconciled here, once, rather than in each chart.

Pure functions of a Dataset; the app wraps them in st.cache_data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apollo_bands import ACTION, alert_band  # noqa: E402
from apollo_datasets import Dataset  # noqa: E402

COMPONENTS = [("toxicity_rate", "Toxicity", 35), ("polarization", "Polarisation", 30),
              ("churn_rate", "Author churn", 20), ("echo_chamber_index", "Echo chamber", 15)]


def _r(name: str) -> str:
    name = str(name)
    return name if name.startswith("r/") else f"r/{name}"


def _read(path: Path | None) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── per-community tables ────────────────────────────────────────────────────

def meso(ds: Dataset) -> pd.DataFrame:
    m = _read(ds.file("meso_report.csv"))
    if m.empty:
        return m
    m["subreddit"] = m["subreddit"].map(_r)
    m["band"] = m["community_health_index"].map(alert_band)
    m["action"] = m["band"].map(ACTION)
    m["penalty"] = 100 - m["community_health_index"]
    # Bubble size and "volume" should mean real activity. Where the corpus was
    # sampled with a per-day cap, total_comments is mostly cap x days (31,000
    # for almost every RC_2026-08 community), so prefer the pre-cap count the
    # filter recorded when there is one.
    m["volume"] = m["total_comments"]
    vol = _read(ds.file("filter_volume_by_subreddit.csv"))
    if not vol.empty and "eligible_before_cap" in vol:
        pre = vol.assign(subreddit=vol["subreddit"].map(_r)).set_index("subreddit")["eligible_before_cap"]
        m["volume"] = m["subreddit"].map(pre).fillna(m["total_comments"])
    return m.sort_values("community_health_index").reset_index(drop=True)


def clusters(ds: Dataset) -> pd.DataFrame:
    c = _read(ds.file("clusters_report.csv"))
    if not c.empty:
        c["subreddit"] = c["subreddit"].map(_r)
        c["band"] = c["community_health_index"].map(alert_band)
    return c


def daily(ds: Dataset) -> pd.DataFrame:
    d = _read(ds.daily)
    if d.empty:
        return d
    d["subreddit"] = d["subreddit"].map(_r)
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None).dt.normalize()
    return d.sort_values(["subreddit", "date"]).reset_index(drop=True)


# ── forecasts ───────────────────────────────────────────────────────────────

def future(ds: Dataset) -> pd.DataFrame:
    f = _read(ds.future_file)
    if not f.empty:
        f["subreddit"] = f["subreddit"].map(_r)
        f["date"] = pd.to_datetime(f["date"])
    return f


def held_out(ds: Dataset) -> pd.DataFrame:
    """Predictions of observed days, with what actually happened (RC_2026-08)."""
    if ds.forecast != "held_out":
        return pd.DataFrame()
    f = _read(ds.file("forecast_results.csv"))
    f["subreddit"] = f["subreddit"].map(_r)
    f["date"] = pd.to_datetime(f["date"])
    return f


def backtest(ds: Dataset) -> pd.DataFrame:
    """
    One row per community: TFT against repeating the last value and the
    recent mean, on days the model did not train on.

    Two datasets ship this as forecast_backtest.csv. RC_2026-08 ships the same
    information per seed and per day (forecast_seeds.csv), so it is reduced to
    the same shape here — canonical seed 42, as in Rizwan's forecast_eval.json.
    """
    bt = _read(ds.backtest_file)
    if not bt.empty:
        bt["subreddit"] = bt["subreddit"].map(_r)
        return bt
    seeds = _read(ds.file("forecast_seeds.csv"))
    if seeds.empty:
        return seeds
    s = seeds[(seeds["seed"] == 42) & seeds["out_of_sample"].astype(bool)]
    if "failed" in s:
        s = s[~s["failed"].astype(bool)]
    g = s.groupby("subreddit")
    out = pd.DataFrame({
        "n_days": g.size(),
        "mae_tft": g.apply(lambda x: (x["p50"] - x["actual"]).abs().mean()),
        "mae_last_value": g.apply(lambda x: (x["persistence"] - x["actual"]).abs().mean()),
        "mae_recent_mean": g.apply(lambda x: (x["mean14"] - x["actual"]).abs().mean()),
        "inside_p10_p90": g.apply(lambda x: ((x["actual"] >= x["p10"]) & (x["actual"] <= x["p90"])).mean()),
    }).reset_index()
    out["subreddit"] = out["subreddit"].map(_r)
    return out


def backtest_summary(ds: Dataset) -> dict:
    bt = backtest(ds)
    if bt.empty:
        return {}
    w = bt["n_days"]
    avg = lambda c: float((bt[c] * w).sum() / w.sum())
    return {
        "communities": int(len(bt)), "days": int(w.sum()),
        "mae_tft": avg("mae_tft"), "mae_last_value": avg("mae_last_value"),
        "mae_recent_mean": avg("mae_recent_mean"), "coverage": avg("inside_p10_p90"),
        "beats_last": int((bt["mae_tft"] < bt["mae_last_value"]).sum()),
        "beats_mean": int((bt["mae_tft"] < bt["mae_recent_mean"]).sum()),
    }


# ── derived views ───────────────────────────────────────────────────────────

def corpus_facts(ds: Dataset) -> dict:
    m = meso(ds)
    d = daily(ds)
    facts = {"communities": len(m), "comments": int(m["total_comments"].sum()) if not m.empty else 0,
             "days": int(d["date"].nunique()) if not d.empty else 0}
    prov = _json(ds.file("data_provenance.json"))
    for key in ("comments", "comments_scored"):
        if isinstance(prov.get(key), (int, float)):
            facts["comments"] = int(prov[key])
    return facts


def phase_space(d: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Where each community sits each day: its toxicity LEVEL (trailing mean) and
    its MOMENTUM (that mean now minus `window` days earlier). Top-right is
    toxic and getting worse — the quadrant the system exists to find early.

    Missing days stay missing: the series is put on a calendar and rolled with
    a minimum count, never zero-filled.
    """
    frames = []
    for sub, g in d.groupby("subreddit"):
        s = g.set_index("date")["avg_toxicity"].asfreq("D")
        level = s.rolling(window, min_periods=max(2, window - 2)).mean()
        frames.append(pd.DataFrame({"subreddit": sub, "date": level.index,
                                    "level": level.values, "momentum": (level - level.shift(window)).values}))
    out = pd.concat(frames, ignore_index=True).dropna()
    return out.sort_values(["date", "subreddit"]).reset_index(drop=True)


def across_corpora(datasets: list[Dataset]) -> pd.DataFrame:
    """Mean toxicity of every community in every real corpus it appears in."""
    rows = []
    for ds in datasets:
        m = meso(ds)
        for _, r in m.iterrows():
            rows.append({"subreddit": r["subreddit"], "dataset": ds.key, "label": ds.label,
                         "toxicity": float(r["toxicity_rate"]), "chi": float(r["community_health_index"])})
    return pd.DataFrame(rows)


def rank_agreement(table: pd.DataFrame, a: str, b: str) -> tuple[float, int]:
    """Spearman ρ of community toxicity between two corpora, over shared communities."""
    p = table.pivot_table(index="subreddit", columns="dataset", values="toxicity")
    if a not in p or b not in p:
        return float("nan"), 0
    both = p[[a, b]].dropna()
    if len(both) < 3:
        return float("nan"), len(both)
    return float(both[a].rank().corr(both[b].rank())), len(both)


def chi_waterfall(row: pd.Series) -> pd.DataFrame:
    """How a community's CHI is assembled: 100 minus four weighted penalties."""
    steps = [{"step": "Start", "value": 100.0}]
    for col, name, w in COMPONENTS:
        v = float(row.get(col, 0) or 0)
        steps.append({"step": f"{name} ×{w}", "value": -w * v, "input": v})
    steps.append({"step": "CHI", "value": float(row["community_health_index"])})
    return pd.DataFrame(steps)


def evidence(ds: Dataset) -> dict:
    """Every evaluation artifact a dataset ships, keyed by what it is."""
    base = ds.path
    out = {
        "validation": _json(base / "validation.json") if ds.kind == "benchmark" else {},
        "forecast_eval": _json(base / "forecast_eval.json"),
        "label_agreement": _json(base / "label_agreement.json"),
        "improvement": _json(base / "improvement" / "test_result.json"),
        "forecast_summary": _json(base / "forecast_summary.json"),
    }
    exp = base / "expectations_check.csv"
    out["expectations"] = pd.read_csv(exp) if exp.exists() else pd.DataFrame()
    return out


def metrics() -> dict:
    return _json(ROOT / "outputs" / "metrics.json")


def nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x
