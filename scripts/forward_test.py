"""
Forward test: forecasts of days that had not happened when the model was trained.

    python scripts/forward_test.py forecast   # train on August only, predict Sep 1-5 (+ baselines)
    python scripts/forward_test.py fetch      # real comments for those days, from Arctic Shift
    python scripts/forward_test.py score      # unitary/toxic-bert, as the corpora were scored
    python scripts/forward_test.py evaluate   # forecast error against what actually happened
    python scripts/forward_test.py all

Why
---
Every forecast evaluation in the project so far scores the model on days that
already existed when it ran: the last five days of a series, held out of
training. That says how well it interpolates a window it has seen around; it
does not say whether it can forecast. In every real corpus the TFT lost those
held-out tests to repeating the recent 14-day mean.

This asks the question properly, on two corpora:

  * RC_2026-08 (Rizwan). A TFT trained only on August predicts 1-5 September.
    Settings are his (lookback 14, horizon 5, 6 epochs, CPU), seeds 42/43/44.
  * Arctic Shift rebuild (Aaqib). His committed forecasts for 16-20 November,
    made from data ending 15 November, scored on the days that are complete.

Baselines are fixed from the training data alone: repeat the last observed day,
and the mean of the last 14 observed days.

How the actual values are measured
----------------------------------
Each community-day is sampled in four 6-hour windows (the first 100 comments
after 00:00, 06:00, 12:00 and 18:00 UTC), scored by toxic-bert exactly as the
corpora were (first 512 characters, 256 tokens, P(toxic)), and averaged. A day
counts only with at least 50 scored comments — Rizwan's coverage rule. Days
that Arctic Shift has no data for stay missing; they are never zero-filled.

The training corpora were sampled across whole days, so a four-window sample
is a cheaper estimate of the same quantity, not the same estimator. Its
sampling error adds to every method's error alike; the comparison between
methods, which is the point, is paired on identical days.

Outputs (aggregates only): outputs/forward_test/. Raw comments and usernames
stay in data/forward_test/, which is git-ignored.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from apollo_datasets import get as get_dataset  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
OUT = ROOT / "outputs" / "forward_test"
RAW = ROOT / "data" / "forward_test"
WINDOWS = (0, 6, 12, 18)
PER_WINDOW = 100
MIN_COMMENTS = 50
REMOVED = {"", "[deleted]", "[removed]", "[ Removed by Reddit ]"}
SEEDS = (42, 43, 44)

CORPORA = {
    "rc_2026_08": {"dataset": "real_torrent_2026_08", "days": [f"2026-09-0{d}" for d in range(1, 6)],
                   "min_chars": 0},
    "arctic_2026_10": {"dataset": "real_arctic_2026_10", "days": ["2026-11-16", "2026-11-17"],
                       "min_chars": 4},
}


def _bare(s: str) -> str:
    return str(s).removeprefix("r/")


def _history(key: str) -> pd.DataFrame:
    ds = get_dataset(CORPORA[key]["dataset"])
    d = pd.read_csv(ds.daily)
    d["subreddit"] = d["subreddit"].map(_bare)
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None).dt.normalize()
    if "meets_min_comments" in d:          # Rizwan's coverage rule, as he applied it
        d = d[d["meets_min_comments"].astype(bool)]
    return d.dropna(subset=["avg_toxicity"]).sort_values(["subreddit", "date"])


# ── 1. forecasts, made from the training data only ──────────────────────────

def baselines(hist: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    rows = []
    for sub, g in hist.groupby("subreddit"):
        last, mean14 = float(g["avg_toxicity"].iloc[-1]), float(g["avg_toxicity"].tail(14).mean())
        for h, day in enumerate(days, 1):
            rows += [{"subreddit": sub, "date": day, "horizon": h, "method": "last_value", "p50": last},
                     {"subreddit": sub, "date": day, "horizon": h, "method": "mean14", "p50": mean14}]
    return pd.DataFrame(rows)


def forecast() -> None:
    import lightning.pytorch as pl
    from models.tft_forecaster import TFTConfig, TFTForecaster

    OUT.mkdir(parents=True, exist_ok=True)
    frames = []

    # RC_2026-08: train on all of August, forecast September.
    hist = _history("rc_2026_08")
    days = CORPORA["rc_2026_08"]["days"]
    assert hist["date"].max() < pd.Timestamp(days[0]), "training data reaches into the forecast window"
    for seed in SEEDS:
        pl.seed_everything(seed, workers=True)
        f = TFTForecaster(TFTConfig(max_epochs=6, lookback_window=14, forecast_horizon=5,
                                    use_comment_count=False))
        f.train(hist[["subreddit", "date", "avg_toxicity", "toxic_rate", "comment_count"]])
        for label in f.eligible_subreddits(hist):
            res = f.predict_quantiles(hist[hist["subreddit"] == label[2:]], subreddit=label, future=True)
            if not res or res.get("subreddit") != label:
                continue
            for h, day in enumerate(days, 1):
                frames.append({"corpus": "rc_2026_08", "subreddit": label[2:], "date": day, "horizon": h,
                               "method": f"tft_seed{seed}", "p10": float(res["p10"][h - 1]),
                               "p50": float(res["p50"][h - 1]), "p90": float(res["p90"][h - 1])})
        print(f"rc_2026_08 seed {seed}: forecasts for "
              f"{len({r['subreddit'] for r in frames if r['method'] == f'tft_seed{seed}'})} communities")
    b = baselines(hist, days).assign(corpus="rc_2026_08")
    trained_until = {"rc_2026_08": str(hist["date"].max().date())}

    # Arctic Shift rebuild: Aaqib's committed future forecasts, unchanged.
    ds = get_dataset("real_arctic_2026_10")
    fa = pd.read_csv(ds.file("forecast_results.csv"))
    fa["subreddit"] = fa["subreddit"].map(_bare)
    fa = fa[fa["date"].isin(CORPORA["arctic_2026_10"]["days"])]
    tft_a = fa.assign(corpus="arctic_2026_10", horizon=fa["day"], method="tft_committed")[
        ["corpus", "subreddit", "date", "horizon", "method", "p10", "p50", "p90"]]
    hist_a = _history("arctic_2026_10")
    hist_a = hist_a[hist_a["subreddit"].isin(set(fa["subreddit"]))]
    b_a = baselines(hist_a, sorted(set(fa["date"]))).assign(corpus="arctic_2026_10")
    trained_until["arctic_2026_10"] = str(hist_a["date"].max().date())

    out = pd.concat([pd.DataFrame(frames), b, tft_a, b_a], ignore_index=True)
    out.to_csv(OUT / "forecasts.csv", index=False)
    (OUT / "forecasts.meta.json").write_text(json.dumps({
        "made_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_until": trained_until, "seeds": list(SEEDS),
        "note": "Written before any comment from the forecast days was fetched.",
    }, indent=2))
    print(f"wrote {len(out)} forecast rows -> {OUT / 'forecasts.csv'}")


# ── 2. fetch ────────────────────────────────────────────────────────────────

def _jobs() -> list[tuple[str, str, str, int]]:
    fc = pd.read_csv(OUT / "forecasts.csv")
    jobs = []
    for key, cfg in CORPORA.items():
        subs = sorted(set(fc.loc[fc["corpus"] == key, "subreddit"]))
        jobs += [(key, s, d, h) for s in subs for d in cfg["days"] for h in WINDOWS]
    # Breadth first: one window for every community-day before a second for any.
    # Arctic Shift throttles under load, so a long fetch may not finish; ordered
    # this way, whatever has arrived covers every community-day evenly, and each
    # later window only tightens the estimate. One window of 100 comments already
    # clears the 50-comment rule.
    priority = {12: 0, 0: 1, 18: 2, 6: 3}
    return sorted(jobs, key=lambda j: (priority[j[3]], j[0] != "rc_2026_08", j[2], j[1]))


def _fetch_one(job) -> str:
    import requests
    key, sub, day, hour = job
    path = RAW / "raw" / key / f"{sub}_{day}_{hour:02d}.json"
    if path.exists():
        return "cached"
    start = int(datetime.fromisoformat(f"{day}T00:00:00+00:00").timestamp()) + hour * 3600
    for attempt in range(5):
        try:
            r = requests.get(API, params={"subreddit": sub, "after": start, "before": start + 6 * 3600,
                                          "limit": PER_WINDOW, "sort": "asc"}, timeout=(15, 150))
            if r.status_code == 200:
                data = r.json().get("data") or []
                keep = [{k: c.get(k) for k in ("id", "author", "body", "created_utc", "subreddit")} for c in data]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(keep), encoding="utf-8")
                return f"ok {len(keep)}"
            err = f"http {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - the service is flaky; retry with backoff
            err = type(exc).__name__
        time.sleep(min(60, 5 * 2 ** attempt) + random.random() * 3)
    return f"failed ({err})"


def fetch(workers: int = 4) -> None:
    jobs = _jobs()
    todo = [j for j in jobs if not (RAW / "raw" / j[0] / f"{j[1]}_{j[2]}_{j[3]:02d}.json").exists()]
    print(f"{len(jobs)} windows, {len(todo)} still to fetch, {workers} at a time", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_fetch_one, j) for j in todo]):
            done += 1
            failed += fut.result().startswith("failed")
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} fetched, {failed} failed", flush=True)


# ── 3. score ────────────────────────────────────────────────────────────────

def score() -> None:
    from transformers import pipeline

    from main import drop_bots_and_boilerplate
    rows = []
    for key in CORPORA:
        for p in sorted((RAW / "raw" / key).glob("*.json")):
            sub, day, _ = p.stem.rsplit("_", 2)
            for c in json.loads(p.read_text(encoding="utf-8")):
                body = str(c.get("body") or "").strip()
                if body in REMOVED or str(c.get("author")) in ("[deleted]", "[removed]"):
                    continue
                if len(body) < CORPORA[key]["min_chars"]:
                    continue
                rows.append({"corpus": key, "subreddit": sub, "date": day, "id": c["id"],
                             "author": str(c.get("author", "")), "body": body})
    df = pd.DataFrame(rows).drop_duplicates(["corpus", "id"])
    df = pd.concat([drop_bots_and_boilerplate(g) for _, g in df.groupby("corpus")], ignore_index=True)
    cache = RAW / "scored.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["corpus", "id", "toxicity_score"])
    todo = df[~(df["corpus"] + df["id"]).isin(done["corpus"] + done["id"])]
    print(f"{len(df)} comments after filtering; {len(todo)} to score", flush=True)
    if len(todo):
        clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=-1)
        scores = []
        texts = [b[:512] for b in todo["body"]]
        for i in range(0, len(texts), 256):
            for out in clf(texts[i:i + 256], batch_size=32, truncation=True, max_length=256):
                scores.append(next((x["score"] for x in out if x["label"] == "toxic"), 0.0))
            print(f"  scored {min(i + 256, len(texts))}/{len(texts)}", flush=True)
        new = todo[["corpus", "id"]].assign(toxicity_score=scores)
        done = pd.concat([done, new], ignore_index=True)
        done.to_csv(cache, index=False)
    scored = df.merge(done, on=["corpus", "id"])
    scored[["corpus", "subreddit", "date", "id", "toxicity_score"]].to_csv(RAW / "scored_comments.csv", index=False)


# ── 4. evaluate ─────────────────────────────────────────────────────────────

def evaluate(boot: int = 5000, seed: int = 2026) -> dict:
    s = pd.read_csv(RAW / "scored_comments.csv")
    actual = (s.groupby(["corpus", "subreddit", "date"])["toxicity_score"]
                .agg(actual="mean", n_comments="count").reset_index())
    actual["counts"] = actual["n_comments"] >= MIN_COMMENTS
    actual.to_csv(OUT / "actual_daily.csv", index=False)

    fc = pd.read_csv(OUT / "forecasts.csv")
    tft_methods = sorted(m for m in fc["method"].unique() if m.startswith("tft"))
    fc_mean = (fc[fc["method"].isin(tft_methods)].groupby(["corpus", "subreddit", "date", "horizon"])
               [["p10", "p50", "p90"]].mean().reset_index().assign(method="tft_mean"))
    fc = pd.concat([fc, fc_mean], ignore_index=True)
    m = fc.merge(actual[actual["counts"]], on=["corpus", "subreddit", "date"])
    m["abs_err"] = (m["p50"] - m["actual"]).abs()
    m.to_csv(OUT / "results.csv", index=False)

    rng = np.random.default_rng(seed)
    summary = {"min_comments_per_day": MIN_COMMENTS, "windows_utc": list(WINDOWS), "per_window": PER_WINDOW}
    for corpus, g in m.groupby("corpus"):
        tft = "tft_mean" if corpus == "rc_2026_08" else "tft_committed"
        piv = g.pivot_table(index=["subreddit", "date"], columns="method", values="abs_err")
        piv = piv.dropna(subset=[tft, "last_value", "mean14"])
        diff = (piv[tft] - piv["mean14"]).to_numpy()
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(boot)] if len(diff) else [np.nan]
        cov = g[(g["method"] == tft)].assign(inside=lambda x: (x["actual"] >= x["p10"]) & (x["actual"] <= x["p90"]))
        per_comm = piv.groupby(level=0).mean()
        summary[corpus] = {
            "community_days": int(len(piv)), "communities": int(piv.index.get_level_values(0).nunique()),
            "days": sorted(set(piv.index.get_level_values(1))),
            "mae": {k: round(float(piv[k].mean()), 5) for k in piv.columns},
            "tft_minus_mean14": {"mean": round(float(np.mean(diff)), 5),
                                 "ci95": [round(float(np.percentile(bs, 2.5)), 5),
                                          round(float(np.percentile(bs, 97.5)), 5)]},
            "tft_beats_mean14_communities": int((per_comm[tft] < per_comm["mean14"]).sum()),
            "tft_beats_last_value_communities": int((per_comm[tft] < per_comm["last_value"]).sum()),
            "coverage_p10_p90": round(float(cov["inside"].mean()), 4) if len(cov) else None,
            "mae_by_horizon": {int(h): {k: round(float(v), 5) for k, v in
                                        gg.groupby("method")["abs_err"].mean().items()
                                        if k in (tft, "last_value", "mean14")}
                               for h, gg in g.groupby("horizon")},
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["forecast", "fetch", "score", "evaluate", "all"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.step in ("forecast", "all"):
        forecast()
    if a.step in ("fetch", "all"):
        fetch(a.workers)
    if a.step in ("score", "all"):
        score()
    if a.step in ("evaluate", "all"):
        evaluate()
