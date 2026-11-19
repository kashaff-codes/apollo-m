"""
Live ingest: keep APOLLO-M current without anyone running a script.

    python scripts/live_ingest.py                 # one run: last 6 hours
    python scripts/live_ingest.py --hours 24      # a wider window (backfill)
    python scripts/live_ingest.py --budget 40     # stop fetching after 40 minutes

Runs on a schedule in GitHub Actions (.github/workflows/live-ingest.yml), every
six hours. Each run:

  1. fetch   the newest comments for every community in live/communities.txt
             from Arctic Shift (near real time; no credentials), sampled in
             two slices across the window so a busy hour does not crowd out
             the rest;
  2. score   them with unitary/toxic-bert, the model every corpus was scored
             with, after the same bot and boilerplate filter;
  3. pulse   append one row per community to outputs/live/pulse.csv: when,
             how many comments, mean and 90th-percentile toxicity, share above
             0.5, distinct authors. Bodies and usernames are never committed;
             authors are kept only as salted hashes in the run cache, for churn;
  4. health  recompute the Community Health Index over the last 7 days with
             the pipeline's own formula (apollo_chi): toxicity and author churn
             from the live data, polarisation and echo from the community's
             position in the SNAP hyperlink graph (structural, taken from the
             latest full pipeline run);
  5. forecast the next day for each community with the recent-mean baseline
             and an empirical 10-90% band. The backtests on all three real
             corpora found the recent mean more accurate than the TFT, so the
             live forecast uses it and says so;
  6. status  outputs/live/status.json: when it ran, what it fetched, what
             failed, how fresh the newest comment is.

Adding a community is one line in live/communities.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apollo_bands import alert_band  # noqa: E402
from apollo_chi import community_health_index  # noqa: E402

LIST = ROOT / "live" / "communities.txt"
OUT = ROOT / "outputs" / "live"
CACHE = ROOT / "data" / "live"          # git-ignored; persisted between runs by the Actions cache
API = "https://arctic-shift.photon-reddit.com/api/comments/search"
PER_SLICE = 100
SLICES = 2
REMOVED = {"[deleted]", "[removed]", ""}
WINDOW_DAYS = 7
STRUCTURE_SOURCES = ["real_torrent_2026_08", "real_arctic_2026_10", "real"]


def communities() -> list[str]:
    names = []
    for line in LIST.read_text(encoding="utf-8").splitlines():
        s = line.split("#", 1)[0].strip().removeprefix("r/")
        if s and s not in names:
            names.append(s)
    return names


# ── 1. fetch ─────────────────────────────────────────────────────────────────

def _get(sub: str, after: int, before: int) -> tuple[list[dict], str]:
    import requests
    err = ""
    for attempt in range(4):
        try:
            r = requests.get(API, params={"subreddit": sub, "after": after, "before": before,
                                          "limit": PER_SLICE, "sort": "desc",
                                          "fields": "id,author,body,created_utc"},
                             timeout=(15, 90))
            if r.status_code == 200:
                body = r.json()
                if body.get("error"):
                    err = str(body["error"])[:60]
                else:
                    return body.get("data") or [], ""
            else:
                err = f"http {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - flaky upstream; retry with backoff
            err = type(exc).__name__
        time.sleep(min(45, 4 * 2 ** attempt) + random.random() * 2)
    return [], err


def fetch(subs: list[str], now: int, hours: float, budget_s: float, workers: int = 3):
    """Newest comments per community, `SLICES` evenly spaced slices across the window."""
    span = int(hours * 3600)
    jobs = [(s, now - span + i * span // SLICES, now - span + (i + 1) * span // SLICES)
            for i in range(SLICES) for s in subs]
    # Breadth first: every community's most recent slice before any older one.
    jobs.sort(key=lambda j: -j[2])
    started = time.time()
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for job in jobs:
            futures[pool.submit(_get, *job)] = job
        for fut in as_completed(futures):
            sub, a, b = futures[fut]
            data, err = fut.result()
            if err:
                failed.append({"subreddit": sub, "after": a, "error": err})
            for c in data:
                rows.append({"subreddit": sub, "id": c.get("id"), "author": str(c.get("author") or ""),
                             "body": str(c.get("body") or "").strip(),
                             "created_utc": int(c.get("created_utc") or 0)})
            if time.time() - started > budget_s:
                for f in futures:
                    f.cancel()
                print(f"  fetch budget reached after {len(rows)} comments", flush=True)
                break
    df = pd.DataFrame(rows, columns=["subreddit", "id", "author", "body", "created_utc"])
    return df.drop_duplicates("id"), failed


# ── 2. score ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df["body"].isin(REMOVED) & ~df["author"].isin(["[deleted]", "[removed]"])]
    df = df[df["body"].str.len() >= 4]
    try:
        from main import drop_bots_and_boilerplate  # the pipeline's own filter
        return drop_bots_and_boilerplate(df.copy())
    except Exception:  # noqa: BLE001 - a minimal copy if main.py cannot import here
        bots = df["author"].str.lower().str.contains(r"bot$|automoderator", regex=True)
        return df[~bots]


def score(texts: list[str]) -> list[float]:
    if not texts:
        return []
    from transformers import pipeline
    clf = pipeline("text-classification", model="unitary/toxic-bert", top_k=None, device=-1)
    out: list[float] = []
    for i in range(0, len(texts), 256):
        batch = [t[:512] for t in texts[i:i + 256]]
        for res in clf(batch, batch_size=32, truncation=True, max_length=256):
            out.append(next((x["score"] for x in res if x["label"] == "toxic"), 0.0))
    return out


# ── 3-6. pulse, health, forecast, status ─────────────────────────────────────

def _salt() -> str:
    path = CACHE / "salt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(os.urandom(16).hex())
    return path.read_text().strip()


def remember_authors(df: pd.DataFrame, run_at: datetime) -> None:
    """Salted author hashes per community and day, for churn. Never committed."""
    salt = _salt()
    store = CACHE / "authors"
    store.mkdir(parents=True, exist_ok=True)
    for (sub, day), g in df.assign(day=pd.to_datetime(df["created_utc"], unit="s").dt.date.astype(str)) \
            .groupby(["subreddit", "day"]):
        path = store / f"{day}.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        seen = set(data.get(sub, []))
        seen |= {hashlib.sha256((salt + a).encode()).hexdigest()[:12] for a in g["author"]}
        data[sub] = sorted(seen)
        path.write_text(json.dumps(data))
    cutoff = (run_at - timedelta(days=2 * WINDOW_DAYS + 2)).date().isoformat()
    for p in store.glob("*.json"):
        if p.stem < cutoff:
            p.unlink()


def churn(sub: str, run_at: datetime) -> float | None:
    """Share of the earlier half-window's authors not seen in the later half: the
    pipeline's definition (MesoLayer splits the window at its midpoint)."""
    store = CACHE / "authors"
    days = [(run_at - timedelta(days=d)).date().isoformat() for d in range(WINDOW_DAYS)]
    late, early = set(), set()
    for i, d in enumerate(days):
        p = store / f"{d}.json"
        if p.exists():
            (late if i < WINDOW_DAYS // 2 else early).update(json.loads(p.read_text()).get(sub, []))
    if len(early) < 20:
        return None
    return round(len(early - late) / len(early), 4)


def structure() -> pd.DataFrame:
    """Polarisation and echo per community from the latest full pipeline runs."""
    frames = []
    for key in STRUCTURE_SOURCES:
        p = ROOT / "outputs" / {"real": "real"}.get(key, key) / "meso_report.csv"
        if p.exists():
            frames.append(pd.read_csv(p, usecols=["subreddit", "polarization", "echo_chamber_index",
                                                  "churn_rate"]))
    s = pd.concat(frames).drop_duplicates("subreddit") if frames else pd.DataFrame(
        columns=["subreddit", "polarization", "echo_chamber_index", "churn_rate"])
    s["subreddit"] = s["subreddit"].astype(str).str.removeprefix("r/")
    return s.set_index("subreddit")


def health(pulse: pd.DataFrame, run_at: datetime) -> pd.DataFrame:
    recent = pulse[pd.to_datetime(pulse["run_at"], utc=True) > run_at - timedelta(days=WINDOW_DAYS)]
    st = structure()
    pol_med = float(st["polarization"].median()) if len(st) else 0.2
    echo_med = float(st["echo_chamber_index"].median()) if len(st) else 0.2
    churn_med = float(st["churn_rate"].median()) if len(st) else 0.8
    rows = []
    for sub, g in recent.groupby("subreddit"):
        n = int(g["comments"].sum())
        if n < 30:   # the pipeline's MIN_COMMENTS_PER_SUB
            continue
        tox = float(np.average(g["mean_toxicity"], weights=g["comments"]))
        c = churn(sub, run_at)
        # Churn needs a week of live author history. Until then, use the value
        # the full pipeline last measured for this community (or the median),
        # and say so: counting it as zero inflated every early CHI by ~15 points.
        churn_from = "live"
        if c is None:
            churn_from = "corpus" if sub in st.index else "median"
            c = float(st.loc[sub, "churn_rate"]) if sub in st.index else churn_med
        pol = float(st.loc[sub, "polarization"]) if sub in st.index else pol_med
        echo = float(st.loc[sub, "echo_chamber_index"]) if sub in st.index else echo_med
        chi = community_health_index(tox, pol, c, echo)
        rows.append({"subreddit": f"r/{sub}", "community_health_index": chi,
                     "alert": alert_band(chi), "toxicity_rate": round(tox, 4),
                     "toxicity_p90": round(float(g["p90_toxicity"].max()), 4),
                     "share_toxic": round(float(np.average(g["share_toxic"], weights=g["comments"])), 4),
                     "churn_rate": round(c, 4), "churn_from": churn_from, "polarization": round(pol, 4), "echo_chamber_index": round(echo, 4),
                     "structure_from": "snap" if sub in st.index else "median",
                     "total_comments": n, "runs": int(len(g)),
                     "last_seen": g["run_at"].max()})
    return pd.DataFrame(rows).sort_values("community_health_index") if rows else pd.DataFrame()


def daily_and_forecast(pulse: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pulse.assign(date=pd.to_datetime(pulse["run_at"], utc=True).dt.date.astype(str))
    daily = (p.groupby(["subreddit", "date"])
              .apply(lambda g: pd.Series({
                  "avg_toxicity": float(np.average(g["mean_toxicity"], weights=g["comments"])),
                  "comment_count": int(g["comments"].sum())}), include_groups=False)
              .reset_index())
    daily["subreddit"] = "r/" + daily["subreddit"]
    fc = []
    for sub, g in daily.groupby("subreddit"):
        y = g.sort_values("date")["avg_toxicity"].to_numpy()
        if len(y) < 3:
            continue
        mean = float(y[-7:].mean())
        resid = y[1:] - pd.Series(y).rolling(7, min_periods=1).mean().shift(1).to_numpy()[1:]
        lo, hi = (np.quantile(resid, [0.1, 0.9]) if len(resid) >= 5 else (-0.02, 0.02))
        nxt = (pd.Timestamp(g["date"].max()) + pd.Timedelta(days=1)).date().isoformat()
        # On a trending series both residual quantiles can sit on one side of
        # the mean; the band must still contain the point forecast.
        lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
        fc.append({"subreddit": sub, "date": nxt, "method": "recent_mean_7d",
                   "p10": round(mean + lo, 4), "p50": round(mean, 4), "p90": round(mean + hi, 4),
                   "history_days": int(len(y))})
    return daily, pd.DataFrame(fc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--budget", type=float, default=40.0, help="minutes allowed for fetching")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-fetch", action="store_true", help="recompute outputs from pulse.csv only")
    args = ap.parse_args()

    run_at = datetime.now(timezone.utc).replace(microsecond=0)
    now = int(run_at.timestamp())
    subs = communities()
    print(f"live ingest {run_at.isoformat()} — {len(subs)} communities, last {args.hours:g}h", flush=True)

    if args.no_fetch:
        prev = json.loads((OUT / "status.json").read_text()) if (OUT / "status.json").exists() else {}
        pulse = pd.read_csv(OUT / "pulse.csv")
        run_at = pd.to_datetime(pulse["run_at"], utc=True).max().to_pydatetime()
        health(pulse, run_at).to_csv(OUT / "meso_report.csv", index=False)
        daily, fc = daily_and_forecast(pulse)
        daily.to_csv(OUT / "daily_series.csv", index=False)
        if len(fc):
            fc.to_csv(OUT / "forecast_future.csv", index=False)
        print("recomputed from pulse.csv;", prev.get("run_at"))
        return
    raw, failed = fetch(subs, now, args.hours, args.budget * 60, args.workers)
    df = clean(raw) if len(raw) else raw
    print(f"  fetched {len(raw)} comments, {len(df)} after filtering, {len(failed)} slices failed", flush=True)
    df = df.assign(toxicity=score(df["body"].tolist())) if len(df) else df.assign(toxicity=[])

    OUT.mkdir(parents=True, exist_ok=True)
    pulse_path = OUT / "pulse.csv"
    pulse = pd.read_csv(pulse_path) if pulse_path.exists() else pd.DataFrame()
    if len(df):
        remember_authors(df, run_at)
        agg = df.groupby("subreddit").agg(
            comments=("toxicity", "size"), mean_toxicity=("toxicity", "mean"),
            p90_toxicity=("toxicity", lambda s: float(np.quantile(s, 0.9))),
            share_toxic=("toxicity", lambda s: float((s > 0.5).mean())),
            authors=("author", "nunique"),
            newest_utc=("created_utc", "max"), oldest_utc=("created_utc", "min"),
        ).reset_index()
        agg.insert(0, "run_at", run_at.isoformat())
        agg["window_hours"] = args.hours
        for c in ("mean_toxicity", "p90_toxicity", "share_toxic"):
            agg[c] = agg[c].round(4)
        pulse = pd.concat([pulse, agg], ignore_index=True)
        pulse.to_csv(pulse_path, index=False)

    report = health(pulse, run_at) if len(pulse) else pd.DataFrame()
    if len(report):
        report.to_csv(OUT / "meso_report.csv", index=False)
    if len(pulse):
        daily, fc = daily_and_forecast(pulse)
        daily.to_csv(OUT / "daily_series.csv", index=False)
        if len(fc):
            fc.to_csv(OUT / "forecast_future.csv", index=False)

    newest = int(df["created_utc"].max()) if len(df) else None
    status = {
        "run_at": run_at.isoformat(),
        "run_at_ms": int(run_at.timestamp() * 1000),   # for Grafana's "x hours ago"
        "window_hours": args.hours,
        "communities_tracked": len(subs),
        "communities_with_data": int(df["subreddit"].nunique()) if len(df) else 0,
        "comments_this_run": int(len(df)),
        "comments_total": int(pulse["comments"].sum()) if len(pulse) else 0,
        "runs_total": int(pulse["run_at"].nunique()) if len(pulse) else 0,
        "first_run": str(pulse["run_at"].min()) if len(pulse) else None,
        "newest_comment_utc": newest,
        "newest_comment_age_min": round((now - newest) / 60, 1) if newest else None,
        "slices_failed": len(failed),
        "failures": failed[:20],
        "scorer": "unitary/toxic-bert",
        "chi_window_days": WINDOW_DAYS,
        "forecast_method": "recent_mean_7d (beat the TFT in every real-corpus backtest)",
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps({k: status[k] for k in ("comments_this_run", "comments_total",
                                              "communities_with_data", "slices_failed",
                                              "newest_comment_age_min")}), flush=True)


if __name__ == "__main__":
    main()
