"""
Build a properly sampled APOLLO-M corpus from the Arctic Shift CSV.

Steps:
1. Load reddit_live_sample.csv, clean and dedupe
2. Find longest contiguous window (>=24 days) across all subreddits
3. Cap at N comments per subreddit per day using reservoir sampling (seed 42)
4. Write apollo_comments_real.csv and a summary JSON
5. Never touches apollo_daily.csv or ground_truth.json
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

DATA = Path("data")
REMOVED = {"[deleted]", "[removed]", "[ Removed by Reddit ]", ""}

def main(cap: int = 1000, seed: int = 42, min_days: int = 24):
    print("Loading reddit_live_sample.csv...")
    df = pd.read_csv(DATA / "reddit_live_sample.csv")
    raw_count = len(df)
    print(f"  Raw rows: {raw_count:,}")

    # Step 1: clean
    drop_counts = Counter()

    # drop missing fields FIRST: astype(str) would turn a null subreddit into the string "nan"
    before = len(df)
    df = df.dropna(subset=["id", "subreddit", "author", "body", "score", "created_utc"])
    drop_counts["missing_fields"] = before - len(df)

    # normalize subreddit
    df["subreddit"] = df["subreddit"].astype(str).str.replace(r"^r/", "", regex=True).str.strip()

    # drop deleted/removed
    before = len(df)
    df["body"] = df["body"].astype(str).str.strip()
    df["author"] = df["author"].astype(str).str.strip()
    df = df[~df["body"].isin(REMOVED)]
    df = df[~df["author"].isin(["[deleted]", "[removed]"])]
    drop_counts["deleted_removed"] = before - len(df)

    # dedupe by id
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    drop_counts["duplicate_id"] = before - len(df)

    print(f"  After cleaning: {len(df):,} rows")

    # Step 2: parse dates
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["created_utc"])
    drop_counts["bad_timestamp"] = before - len(df)   # counted, so the reconciliation adds up
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.date

    # find contiguous window
    all_dates = sorted(df["date"].unique())
    print(f"  Date range: {all_dates[0]} to {all_dates[-1]}")

    # find longest run of consecutive dates
    best_start, best_len = all_dates[0], 1
    cur_start, cur_len = all_dates[0], 1
    for i in range(1, len(all_dates)):
        if (all_dates[i] - all_dates[i-1]).days == 1:
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_start = all_dates[i]
            cur_len = 1

    best_end = best_start + timedelta(days=best_len - 1)
    print(f"  Longest contiguous window: {best_start} to {best_end} ({best_len} days)")

    if best_len < min_days:
        # Never fall back to a window with gaps: that breaks the contiguity guarantee.
        raise SystemExit(
            f"No contiguous window of >= {min_days} days found (longest is {best_len}). "
            "Refusing to build a corpus with date gaps."
        )

    # filter to window
    df = df[(df["date"] >= best_start) & (df["date"] <= best_end)]
    print(f"  Rows in window: {len(df):,}")

    # Step 3: reservoir sampling per subreddit per day
    rng = random.Random(seed)
    reservoirs: dict[tuple, list] = defaultdict(list)
    eligible: Counter = Counter()

    for row in df.itertuples(index=False):
        key = (row.subreddit, str(row.date))
        eligible[key] += 1
        rec = {
            "id": row.id,
            "subreddit": row.subreddit,
            "author": row.author,
            "body": row.body,
            "score": row.score,
            "created_utc": int(row.created_utc),
        }
        res = reservoirs[key]
        if cap <= 0 or len(res) < cap:
            res.append(rec)
        else:
            j = rng.randrange(eligible[key])
            if j < cap:
                res[j] = rec

    # build output
    records = []
    for key in sorted(reservoirs):
        for rec in sorted(reservoirs[key], key=lambda r: r["created_utc"]):
            records.append(rec)

    out_df = pd.DataFrame(records)
    # Date per row for the coverage check only (kept out of out_df so the CSV schema is unchanged).
    out_dates = pd.to_datetime(out_df["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    out_df["group_id"] = out_df["subreddit"]
    out_df["toxicity_score"] = float("nan")
    out_df["is_toxic"] = 0
    out_df["toxicity_label"] = 2

    # check subreddit coverage
    window_dates = set(str(best_start + timedelta(days=i)) for i in range((best_end - best_start).days + 1))
    subs_with_gaps = []
    subs_complete = []
    for sub in out_df["subreddit"].unique():
        sub_dates = set(out_dates[out_df["subreddit"] == sub].unique())
        if window_dates - sub_dates:
            subs_with_gaps.append(sub)
        else:
            subs_complete.append(sub)

    print(f"\n  Subreddits with complete coverage ({len(subs_complete)}): OK for forecasting")
    print(f"  Subreddits with gaps ({len(subs_with_gaps)}): kept; the forecaster keeps real calendar gaps and needs >=60% of days observed")
    if subs_with_gaps:
        print(f"    Gaps in: {', '.join(subs_with_gaps[:10])}")

    # save
    out_path = DATA / "apollo_comments_real.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Saved {len(out_df):,} rows to {out_path}")

    # summary JSON
    per_sub_day = {}
    for (sub, day), res in reservoirs.items():
        per_sub_day[f"{sub}|{day}"] = {
            "eligible": eligible[(sub, day)],
            "kept": len(res),
            "hit_cap": bool(cap > 0 and eligible[(sub, day)] > cap)
        }

    from collections import Counter as C
    sub_counts = C()
    for (sub, _), res in reservoirs.items():
        sub_counts[sub] += len(res)

    summary = {
        "raw_rows": raw_count,
        "drop_counts": dict(drop_counts),
        "rows_in_window": len(df),
        "rows_kept": len(out_df),
        "window_start": str(best_start),
        "window_end": str(best_end),
        "window_days": (best_end - best_start).days + 1,
        "cap_per_sub_per_day": cap,
        "seed": seed,
        "subreddits_complete": sorted(subs_complete),
        "subreddits_with_gaps": sorted(subs_with_gaps),
        "kept_per_subreddit": dict(sorted(sub_counts.items())),
        "per_subreddit_day": per_sub_day,
    }

    summary_path = DATA / "corpus_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary saved to {summary_path}")
    print(f"\nReconciliation:")
    print(f"  Raw:          {raw_count:,}")
    print(f"  After clean:  {raw_count - sum(drop_counts.values()):,}")
    print(f"  In window:    {len(df):,}")
    print(f"  After cap:    {len(out_df):,}")
    print("\nDone. apollo_comments_real.csv has NO toxicity scores yet (toxicity_score is NaN).")
    print("Score it (MICRO), save the scored result as data/apollo_comments.csv, then run run_full_from_cache.py.")
    print(f"Subreddits ready for forecasting: {len(subs_complete)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-days", type=int, default=24)
    a = ap.parse_args()
    main(a.cap, a.seed, a.min_days)
