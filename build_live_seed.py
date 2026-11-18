"""
Build outputs/live_seed.csv — the comments the dashboard's Live page replays.

    python build_live_seed.py                 # 60 newest comments per community
    python build_live_seed.py --per-sub 100

Why this exists
---------------
The Live page streams a committed slice of scored comments so that it moves for
every visitor, with no background process. That slice was cut on 11 August from
the corpus prepare_data.py had fabricated — Twitter and Jigsaw text pasted onto
Reddit metadata — so half of what the page called "real comments" were tweets
("RT @…", hashtags) under invented subreddits such as r/WAPOauto. The page's
caption said the comments were real. They were not.

This rebuilds it from genuine Reddit comments:

  * fetched from Arctic Shift, the project's credential-free source, for the 39
    communities of the RC_2026-08 corpus;
  * bots and canned text removed with the pipeline's own filter
    (main.drop_bots_and_boilerplate);
  * scored by the pipeline's own model, unitary/toxic-bert, with the pipeline's
    threshold (main.Config.TOXICITY_THRESHOLD);
  * written WITHOUT author names, with bodies cut to 280 characters, and in
    posting order so the replay interleaves communities as Reddit did.

The run is recorded in outputs/live_seed.provenance.json.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from apollo_datasets import OUTPUTS, get

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
REMOVED = {"", "[deleted]", "[removed]", "[ Removed by Reddit ]"}


def fetch(sub: str, n: int, before: int) -> list[dict]:
    r = requests.get(API, params={"subreddit": sub, "limit": min(n, 100), "before": before}, timeout=40)
    r.raise_for_status()
    return r.json().get("data") or []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-sub", type=int, default=60)
    args = ap.parse_args()

    from main import Config, drop_bots_and_boilerplate
    from transformers import pipeline

    subs = [s.removeprefix("r/") for s in pd.read_csv(get("real_torrent_2026_08").file("meso_report.csv"))["subreddit"]]
    now = int(datetime.now(timezone.utc).timestamp())
    rows, failed = [], []
    for s in subs:
        try:
            for c in fetch(s, args.per_sub, now):
                body = str(c.get("body") or "").strip()
                if body in REMOVED or len(body) < 4:
                    continue
                rows.append({"subreddit": s, "author": str(c.get("author", "")), "body": body,
                             "created_utc": int(c["created_utc"])})
        except Exception as exc:  # noqa: BLE001 - one community failing must not stop the rest
            failed.append(f"{s}: {type(exc).__name__}")
        time.sleep(0.4)                                        # be polite to a free service
    df = drop_bots_and_boilerplate(pd.DataFrame(rows))

    clf = pipeline("text-classification", model=Config.NLP_MODEL, top_k=None, device=-1)
    scores = []
    for out in clf([b[:512] for b in df["body"]], batch_size=32, truncation=True):
        scores.append(next((x["score"] for x in out if x["label"] == "toxic"), 0.0))
    df["toxicity"] = [round(float(x), 4) for x in scores]
    df["is_toxic"] = df["toxicity"] >= Config.TOXICITY_THRESHOLD
    df["body"] = df["body"].str.replace(r"\s+", " ", regex=True).str.slice(0, 280)
    df = df.sort_values("created_utc").drop(columns=["author"]).reset_index(drop=True)

    df[["subreddit", "body", "toxicity", "is_toxic", "created_utc"]].to_csv(OUTPUTS / "live_seed.csv", index=False)
    prov = {
        "source": API, "fetched_at_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(),
        "communities_requested": len(subs), "communities_failed": failed,
        "comments": int(len(df)), "communities_with_comments": int(df["subreddit"].nunique()),
        "posted_between": [datetime.fromtimestamp(int(df["created_utc"].min()), timezone.utc).isoformat(),
                           datetime.fromtimestamp(int(df["created_utc"].max()), timezone.utc).isoformat()],
        "model": Config.NLP_MODEL, "threshold": Config.TOXICITY_THRESHOLD,
        "toxic_rate": round(float(df["is_toxic"].mean()), 4),
        "authors_stored": False, "body_max_chars": 280,
    }
    (OUTPUTS / "live_seed.provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
