"""
Gate for step 2: does the filtered CSV agree with its own summary, and is it what the filter promised?

Checks (all must pass):
  * rows read -> matched -> dropped (by reason) -> eligible add up exactly,
  * CSV data rows == summary comments_kept == sum of per-sub-day kept == sum of min(eligible, cap),
  * per-subreddit CSV counts == summary kept_per_subreddit,
  * every (subreddit, UTC day) in the CSV has kept == min(eligible, cap) from the summary,
  * comment ids unique, no empty/removed bodies, timestamps inside the window, only requested subreddits,
  * numeric columns (score, created_utc) are not degenerate.

Prints the volume table (eligible before cap, kept, days with data, days hitting the cap) per subreddit.

    python scripts/local/torrent_2026_08/check_filter_output.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402
from filter_arctic_dump import DEFAULT_SUBREDDITS, REMOVED  # noqa: E402
from gates import Gate, describe_numeric  # noqa: E402


def main() -> None:
    s = json.loads(C.FILTER_SUMMARY.read_text(encoding="utf-8"))
    df = pd.read_csv(C.FILTERED_CSV, dtype={"id": str, "subreddit": str, "author": str, "body": str},
                     keep_default_na=False)
    g = Gate("step 2: filtered CSV vs summary")
    st = s["stats"]
    psd = s["per_subreddit_day"]
    elig_total = sum(v["eligible"] for v in psd.values())
    dropped = {k: st.get(k, 0) for k in ("bad_json", "missing_field", "deleted_or_removed", "bad_timestamp",
                                          "outside_window", "duplicate_id")}
    g.check("matched = sum(dropped by reason) + eligible",
            st["matched_subreddit"] == sum(dropped.values()) + elig_total,
            f"{st['matched_subreddit']:,} = {sum(dropped.values()):,} {dropped} + {elig_total:,}")
    kept_sum = sum(v["kept"] for v in psd.values())
    capped = sum(min(v["eligible"], s["cap_per_sub_per_day"]) for v in psd.values())
    g.check("CSV rows == comments_kept == sum kept == sum min(eligible,cap)",
            len(df) == s["comments_kept"] == kept_sum == capped,
            f"{len(df):,} / {s['comments_kept']:,} / {kept_sum:,} / {capped:,}")
    per_sub = df["subreddit"].value_counts().to_dict()
    g.check("per-subreddit CSV counts == summary kept_per_subreddit", per_sub == s["kept_per_subreddit"],
            f"{len(per_sub)} subreddits")
    df["day"] = pd.to_datetime(df["created_utc"].astype(np.int64), unit="s", utc=True).dt.strftime("%Y-%m-%d")
    csv_sd = df.groupby(["subreddit", "day"]).size()
    mism = [k for k, v in psd.items() if csv_sd.get(tuple(k.split("|")), 0) != v["kept"]]
    g.check("every sub-day: CSV rows == summary kept", not mism and len(csv_sd) == len(psd),
            f"{len(psd)} sub-days, {len(mism)} mismatches")
    g.check("comment ids unique", df["id"].is_unique, f"{df['id'].nunique():,} unique")
    bad_body = df["body"].str.strip().isin(REMOVED).sum()
    g.check("no empty/removed bodies", bad_body == 0, f"{bad_body} found")
    g.check("no [deleted] authors", (df["author"] == "[deleted]").sum() == 0, "")
    lo, hi = s["window"]
    ts = df["created_utc"].astype(np.int64)
    g.check("timestamps inside window", bool(ts.between(lo, hi).all()),
            f"{pd.to_datetime(ts.min(), unit='s')} .. {pd.to_datetime(ts.max(), unit='s')}")
    g.check("only requested subreddits", set(df["subreddit"]) <= set(DEFAULT_SUBREDDITS),
            f"{df['subreddit'].nunique()} of {len(DEFAULT_SUBREDDITS)}")
    for col in ("score", "created_utc"):
        d = describe_numeric(df[col])
        g.check(f"{col} not degenerate", d["ok"], d["text"])
    g.check("SHA-256 recorded", bool(s.get("input_sha256")), s.get("input_sha256", "missing"))

    # Volume table (eligible before cap is the real volume; kept is what was sampled).
    rows = []
    for sub in sorted(DEFAULT_SUBREDDITS, key=str.lower):
        items = {k: v for k, v in psd.items() if k.split("|")[0] == sub}
        rows.append({"subreddit": sub, "days_with_data": len(items),
                     "eligible_before_cap": sum(v["eligible"] for v in items.values()),
                     "kept": sum(v["kept"] for v in items.values()),
                     "days_hit_cap": sum(v["hit_cap"] for v in items.values()),
                     "min_kept_per_day": min((v["kept"] for v in items.values()), default=0)})
    vol = pd.DataFrame(rows).sort_values("eligible_before_cap", ascending=False)
    pd.set_option("display.width", 200)
    print(vol.to_string(index=False))
    print(f"\ntotal eligible before cap {vol.eligible_before_cap.sum():,}; kept {vol.kept.sum():,}; "
          f"sub-days {len(psd)}; sub-days hitting cap {s['days_hitting_cap']}; "
          f"subreddits without data {s['subreddits_without_data']}; lines read {st['lines_read']:,}; "
          f"seconds {s['seconds']}")
    vol.to_csv(C.OUT_DIR / "filter_volume_by_subreddit.csv", index=False)
    g.finish()


if __name__ == "__main__":
    main()
