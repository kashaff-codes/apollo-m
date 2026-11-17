"""
Step 4/5a: per-subreddit per-day series and the coverage rules, from scored_comments.csv.

Coverage rules (config.py, fixed before scoring): a day counts for a community only with >= MIN_COMMENTS_PER_DAY scored
comments; a community qualifies for forecasting only with a run of >= MIN_CONSECUTIVE_DAYS consecutive counting days in
the one window 2026-08-01..2026-08-31, and the forecaster is fed exactly that run (the latest such run). Missing days
are never zero-filled: a day with no comments simply has no row.

Writes (aggregates only, no usernames or text):
    outputs/real_torrent_2026_08/daily_series.csv   subreddit, date, avg_toxicity, toxic_rate (>=0.6), toxic_rate_050,
                                                    comment_count, unique_authors, meets_min_comments, in_forecast_run,
                                                    time_idx, group_id
    outputs/real_torrent_2026_08/coverage.csv       one row per subreddit with the rule outcome

    python scripts/local/torrent_2026_08/build_daily.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from gates import Gate, degenerate_table  # noqa: E402


def runs(dates: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Maximal runs of consecutive calendar days."""
    out = []
    for d in sorted(dates):
        if out and (d - out[-1][1]).days == 1:
            out[-1] = (out[-1][0], d, out[-1][2] + 1)
        else:
            out.append((d, d, 1))
    return out


def main() -> None:
    sc = pd.read_csv(C.SCORED, usecols=["id", "subreddit", "author", "created_utc", "toxicity_score", "is_toxic",
                                        "is_toxic_050"], dtype={"id": str, "subreddit": str, "author": str})
    sc["date"] = pd.to_datetime(sc["created_utc"], unit="s", utc=True).dt.tz_localize(None).dt.floor("D")
    daily = (sc.groupby(["subreddit", "date"], as_index=False)
               .agg(avg_toxicity=("toxicity_score", "mean"), toxic_rate=("is_toxic", "mean"),
                    toxic_rate_050=("is_toxic_050", "mean"), comment_count=("id", "count"),
                    unique_authors=("author", "nunique"))
               .sort_values(["subreddit", "date"]).reset_index(drop=True))
    daily["meets_min_comments"] = daily["comment_count"] >= C.MIN_COMMENTS_PER_DAY
    daily["in_forecast_run"] = False

    cov = []
    for sub, g in daily.groupby("subreddit"):
        ok_days = g.loc[g["meets_min_comments"], "date"].tolist()
        rs = runs(ok_days)
        longest = max(rs, key=lambda r: r[2]) if rs else (None, None, 0)
        qual = [r for r in rs if r[2] >= C.MIN_CONSECUTIVE_DAYS]
        chosen = qual[-1] if qual else None
        if chosen:
            m = (daily["subreddit"] == sub) & daily["date"].between(chosen[0], chosen[1])
            daily.loc[m, "in_forecast_run"] = True
        cov.append({"subreddit": sub, "comments_scored": int(g["comment_count"].sum()),
                    "days_with_data": len(g), "days_meeting_min": len(ok_days),
                    "min_comments_per_day": int(g["comment_count"].min()),
                    "longest_run_days": longest[2],
                    "run_start": chosen[0].date().isoformat() if chosen else "",
                    "run_end": chosen[1].date().isoformat() if chosen else "",
                    "qualifies_forecast": bool(chosen),
                    "reason": ("" if chosen else
                               f"longest run of days with >= {C.MIN_COMMENTS_PER_DAY} comments is {longest[2]} "
                               f"< {C.MIN_CONSECUTIVE_DAYS}")})
    cov = pd.DataFrame(cov).sort_values(["qualifies_forecast", "subreddit"], ascending=[False, True])
    daily["time_idx"] = daily.groupby("subreddit").cumcount()
    daily["group_id"] = daily["subreddit"]
    out = daily.assign(date=daily["date"].dt.strftime("%Y-%m-%d"))
    for col in ("avg_toxicity", "toxic_rate", "toxic_rate_050"):
        out[col] = out[col].round(6)
    out.to_csv(C.OUT_DIR / "daily_series.csv", index=False)
    cov.to_csv(C.OUT_DIR / "coverage.csv", index=False)

    # ── Gate ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    g = Gate("step 4/5a: daily series and coverage")
    per_sub_scored = sc["subreddit"].value_counts()
    per_sub_daily = daily.groupby("subreddit")["comment_count"].sum()
    g.check("sum comment_count == comments scored", int(daily["comment_count"].sum()) == len(sc),
            f"{int(daily['comment_count'].sum()):,} == {len(sc):,}")
    g.check("per-subreddit daily sums == per-subreddit scored", per_sub_daily.sort_index().equals(
        per_sub_scored.sort_index().rename("comment_count")), f"{len(per_sub_daily)} subreddits")
    g.check("one row per (subreddit, day) present; none zero-filled",
            len(daily) == sc.groupby(["subreddit", "date"]).ngroups and int(daily["comment_count"].min()) >= 1,
            f"{len(daily)} rows, min comment_count {int(daily['comment_count'].min())}")
    g.check("all dates inside the window",
            bool(daily["date"].between(pd.Timestamp(C.WINDOW_START), pd.Timestamp(C.WINDOW_END)).all()), "")
    degenerate_table(g, daily[["avg_toxicity", "toxic_rate", "toxic_rate_050", "comment_count", "unique_authors",
                               "time_idx"]], "daily")
    within = daily.groupby("subreddit")["avg_toxicity"].std()
    g.check("toxicity varies within every community (daily std > 0)", bool((within > 0).all()),
            f"daily std min {within.min():.4f}  median {within.median():.4f}  max {within.max():.4f}")
    means = sc.groupby("subreddit")["toxicity_score"].mean()
    g.check("toxicity varies across communities", means.nunique() > 1 and means.std() > 0,
            f"community means {means.min():.4f} .. {means.max():.4f}, std {means.std():.4f}")
    fr = daily[daily["in_forecast_run"]]
    ok_runs = all(((grp["date"].diff().dropna() == pd.Timedelta(days=1)).all()
                   and len(grp) >= C.MIN_CONSECUTIVE_DAYS and (grp["comment_count"] >= C.MIN_COMMENTS_PER_DAY).all())
                  for _, grp in fr.groupby("subreddit"))
    nq = int(cov["qualifies_forecast"].sum())
    g.check("forecast runs consecutive, >= 24 days, >= 50 comments/day", ok_runs and nq > 0,
            f"{nq} of {len(cov)} communities qualify")
    pd.set_option("display.width", 220)
    print(cov.to_string(index=False))
    g.finish()


if __name__ == "__main__":
    main()
