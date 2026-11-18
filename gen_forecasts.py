"""
Five-day TFT forecasts for every eligible community, and a backtest of the model.

    python gen_forecasts.py                          # benchmark
    python gen_forecasts.py --dataset real_api_2026_07
    python gen_forecasts.py --real                   # same as the line above

Writes, into the dataset's directory (see apollo_datasets.py):

  forecast_future.csv    the five calendar days AFTER each community's last
                         observation — a forecast in the ordinary sense.
  forecast_backtest.csv  the model on the last five observed days, which are
                         held out of training, against two naive baselines
                         (repeat the last value; repeat the recent mean).

What this replaced
------------------
This script used to call predict_quantiles() without future=True, which makes
pytorch-forecasting predict the last five rows already in the series, and then
stamped those rows with dates after the data (a hard-coded 2026-08-11 start).
The values were genuine model output for observed days; the dates presented
them as the future, and the dashboard plotted them on a "days ahead" axis.
Rizwan's RC_2026-08 handoff identified this. For communities whose series were
shorter than the longest one, the training cutoff did not exclude those days
either, so part of what was shown as a forecast was an in-sample fit.

Those files are still committed as outputs/forecast_results.csv and
outputs/real/forecast_results.csv, because the paper's reproduction scripts
read them (docs/paper/inputs.sha256 pins them). This script never writes that
name, so re-running it cannot alter a paper input.
"""
from __future__ import annotations

import argparse
import json
import logging
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

from apollo_datasets import DATASETS, get  # noqa: E402
from models.tft_forecaster import TFTConfig, TFTForecaster  # noqa: E402

HORIZON, LOOKBACK, EPOCHS = 5, 14, 6


def risk(p50: float) -> str:
    return "CRITICAL" if p50 > 0.8 else "HIGH" if p50 > 0.6 else "MEDIUM" if p50 > 0.4 else "LOW"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="benchmark", choices=sorted(DATASETS))
    ap.add_argument("--real", action="store_true", help="alias for --dataset real_api_2026_07")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    ds = get("real_api_2026_07" if args.real else args.dataset)

    import lightning.pytorch as pl
    pl.seed_everything(args.seed, workers=True)

    meso = pd.read_csv(ds.file("meso_report.csv"))
    targets = set(meso["subreddit"].astype(str).str.replace(r"^r/", "", regex=True))
    daily = pd.read_csv(ds.daily)
    daily["subreddit"] = daily["subreddit"].astype(str).str.replace(r"^r/", "", regex=True)
    daily = daily[daily["subreddit"].isin(targets)].copy()
    daily = daily.dropna(subset=["avg_toxicity"])

    f = TFTForecaster(TFTConfig(max_epochs=args.epochs, lookback_window=LOOKBACK,
                                forecast_horizon=HORIZON, use_comment_count=False))
    eligible = f.eligible_subreddits(daily)
    print(f"[{ds.key}] training TFT on {len(eligible)} eligible communities "
          f"({daily['subreddit'].nunique()} in the series)")
    f.train(daily)

    rows = []
    last_obs = pd.to_datetime(daily["date"], utc=True).groupby(daily["subreddit"]).max()
    for label in eligible:
        bare = label[2:]
        res = f.predict_quantiles(daily[daily["subreddit"] == bare], subreddit=label, future=True)
        if not res or res.get("subreddit") != label:
            continue
        start = last_obs[bare].normalize()
        for i in range(HORIZON):
            p50 = float(res["p50"][i])
            rows.append({
                "subreddit": label, "day": i + 1,
                "date": (start + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d"),
                "last_observed": start.strftime("%Y-%m-%d"),
                "p10": round(float(res["p10"][i]), 4), "p50": round(p50, 4),
                "p90": round(float(res["p90"][i]), 4),
                "interval_width": round(float(res["p90"][i] - res["p10"][i]), 4),
                "risk_level": risk(p50), "method": "TFT", "kind": "future",
            })
    future = pd.DataFrame(rows)
    future.to_csv(ds.file("forecast_future.csv"), index=False)

    bt = f.backtest(daily)
    bt.to_csv(ds.file("forecast_backtest.csv"), index=False)

    summary = {"dataset": ds.key, "communities_forecast": int(future["subreddit"].nunique()),
               "epochs": args.epochs, "seed": args.seed, "lookback": LOOKBACK, "horizon": HORIZON}
    if not bt.empty:
        w = bt["n_days"]
        wavg = lambda c: round(float((bt[c] * w).sum() / w.sum()), 5)
        summary.update({
            "backtest_communities": int(len(bt)), "backtest_days": int(w.sum()),
            "mae_tft": wavg("mae_tft"), "mae_last_value": wavg("mae_last_value"),
            "mae_recent_mean": wavg("mae_recent_mean"),
            "coverage_p10_p90": wavg("inside_p10_p90"),
            "tft_beats_last_value_in": int((bt["mae_tft"] < bt["mae_last_value"]).sum()),
            "tft_beats_recent_mean_in": int((bt["mae_tft"] < bt["mae_recent_mean"]).sum()),
        })
    ds.file("forecast_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
