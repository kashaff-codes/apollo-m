"""
Step 5 (macro): the repo's TFT on the qualifying communities, over TFT_SEEDS, with honest dates and a baseline check.

models/tft_forecaster.py is used unchanged, configured as gen_forecasts.py configures it (max_epochs 6, lookback 14,
horizon 5; other TFTConfig defaults). Two things are added AROUND it, in this file only:

  1. Which days are predicted. TFTForecaster.train() trains on time_idx <= max_time_idx - horizon, and
     predict_quantiles() on observed data predicts the last `horizon` observed days of the series (pytorch-forecasting
     predict_mode). This script reads the predicted time_idx from TimeSeriesDataSet.decoded_index and labels each
     prediction with the date it actually predicts. It records whether those steps lie beyond the training cutoff.
  2. A baseline check, fixed in config.py before running: MAE of p50 against the observed daily mean, versus
     persistence (the last encoder day's value) and the mean of the 14 encoder days; share of observations inside
     p10..p90; interval width by horizon day.

Training is stochastic: every number is reported per seed and as a spread across seeds. The canonical seed's run is
written to forecast_results.csv in the column layout of outputs/real/forecast_results.csv plus `actual`,
`interval_width`, `out_of_sample` and `kind`.

Model checkpoints go to data/real_torrent_2026_08/tft_models/seed<N>/ (gitignored), never to models/.

    python scripts/local/torrent_2026_08/forecast.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

sys.path.insert(0, str(C.ROOT))
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
import lightning.pytorch as pl  # noqa: E402
from pytorch_forecasting import TimeSeriesDataSet  # noqa: E402

from gates import Gate, degenerate_table  # noqa: E402
from models.tft_forecaster import TFTConfig, TFTForecaster  # noqa: E402


def risk_level(p50: float) -> str:      # gen_forecasts.py
    return "CRITICAL" if p50 > 0.8 else "HIGH" if p50 > 0.6 else "MEDIUM" if p50 > 0.4 else "LOW"


def run_seed(d: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, float]:
    pl.seed_everything(seed, workers=True)
    f = TFTForecaster(TFTConfig(max_epochs=C.TFT_MAX_EPOCHS, lookback_window=C.LOOKBACK,
                                forecast_horizon=C.HORIZON, model_dir=C.DATA_DIR / "tft_models" / f"seed{seed}"))
    t0 = time.time()
    f.train(d)
    train_s = time.time() - t0
    cutoff = int(f.prepare_dataframe(d)["time_idx"].max()) - C.HORIZON
    rows = []
    for sub in sorted(d["subreddit"].unique()):
        sub_df = d[d["subreddit"] == sub]
        res = f.predict_quantiles(sub_df, subreddit=f"r/{sub}")
        if not res or res["subreddit"] != f"r/{sub}":
            rows.append({"seed": seed, "subreddit": f"r/{sub}", "day": None, "failed": True})
            continue
        prep = f.prepare_dataframe(sub_df).set_index("time_idx")
        ds = TimeSeriesDataSet.from_dataset(f.training_dataset, f.prepare_dataframe(sub_df), predict=True,
                                            stop_randomization=True)
        first = int(ds.decoded_index["time_idx_first_prediction"].iloc[0])
        enc = prep.loc[first - C.LOOKBACK:first - 1, "avg_toxicity"]
        for i in range(C.HORIZON):
            t = first + i
            rows.append({
                "seed": seed, "subreddit": f"r/{sub}", "day": i + 1,
                "date": prep.loc[t, "date"].strftime("%Y-%m-%d"), "time_idx": t,
                "p10": float(res["p10"][i]), "p50": float(res["p50"][i]), "p90": float(res["p90"][i]),
                "actual": float(prep.loc[t, "avg_toxicity"]),
                "persistence": float(prep.loc[first - 1, "avg_toxicity"]),
                "mean14": float(enc.mean()), "encoder_min": float(enc.min()), "encoder_max": float(enc.max()),
                "encoder_days": int(len(enc)), "training_cutoff_time_idx": cutoff,
                "out_of_sample": bool(t > cutoff), "failed": False})
    return pd.DataFrame(rows), train_s


def evaluate(fs: pd.DataFrame) -> dict:
    ok = fs[~fs["failed"]].copy()
    oos = ok[ok["out_of_sample"]]
    ok["width"] = ok["p90"] - ok["p10"]
    per_sub = oos.groupby("subreddit").apply(lambda g: pd.Series({
        "mae_tft": float(np.mean(np.abs(g["p50"] - g["actual"]))),
        "mae_persistence": float(np.mean(np.abs(g["persistence"] - g["actual"]))),
        "mae_mean14": float(np.mean(np.abs(g["mean14"] - g["actual"])))}))
    width_by_day = ok.groupby("day")["width"].mean()
    nondecr = ok.sort_values("day").groupby("subreddit")["width"].apply(
        lambda w: bool((np.diff(w.to_numpy()) >= -1e-12).all()))
    inside_enc = ok[ok["day"] == 1].apply(lambda r: r["encoder_min"] <= r["p50"] <= r["encoder_max"], axis=1)
    return {
        "communities": int(ok["subreddit"].nunique()),
        "failed_communities": sorted(fs.loc[fs["failed"], "subreddit"].unique().tolist()),
        "predicted_days_out_of_sample": int(len(oos)), "predicted_days_total": int(len(ok)),
        "mae_tft_p50": float(np.mean(np.abs(oos["p50"] - oos["actual"]))) if len(oos) else None,
        "mae_persistence": float(np.mean(np.abs(oos["persistence"] - oos["actual"]))) if len(oos) else None,
        "mae_mean14": float(np.mean(np.abs(oos["mean14"] - oos["actual"]))) if len(oos) else None,
        "communities_tft_beats_persistence": int((per_sub["mae_tft"] < per_sub["mae_persistence"]).sum()),
        "communities_tft_beats_mean14": int((per_sub["mae_tft"] < per_sub["mae_mean14"]).sum()),
        "coverage_p10_p90": float(((oos["actual"] >= oos["p10"]) & (oos["actual"] <= oos["p90"])).mean())
        if len(oos) else None,
        "mean_interval_width_by_day": {int(k): round(float(v), 6) for k, v in width_by_day.items()},
        "width_day5_gt_day1": bool(width_by_day.iloc[-1] > width_by_day.iloc[0]),
        "share_communities_width_nondecreasing": float(nondecr.mean()),
        "share_day1_p50_inside_encoder_range": float(inside_enc.mean()),
        "quantile_order_ok": bool(((ok["p10"] <= ok["p50"]) & (ok["p50"] <= ok["p90"])).all()),
    }


def main() -> None:
    daily = pd.read_csv(C.OUT_DIR / "daily_series.csv")
    d = daily[daily["in_forecast_run"]][["subreddit", "date", "avg_toxicity", "toxic_rate", "comment_count"]].copy()
    subs = sorted(d["subreddit"].unique())
    print(f"TFT on {len(subs)} qualifying communities, {len(d)} community-days, seeds {C.TFT_SEEDS}", flush=True)
    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    all_rows, evals, train_secs = [], {}, {}
    for seed in C.TFT_SEEDS:
        fs, secs = run_seed(d, seed)
        all_rows.append(fs)
        evals[seed] = evaluate(fs)
        train_secs[seed] = round(secs, 1)
        print(f"seed {seed}: trained in {secs:.0f}s; {json.dumps(evals[seed])}", flush=True)
    allf = pd.concat(all_rows, ignore_index=True)
    allf.to_csv(C.OUT_DIR / "forecast_seeds.csv", index=False)

    ok = allf[~allf["failed"]]
    spread = ok.groupby(["subreddit", "day"])["p50"].agg(lambda s: s.max() - s.min())
    summary = {
        "what_is_predicted": "the last 5 observed days of each community's run, from the 14 days before them, by a "
                             "model trained on time_idx <= max - 5 (existing TFTForecaster behaviour); NOT future days",
        "seeds": list(C.TFT_SEEDS), "canonical_seed": C.TFT_CANONICAL_SEED, "device": device,
        "train_seconds": train_secs, "config": {"max_epochs": C.TFT_MAX_EPOCHS, "lookback": C.LOOKBACK,
                                                "horizon": C.HORIZON},
        "per_seed": {str(k): v for k, v in evals.items()},
        "across_seeds": {
            "p50_range_per_community_day_mean": float(spread.mean()),
            "p50_range_per_community_day_max": float(spread.max()),
            "mae_tft_p50_min_max": [min(e["mae_tft_p50"] for e in evals.values()),
                                    max(e["mae_tft_p50"] for e in evals.values())],
            "coverage_min_max": [min(e["coverage_p10_p90"] for e in evals.values()),
                                 max(e["coverage_p10_p90"] for e in evals.values())],
            "communities_beating_persistence_per_seed": [e["communities_tft_beats_persistence"]
                                                         for e in evals.values()],
        },
    }
    (C.OUT_DIR / "forecast_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    can = ok[ok["seed"] == C.TFT_CANONICAL_SEED].copy()
    out = pd.DataFrame({
        "day": can["day"].astype(int), "date": can["date"], "predicted_toxicity": can["p50"].round(4),
        "p10": can["p10"].round(4), "p50": can["p50"].round(4), "p90": can["p90"].round(4),
        "risk_level": can["p50"].map(risk_level), "method": "TFT", "subreddit": can["subreddit"],
        "interval_width": (can["p90"] - can["p10"]).round(4), "actual": can["actual"].round(4),
        "out_of_sample": can["out_of_sample"],
        "kind": "prediction of an observed day (last 5 days of the run held out of training)"})
    out.to_csv(C.OUT_DIR / "forecast_results.csv", index=False)

    # ── Gate ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    g = Gate("step 5: TFT forecast")
    g.check("no community failed to forecast", not allf["failed"].any(),
            str(sorted(allf.loc[allf["failed"], "subreddit"].unique())))
    cnt = ok.groupby(["seed", "subreddit"]).size()
    g.check("E9 every qualifying community: 5 days per seed", len(cnt) == len(subs) * len(C.TFT_SEEDS)
            and bool((cnt == C.HORIZON).all()), f"{len(subs)} communities x {len(C.TFT_SEEDS)} seeds")
    g.check("E9 p50 > 0 everywhere", bool((ok["p50"] > 0).all()), f"min p50 {ok['p50'].min():.4f}")
    g.check("no NaN/inf in forecast numbers", bool(np.isfinite(ok[["p10", "p50", "p90", "actual"]].to_numpy()).all()), "")
    degenerate_table(g, ok.assign(width=ok["p90"] - ok["p10"])[["p10", "p50", "p90", "width", "actual"]], "forecast")
    last5 = d.sort_values("date").groupby("subreddit")["date"].apply(lambda s: list(s)[-C.HORIZON:])
    dates_ok = all(list(grp.sort_values("day")["date"]) == last5[sub.replace("r/", "")]
                   for (_, sub), grp in ok.groupby(["seed", "subreddit"]))
    g.check("predicted dates == last 5 observed dates of each run", dates_ok, "")
    g.check("predictions out of sample (beyond training cutoff)", bool(ok["out_of_sample"].all()),
            f"{int(ok['out_of_sample'].sum())} of {len(ok)} community-days")
    for seed, e in evals.items():
        w = e["mean_interval_width_by_day"]
        g.check(f"E6 seed {seed}: mean width day 5 > day 1", e["width_day5_gt_day1"],
                f"{w[1]:.4f} -> {w[5]:.4f}; non-decreasing in {e['share_communities_width_nondecreasing']:.0%}")
        g.check(f"seed {seed}: quantiles ordered p10<=p50<=p90", e["quantile_order_ok"], "")
    g.finish()


if __name__ == "__main__":
    main()
