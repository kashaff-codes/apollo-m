"""
Investigations of every "differs" and every odd-looking output (docs/NEW_DATA_NOTES.md Part C). Read-only on outputs.

  I1  E2b: is r/science vs the lowest community (r/geopolitics) a real difference? Comment-level Welch interval and a
      day-level paired bootstrap (resample the 31 days; comment-weighted monthly means), seed 5, 5,000 resamples.
  I2  DBSCAN flags most communities as outliers: share here vs the August committed outputs; nearest-neighbour
      distances in the standardised 4-feature space against eps = 0.5.
  I3  GNN raw outputs: range here vs August.
  I4  Churn vs the per-day cap: Spearman between churn and pre-cap volume; churn of capped vs uncapped communities.
  I5  Why the TFT loses to the 14-day mean: per seed bias, error vs community level (shrinkage), MAE by horizon day,
      direction agreement over the 5 predicted days, day-1 p50 inside the encoder range.

    python scripts/local/torrent_2026_08/investigate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

BOOT, BOOT_SEED = 5000, 5


def main() -> None:
    out: dict = {}
    sc = pd.read_csv(C.SCORED, usecols=["subreddit", "toxicity_score", "created_utc"], dtype={"subreddit": str})
    daily = pd.read_csv(C.OUT_DIR / "daily_series.csv")
    meso = pd.read_csv(C.OUT_DIR / "meso_report.csv")
    clus = pd.read_csv(C.OUT_DIR / "clusters_report.csv")
    fs = pd.read_csv(C.OUT_DIR / "forecast_seeds.csv")

    # I1
    a, b = "science", meso.sort_values("toxicity_rate").iloc[0]["subreddit"].replace("r/", "")
    xa = sc.loc[sc["subreddit"] == a, "toxicity_score"].to_numpy()
    xb = sc.loc[sc["subreddit"] == b, "toxicity_score"].to_numpy()
    diff = xa.mean() - xb.mean()
    se = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
    da = daily[daily["subreddit"] == a].set_index("date")
    db = daily[daily["subreddit"] == b].set_index("date")
    days = sorted(set(da.index) & set(db.index))
    rng = np.random.default_rng(BOOT_SEED)
    boots = []
    for _ in range(BOOT):
        d = rng.choice(days, size=len(days), replace=True)
        wa = (da.loc[d, "avg_toxicity"] * da.loc[d, "comment_count"]).sum() / da.loc[d, "comment_count"].sum()
        wb = (db.loc[d, "avg_toxicity"] * db.loc[d, "comment_count"]).sum() / db.loc[d, "comment_count"].sum()
        boots.append(wa - wb)
    out["I1_science_vs_lowest"] = {
        "lowest": b, "mean_science": round(float(xa.mean()), 5), "mean_lowest": round(float(xb.mean()), 5),
        "diff": round(float(diff), 5), "comment_level_95ci": [round(float(diff - 1.96 * se), 5),
                                                              round(float(diff + 1.96 * se), 5)],
        "day_bootstrap_95ci": [round(float(np.percentile(boots, 2.5)), 5), round(float(np.percentile(boots, 97.5)), 5)],
        "share_of_bootstrap_with_science_lower": round(float(np.mean(np.array(boots) < 0)), 4),
        "n_comments": [int(len(xa)), int(len(xb))], "days": len(days)}

    # I2
    feats = ["toxicity_rate", "polarization", "echo_chamber_index", "churn_rate"]
    X = StandardScaler().fit_transform(clus[feats].fillna(0).values)
    nn = NearestNeighbors(n_neighbors=5).fit(X)        # DBSCAN core point: >= 5 points (incl. itself) within eps
    dist, _ = nn.kneighbors(X)
    aug = pd.read_csv(C.ROOT / "outputs" / "real" / "clusters_report.csv")
    out["I2_dbscan"] = {
        "outliers_here": int(clus["is_outlier"].sum()), "communities_here": int(len(clus)),
        "outliers_august": int(aug["is_outlier"].sum()), "communities_august": int(len(aug)),
        "eps": 0.5, "min_samples": 5,
        "distance_to_4th_neighbour_median": round(float(np.median(dist[:, 4])), 3),
        "distance_to_4th_neighbour_min": round(float(dist[:, 4].min()), 3),
        "points_with_4th_neighbour_within_eps": int((dist[:, 4] <= 0.5).sum())}

    # I3
    am = pd.read_csv(C.ROOT / "outputs" / "real" / "meso_report.csv")
    out["I3_gnn_raw"] = {"here_min_max": [float(meso["gnn_risk_raw"].min()), float(meso["gnn_risk_raw"].max())],
                         "here_distinct_at_6dp": int(meso["gnn_risk_raw"].nunique()),
                         "august_min_max": [float(am["gnn_risk_raw"].min()), float(am["gnn_risk_raw"].max())]}

    # I4
    vol = pd.read_csv(C.OUT_DIR / "filter_volume_by_subreddit.csv")
    mm = meso.assign(sub=meso["subreddit"].str.replace("r/", "", regex=False)).merge(vol, left_on="sub",
                                                                                    right_on="subreddit")
    rho = spearmanr(mm["churn_rate"], mm["eligible_before_cap"]).statistic
    capped = mm["days_hit_cap"] >= 25
    out["I4_churn_vs_cap"] = {
        "spearman_churn_vs_precap_volume": round(float(rho), 3),
        "median_churn_capped_25plus_days": round(float(mm.loc[capped, "churn_rate"].median()), 4),
        "median_churn_capped_fewer_days": round(float(mm.loc[~capped, "churn_rate"].median()), 4),
        "n_capped_25plus_days": int(capped.sum()), "n_other": int((~capped).sum()),
        "august_churn_min_max": [float(am["churn_rate"].min()), float(am["churn_rate"].max())]}

    # I5
    ok = fs[~fs["failed"]].copy()
    ok["err"] = ok["p50"] - ok["actual"]
    lvl = daily.groupby("subreddit")["avg_toxicity"].mean()
    per_seed = {}
    for seed, g in ok.groupby("seed"):
        pc = g.groupby("subreddit").agg(err=("err", "mean"), p50=("p50", "mean"), actual=("actual", "mean"))
        pc["level"] = pc.index.str.replace("r/", "", regex=False).map(lvl)
        slope = float(np.polyfit(pc["level"], pc["err"], 1)[0])
        dirs = g.sort_values("day").groupby("subreddit").apply(
            lambda x: np.sign(x["p50"].iloc[-1] - x["p50"].iloc[0]) == np.sign(x["actual"].iloc[-1] - x["actual"].iloc[0]))
        in_enc = g[g["day"] == 1].apply(lambda r: r["encoder_min"] <= r["p50"] <= r["encoder_max"], axis=1)
        per_seed[int(seed)] = {
            "bias_mean_p50_minus_actual": round(float(g["err"].mean()), 5),
            "slope_error_vs_community_level": round(slope, 3),
            "spread_of_community_mean_p50": round(float(pc["p50"].std()), 4),
            "spread_of_community_mean_actual": round(float(pc["actual"].std()), 4),
            "mae_by_day": {int(k): round(float(v), 5) for k, v in g.groupby("day")["err"].apply(
                lambda e: np.abs(e).mean()).items()},
            "direction_agreement_day1_to_day5": round(float(dirs.mean()), 3),
            "day1_p50_inside_encoder_range": round(float(in_enc.mean()), 3)}
    within_sd = daily[daily["in_forecast_run"]].groupby("subreddit")["avg_toxicity"].std().median()
    out["I5_tft"] = {"per_seed": per_seed, "median_within_community_daily_sd": round(float(within_sd), 4),
                     "note": "slope < 0 means the TFT pulls every community toward the cross-community mean "
                             "(over-predicts low communities, under-predicts high ones)"}
    (C.OUT_DIR / "investigations.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
