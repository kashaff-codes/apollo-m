"""
Improvement experiment B (outputs/real_torrent_2026_08/improvement/PREREGISTRATION.md, Part B).

B1  From the existing forecast_seeds.csv (no new training): per seed and community, MAE of TFT p50, persistence and the
    14-day mean over the held-out days 2026-08-27..31; communities the TFT wins; blends w*p50 + (1-w)*mean14 for
    w in {0, 0.25, 0.5, 0.75, 1}; paired bootstrap over communities (2,000, seed 2029) of MAE(blend) - MAE(mean14).
B2  Rolling-origin backtest of persistence and the 14-day mean at origins 2026-08-14/17/20/23/26 (5 days each).
    TFT re-trained per origin, CPU, unchanged TFTForecaster, seeds 42/43/44, only at origins 20/23/26 (the existing
    code needs 19 training days), under the time cap; the origin-26 run must reproduce forecast_seeds.csv exactly.

    python scripts/local/torrent_2026_08/improve_forecast.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from gates import Gate  # noqa: E402

IMP_OUT = C.OUT_DIR / "improvement"
IMP_DATA = C.DATA_DIR / "improvement"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
BOOT, BOOT_SEED = 2000, 2029
ORIGINS = ("2026-08-14", "2026-08-17", "2026-08-20", "2026-08-23", "2026-08-26")
TFT_ORIGINS = ("2026-08-20", "2026-08-23", "2026-08-26")
FIRST_ORIGIN_CAP_S, TOTAL_CAP_S = 600, 1800


def b1(g: Gate) -> dict:
    fs = pd.read_csv(C.OUT_DIR / "forecast_seeds.csv")
    fs = fs[~fs["failed"]]
    rng = np.random.default_rng(BOOT_SEED)
    out, rows = {}, []
    for seed, d in fs.groupby("seed"):
        pc = d.groupby("subreddit").apply(lambda x: pd.Series({
            "mae_tft": np.mean(np.abs(x["p50"] - x["actual"])),
            "mae_persistence": np.mean(np.abs(x["persistence"] - x["actual"])),
            "mae_mean14": np.mean(np.abs(x["mean14"] - x["actual"])),
            **{f"mae_blend_{w}": np.mean(np.abs(w * x["p50"] + (1 - w) * x["mean14"] - x["actual"])) for w in WEIGHTS}}))
        pc["seed"] = int(seed)
        rows.append(pc.reset_index())
        subs = pc.index.to_numpy()
        boots = {w: [] for w in WEIGHTS}
        for _ in range(BOOT):
            i = rng.choice(len(subs), size=len(subs), replace=True)
            s = pc.iloc[i]
            for w in WEIGHTS:
                boots[w].append(float(s[f"mae_blend_{w}"].mean() - s["mae_mean14"].mean()))
        out[int(seed)] = {
            "communities": int(len(pc)), "community_days": int(len(d)),
            "mae_tft": float(pc["mae_tft"].mean()), "mae_persistence": float(pc["mae_persistence"].mean()),
            "mae_mean14": float(pc["mae_mean14"].mean()),
            "tft_wins_vs_persistence": int((pc["mae_tft"] < pc["mae_persistence"]).sum()),
            "tft_wins_vs_mean14": int((pc["mae_tft"] < pc["mae_mean14"]).sum()),
            "blends": {str(w): {"mae": float(pc[f"mae_blend_{w}"].mean()),
                                "diff_vs_mean14": float(pc[f"mae_blend_{w}"].mean() - pc["mae_mean14"].mean()),
                                "diff_ci95": [float(np.percentile(boots[w], 2.5)), float(np.percentile(boots[w], 97.5))]}
                       for w in WEIGHTS}}
        g.check(f"B1 seed {seed}: 38 communities x 5 days", len(pc) == 38 and len(d) == 190, f"{len(pc)} / {len(d)}")
        g.check(f"B1 seed {seed}: blend w=0 == 14-day mean, w=1 == TFT",
                abs(out[int(seed)]["blends"]["0.0"]["mae"] - out[int(seed)]["mae_mean14"]) < 1e-12
                and abs(out[int(seed)]["blends"]["1.0"]["mae"] - out[int(seed)]["mae_tft"]) < 1e-12, "")
    beats = {str(w): all(out[s]["blends"][str(w)]["diff_ci95"][1] < 0 for s in out) for w in WEIGHTS if w > 0}
    pd.concat(rows).to_csv(IMP_OUT / "forecast_b1_per_community.csv", index=False)
    return {"per_seed": out, "blend_beats_mean14_all_seeds": beats, "any_blend_beats_mean14": any(beats.values())}


def b2(g: Gate) -> dict:
    daily = pd.read_csv(C.OUT_DIR / "daily_series.csv")
    d = daily[daily["in_forecast_run"]][["subreddit", "date", "avg_toxicity", "toxic_rate", "comment_count"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    rows = []
    for o in ORIGINS:
        o_ts = pd.Timestamp(o)
        for sub, x in d.groupby("subreddit"):
            x = x.set_index("date")["avg_toxicity"]
            hist = x.loc[o_ts - pd.Timedelta(days=13):o_ts]
            fut = x.loc[o_ts + pd.Timedelta(days=1):o_ts + pd.Timedelta(days=5)]
            assert len(hist) == 14 and len(fut) == 5, (sub, o, len(hist), len(fut))
            rows.append({"origin": o, "subreddit": sub,
                         "mae_persistence": float(np.mean(np.abs(x.loc[o_ts] - fut))),
                         "mae_mean14": float(np.mean(np.abs(hist.mean() - fut)))})
    bl = pd.DataFrame(rows)
    bl.to_csv(IMP_OUT / "forecast_b2_baselines.csv", index=False)
    g.check("B2 baselines: 5 origins x 38 communities", len(bl) == 190 and bl["subreddit"].nunique() == 38, str(len(bl)))
    res = {"origins": list(ORIGINS),
           "baselines_by_origin": bl.groupby("origin")[["mae_persistence", "mae_mean14"]].mean().round(6).to_dict("index"),
           "baselines_pooled": bl[["mae_persistence", "mae_mean14"]].mean().round(6).to_dict(),
           "mean14_better_than_persistence_communities_pooled": int(
               (bl.groupby("subreddit")["mae_mean14"].mean() < bl.groupby("subreddit")["mae_persistence"].mean()).sum()),
           "mean14_better_than_persistence_by_origin": bl.assign(b=bl["mae_mean14"] < bl["mae_persistence"]).groupby(
               "origin")["b"].sum().astype(int).to_dict()}

    # TFT per origin (CPU), unchanged TFTForecaster via forecast.run_seed; checkpoints go to the improvement folder.
    import forecast
    C.DATA_DIR = IMP_DATA
    t_all = time.time()
    tft_rows, timing, aborted = [], {}, None
    for k, o in enumerate(TFT_ORIGINS):
        t0 = time.time()
        trunc = d[d["date"] <= pd.Timestamp(o) + pd.Timedelta(days=5)].copy()
        trunc["date"] = trunc["date"].dt.strftime("%Y-%m-%d")
        for seed in C.TFT_SEEDS:
            fs, _ = forecast.run_seed(trunc, seed)
            fs["origin"] = o
            tft_rows.append(fs)
        timing[o] = round(time.time() - t0, 1)
        if k == 0 and timing[o] > FIRST_ORIGIN_CAP_S:
            aborted = f"first origin took {timing[o]}s > {FIRST_ORIGIN_CAP_S}s"
            break
        if time.time() - t_all > TOTAL_CAP_S:
            aborted = f"total time {time.time() - t_all:.0f}s > {TOTAL_CAP_S}s"
            break
    import torch
    res["tft"] = {"device": "cpu" if not torch.cuda.is_available() else "GPU VISIBLE", "seconds_per_origin": timing,
                  "total_seconds": round(time.time() - t_all, 1), "aborted": aborted}
    if tft_rows:
        tf = pd.concat(tft_rows, ignore_index=True)
        tf = tf[~tf["failed"]]
        tf.to_csv(IMP_OUT / "forecast_b2_tft.csv", index=False)
        ref = pd.read_csv(C.OUT_DIR / "forecast_seeds.csv")
        mine = tf[tf["origin"] == "2026-08-26"].sort_values(["seed", "subreddit", "day"]).reset_index(drop=True)
        ref = ref.sort_values(["seed", "subreddit", "day"]).reset_index(drop=True)
        if len(mine):
            diff = float(np.abs(mine["p50"].to_numpy() - ref["p50"].to_numpy()).max())
            g.check("B2 origin-26 TFT re-run reproduces forecast_seeds.csv", diff < 1e-9 and len(mine) == len(ref),
                    f"max |diff| {diff:.2e}")
        g.check("B2 TFT ran on CPU", res["tft"]["device"] == "cpu", res["tft"]["device"])
        g.check("B2 TFT predictions out of sample", bool(tf["out_of_sample"].all()), f"{len(tf)} community-days")
        pc = tf.groupby(["origin", "seed", "subreddit"]).apply(lambda x: pd.Series({
            "mae_tft": np.mean(np.abs(x["p50"] - x["actual"])),
            "mae_persistence": np.mean(np.abs(x["persistence"] - x["actual"])),
            "mae_mean14": np.mean(np.abs(x["mean14"] - x["actual"]))})).reset_index()
        chk = pc.groupby(["origin", "subreddit"])[["mae_persistence", "mae_mean14"]].first().reset_index()
        chk["subreddit"] = chk["subreddit"].str.replace(r"^r/", "", regex=True)   # TFT rows carry the r/ prefix
        chk = chk.merge(bl, on=["origin", "subreddit"], suffixes=("_tft_run", ""), validate="one_to_one")
        dmax = max(float((chk["mae_mean14_tft_run"] - chk["mae_mean14"]).abs().max()),
                   float((chk["mae_persistence_tft_run"] - chk["mae_persistence"]).abs().max()))
        g.check("B2 baselines identical in TFT run and baseline-only run",
                len(chk) == len(TFT_ORIGINS) * 38 and dmax < 1e-12, f"{len(chk)} rows, max |diff| {dmax:.2e}")
        by = pc.groupby(["origin", "seed"]).agg(mae_tft=("mae_tft", "mean"), mae_persistence=("mae_persistence", "mean"),
                                                mae_mean14=("mae_mean14", "mean"),
                                                tft_wins_vs_mean14=("mae_tft", lambda s: 0))
        wins = pc.assign(w=pc["mae_tft"] < pc["mae_mean14"], wp=pc["mae_tft"] < pc["mae_persistence"]).groupby(
            ["origin", "seed"])[["w", "wp"]].sum()
        by["tft_wins_vs_mean14"] = wins["w"].astype(int)
        by["tft_wins_vs_persistence"] = wins["wp"].astype(int)
        res["tft"]["by_origin_seed"] = {f"{o}|{s}": {k: (round(float(v), 6) if isinstance(v, float) else int(v))
                                                     for k, v in r.items()} for (o, s), r in by.iterrows()}
        res["tft"]["pooled_common_origins"] = {
            "mae_tft_mean_over_seeds": float(pc["mae_tft"].mean()),
            "mae_tft_per_seed": pc.groupby("seed")["mae_tft"].mean().round(6).to_dict(),
            "mae_persistence": float(pc.groupby(["origin", "subreddit"])["mae_persistence"].first().mean()),
            "mae_mean14": float(pc.groupby(["origin", "subreddit"])["mae_mean14"].first().mean())}
    return res


def main() -> None:
    IMP_OUT.mkdir(parents=True, exist_ok=True)
    g = Gate("B: forecast blends and rolling origins")
    res = {"B1": b1(g), "B2": b2(g)}
    (IMP_OUT / "forecast_improvement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    g.finish()


if __name__ == "__main__":
    main()
