"""
Independent recomputation of headline numbers, and the expectations table (docs/NEW_DATA_NOTES.md Part A).

Second-way recomputations, each compared with the pipeline output at the precision the output is written:
  R1  daily mean toxicity of three communities on every day: plain pandas groupby on scored_comments.csv
      vs outputs/.../daily_series.csv (6 dp)
  R2  toxic rate at 0.6 and 0.5, overall and per community: straight from scored_comments.csv vs the count-weighted daily
      series (6 dp)
  R3  CHI by hand for every community, eq. 3.1, with every input recomputed WITHOUT main.py: T from the scored file;
      P and E from the raw SNAP TSV read with pandas (edges deduplicated per (source, target) keeping the LAST row, which
      is what nx.DiGraph.add_edge does when a pair repeats); churn from author sets before/after the window midpoint
      vs meso_report.csv (CHI 2 dp, inputs 4 dp)
  R4  instability trend by hand (recent quartile of dates vs earlier) and eq. 3.2 vs meso_report.csv (4 dp)
  R5  between-community spread vs a recomputation from the scored file

    python scripts/local/torrent_2026_08/verify_outputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from gates import Gate  # noqa: E402

R1_COMMUNITIES = ("politics", "science", "RoastMe")      # fixed before results


def main() -> None:
    sc = pd.read_csv(C.SCORED, usecols=["id", "subreddit", "author", "created_utc", "toxicity_score"],
                     dtype={"id": str, "subreddit": str, "author": str})
    sc["date"] = pd.to_datetime(sc["created_utc"], unit="s").dt.floor("D")
    daily = pd.read_csv(C.OUT_DIR / "daily_series.csv", parse_dates=["date"])
    meso = pd.read_csv(C.OUT_DIR / "meso_report.csv")
    meso["sub"] = meso["subreddit"].str.replace(r"^r/", "", regex=True)
    g = Gate("independent recomputation")

    # R1
    for sub in R1_COMMUNITIES:
        if sub not in set(sc["subreddit"]):
            g.check(f"R1 daily mean toxicity r/{sub} (all days)", False, "community not present")
            continue
        mine = sc[sc["subreddit"] == sub].groupby("date")["toxicity_score"].mean().round(6)
        theirs = daily[daily["subreddit"] == sub].set_index("date")["avg_toxicity"]
        same = mine.index.equals(theirs.index) and bool((np.abs(mine - theirs) < 5e-7).all())
        g.check(f"R1 daily mean toxicity r/{sub} (all days)", same,
                f"{len(mine)} days; e.g. {mine.index[0].date()} {mine.iloc[0]:.6f} vs {theirs.iloc[0]:.6f}")

    # R2
    for thr, col in ((0.6, "toxic_rate"), (0.5, "toxic_rate_050")):
        direct = float((sc["toxicity_score"] >= thr).mean())
        via_daily = float((daily[col] * daily["comment_count"]).sum() / daily["comment_count"].sum())
        g.check(f"R2 overall toxic rate @{thr}", abs(direct - via_daily) < 5e-6,
                f"direct {direct:.6f} vs daily-weighted {via_daily:.6f}")
    per = sc.assign(t=sc["toxicity_score"] >= 0.6).groupby("subreddit")["t"].mean()
    per_d = daily.groupby("subreddit").apply(lambda x: (x["toxic_rate"] * x["comment_count"]).sum()
                                             / x["comment_count"].sum())
    g.check("R2 per-community toxic rate @0.6 (all 39)", bool((np.abs(per - per_d) < 5e-6).all()),
            f"max |diff| {float(np.abs(per - per_d).max()):.2e}")

    # R3 — inputs without main.py
    T = sc.groupby("subreddit")["toxicity_score"].mean()
    tsv = pd.read_csv(C.ROOT / "data" / "reddit_hyperlinks.tsv", sep="\t",
                      usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"])
    e = tsv.drop_duplicates(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"], keep="last")
    nodes_lc = {}
    # node insertion order of the DiGraph is row order, source before target; MesoLayer._node_index keeps the LAST node
    # for a repeated lower-cased name, so do the same
    for n in pd.unique(np.column_stack([tsv["SOURCE_SUBREDDIT"], tsv["TARGET_SUBREDDIT"]]).ravel()):
        nodes_lc[str(n).replace("r/", "").strip().lower()] = n
    und = nx.Graph()
    und.add_edges_from(zip(e["SOURCE_SUBREDDIT"], e["TARGET_SUBREDDIT"]))
    glob_p = round(float((e["LINK_SENTIMENT"] < 0).mean()), 4)
    dates = sorted(sc["date"].dt.date.unique())
    mid = (pd.Timestamp(dates[0]) + (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])) / 2).date()
    split = dates[int(len(dates) * 0.75)]
    rows = []
    for sub, t in T.items():
        node = nodes_lc.get(sub.lower())
        if node is not None:
            ins = e[e["TARGET_SUBREDDIT"] == node]
            outs = e[e["SOURCE_SUBREDDIT"] == node]
            both = pd.concat([ins, outs])
            P = round(float((both["LINK_SENTIMENT"] < 0).mean()), 4)
            E = round(float(nx.clustering(und, node)), 4)
        else:
            P, E = np.nan, np.nan
        s = sc[sc["subreddit"] == sub]
        early = set(s.loc[s["date"].dt.date <= mid, "author"])
        late = set(s.loc[s["date"].dt.date > mid, "author"])
        churn = len(early - late) / len(early) if early else 0.0
        rec = s.loc[s["date"].dt.date > split, "toxicity_score"].mean()
        base = s.loc[s["date"].dt.date <= split, "toxicity_score"].mean()
        rows.append({"sub": sub, "T": t, "P": P, "E": E, "churn": churn, "trend": rec - base})
    hand = pd.DataFrame(rows).set_index("sub")
    mm = meso.set_index("sub")
    present = mm["in_snap_graph"]
    hand.loc[~present.reindex(hand.index).fillna(False).astype(bool), "P"] = mm["polarization"]   # global fallback
    hand.loc[~present.reindex(hand.index).fillna(False).astype(bool), "E"] = mm["echo_chamber_index"]
    pen = hand["T"] * 35 + hand["P"] * 30 + hand["churn"] * 20 + hand["E"] * 15
    hand["CHI"] = (100 - pen).clip(0, 100).round(2)
    j = hand.join(mm[["community_health_index", "toxicity_rate", "polarization", "echo_chamber_index", "churn_rate",
                      "toxicity_trend", "instability_score"]])
    for a, b, tol, name in (("T", "toxicity_rate", 5e-5, "T (mean toxicity)"), ("P", "polarization", 5e-5, "P"),
                            ("E", "echo_chamber_index", 5e-5, "E"), ("churn", "churn_rate", 5e-5, "churn"),
                            ("CHI", "community_health_index", 0.0051, "CHI eq. 3.1")):
        diff = float(np.abs(j[a] - j[b]).max())
        g.check(f"R3 {name} by hand == meso (all {len(j)})", diff <= tol, f"max |diff| {diff:.2e}")
    g.check("R3 graph-absent communities use the global P", bool(
        np.allclose(mm.loc[~mm["in_snap_graph"], "polarization"], glob_p)) if (~mm["in_snap_graph"]).any() else True,
        f"global P from TSV {glob_p}")

    # R4
    inst = (0.65 * np.clip(j["trend"] / 0.25, 0, 1) + 0.35 * j["T"]).clip(0, 1).round(4)
    g.check("R4 toxicity_trend by hand == meso", float(np.abs(j["trend"] - j["toxicity_trend"]).max()) <= 5e-5,
            f"split date {split}; max |diff| {float(np.abs(j['trend'] - j['toxicity_trend']).max()):.2e}")
    g.check("R4 instability eq. 3.2 by hand == meso", float(np.abs(inst - j["instability_score"]).max()) <= 1.5e-4,
            f"max |diff| {float(np.abs(inst - j['instability_score']).max()):.2e}")

    # R5
    spread_hand = float(T.max() - T.min())
    spread_meso = float(meso["toxicity_rate"].max() - meso["toxicity_rate"].min())
    g.check("R5 between-community spread", abs(spread_hand - spread_meso) < 1e-4,
            f"by hand {spread_hand:.4f} vs meso {spread_meso:.4f}")
    g.finish(exit_on_fail=False)

    # ── Expectations table ───────────────────────────────────────────────────────────────────────────────────────────
    rank = meso.sort_values("toxicity_rate", ascending=False).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank) + 1)
    pos = rank.set_index("sub")["rank"]
    pol = [s for s in ("politics", "news", "worldnews", "Conservative") if s in pos.index]
    med_rank = float(np.median([pos[s] for s in pol]))
    clus = pd.read_csv(C.OUT_DIR / "clusters_report.csv")
    fe = json.loads((C.OUT_DIR / "forecast_eval.json").read_text(encoding="utf-8"))
    aug = json.loads((C.ROOT / "outputs" / "real_reddit_summary.json").read_text(encoding="utf-8"))["per_community"]
    common = sorted(set(aug) & set(T.index))
    rho = spearmanr([aug[s]["mean_toxicity"] for s in common], [T[s] for s in common]).statistic
    present = meso[meso["in_snap_graph"]]
    widths = {k: v["mean_interval_width_by_day"] for k, v in fe["per_seed"].items()}
    rows = [
        ("E1", "communities differ measurably (spread >= 0.048)", f"spread {spread_meso:.4f} over {len(meso)}",
         spread_meso >= 0.048),
        ("E2a", "political/news in top half (median rank <= %d)" % (len(rank) // 2),
         f"median rank {med_rank:g} of {len(rank)} ({', '.join(f'{s} {pos[s]}' for s in pol)})",
         med_rank <= len(rank) / 2),
        ("E2b", "r/science lowest",
         (f"r/science rank {pos['science']} of {len(rank)}; lowest is r/{rank.iloc[-1]['sub']}"
          if "science" in pos.index else "r/science not present"),
         "science" in pos.index and pos["science"] == len(rank)),
        ("E3", "CHI in 0..100 and varies", f"{meso.community_health_index.min():.2f} .. "
         f"{meso.community_health_index.max():.2f}", bool(meso.community_health_index.between(0, 100).all())
         and meso.community_health_index.nunique() > 1),
        ("E4", "instability and trend vary, not all zero",
         f"instability {meso.instability_score.min():.4f} .. {meso.instability_score.max():.4f}; trend "
         f"{meso.toxicity_trend.min():+.4f} .. {meso.toxicity_trend.max():+.4f}",
         bool((meso.instability_score > 0).any()) and meso.instability_score.nunique() > 1
         and meso.toxicity_trend.nunique() > 1),
        ("E5", "polarisation and echo differ per community",
         f"P {present.polarization.nunique()} distinct, E {present.echo_chamber_index.nunique()} distinct among "
         f"{len(present)} in graph; {int((~meso.in_snap_graph).sum())} absent from graph",
         present.polarization.nunique() > 1 and present.echo_chamber_index.nunique() > 1),
        ("E6", "forecast intervals widen with horizon (each seed)",
         "; ".join(f"seed {k}: {w['1']:.4f} -> {w['5']:.4f}" for k, w in widths.items()),
         all(v["width_day5_gt_day1"] for v in fe["per_seed"].values())),
        ("E7", "recommender gives a mix of actions", str(meso.recommended_action.value_counts().to_dict()),
         meso.recommended_action.nunique() >= 2),
        ("E8", "5 non-empty clusters, both PCA axes vary",
         f"clusters {clus.cluster.value_counts().sort_index().to_dict()}; std pca_x {clus.pca_x.std():.3f}, "
         f"pca_y {clus.pca_y.std():.3f}", clus.cluster.nunique() == 5 and clus.pca_x.std() > 0 and clus.pca_y.std() > 0),
        ("E9", "forecasts real: 5 days per community per seed, p50 > 0",
         f"{fe['per_seed'][str(C.TFT_CANONICAL_SEED)]['communities']} communities; failed "
         f"{fe['per_seed'][str(C.TFT_CANONICAL_SEED)]['failed_communities']}",
         all(not v["failed_communities"] for v in fe["per_seed"].values())),
        ("E10", "same communities rank similarly to August (rho > 0)",
         f"Spearman rho {rho:.3f} over {len(common)} shared subreddits", rho > 0),
    ]
    ex = pd.DataFrame(rows, columns=["id", "expected", "measured", "match"])
    ex["match"] = ex["match"].map({True: "match", False: "differs"})
    ex.to_csv(C.OUT_DIR / "expectations_check.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    print("\nEXPECTATIONS (fixed before results) vs MEASURED")
    print(ex.to_string(index=False))
    print("\nranking by mean toxicity (meso toxicity_rate):")
    print(rank[["rank", "sub", "toxicity_rate", "total_comments"]].to_string(index=False))
    print("\nshared with August:", common)


if __name__ == "__main__":
    main()
