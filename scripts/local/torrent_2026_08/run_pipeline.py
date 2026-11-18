"""
Step 5: the repo's real-data path (run_real_pipeline.py), parameterised for the RC_2026-08 torrent corpus.

Same layers, same code, same order as run_real_pipeline.py — nothing in main.py, modules/ or models/ is changed:

    scored comments -> MesoLayer.analyze_all_communities (CHI, instability, polarisation + echo from the SNAP graph,
    churn) -> modules.gnn_infer.community_risk (gnn_risk) -> models/moderation_recommender.pkl (recommended_action)
    -> UnsupervisedLayer (K-Means + PCA, DBSCAN) -> AlertSystem.evaluate (alerts)

Differences from run_real_pipeline.py, all about WHERE things go, none about HOW they are computed:
  * input is data/real_torrent_2026_08/scored_comments.csv; outputs go to outputs/real_torrent_2026_08/;
  * alerts are written (AlertSystem.evaluate per community) instead of being left to the dashboard;
  * the daily series is built by build_daily.py (coverage rules) instead of here.

    python scripts/local/torrent_2026_08/run_pipeline.py
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

os.chdir(C.ROOT)                       # main.Config uses paths relative to the repo root (data/reddit_hyperlinks.tsv)
sys.path.insert(0, str(C.ROOT))
warnings.filterwarnings("ignore")
from main import AlertSystem, Config, DataLoader_APOLLO, MesoLayer, UnsupervisedLayer  # noqa: E402

from gates import Gate, degenerate_table, describe_numeric  # noqa: E402

logging.getLogger().setLevel(logging.INFO)
ACTIONS = {0: "NO_ACTION", 1: "MONITOR", 2: "WARN", 3: "INCREASE_MODERATION", 4: "EMERGENCY_INTERVENTION"}


def main() -> None:
    cfg = Config()
    assert cfg.TOXICITY_THRESHOLD == C.TOXIC_THRESHOLD, "config.TOXIC_THRESHOLD drifted from main.Config"
    df = pd.read_csv(C.SCORED, dtype={"id": str, "subreddit": str, "author": str, "body": str}, keep_default_na=False)
    df["subreddit"] = df["subreddit"].astype(str).apply(lambda s: s if s.startswith("r/") else f"r/{s}")
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=["created_utc", "subreddit", "toxicity_score"])
    print(f"torrent corpus: {len(df):,} comments ({n0 - len(df)} dropped for missing values), "
          f"{df['subreddit'].nunique()} communities, {df['created_utc'].dt.date.nunique()} days")

    graph = DataLoader_APOLLO(cfg).load_hyperlink_graph()
    meso = MesoLayer(cfg)
    results = meso.analyze_all_communities(df, graph)
    m = pd.DataFrame(results).sort_values("community_health_index").reset_index(drop=True)

    from modules.gnn_infer import community_risk
    risk = community_risk(graph, results)
    key = m["subreddit"].str.replace(r"^r/", "", regex=True).str.lower()
    raw = key.map(risk)
    m["gnn_risk_raw"] = raw.round(6)
    m["gnn_risk"] = raw.rank(pct=True).round(4)

    with open(C.ROOT / "models" / "moderation_recommender.pkl", "rb") as fh:
        rec = pickle.load(fh)
    X = m[["community_health_index", "toxicity_rate", "polarization", "churn_rate", "echo_chamber_index"]].to_numpy()
    m["recommended_action"] = [ACTIONS.get(int(a), "UNKNOWN") for a in rec.predict(X)]

    in_graph = key.map(lambda k: meso._lookup(graph, k) is not None)
    m["in_snap_graph"] = in_graph.values
    m.to_csv(C.OUT_DIR / "meso_report.csv", index=False)

    unsup = UnsupervisedLayer(cfg)
    feat = unsup.detect_outlier_communities(unsup.cluster_communities(m.copy()))
    feat.to_csv(C.OUT_DIR / "clusters_report.csv", index=False)

    alerts = [AlertSystem.evaluate(r) for r in m.to_dict("records")]
    (C.OUT_DIR / "alerts.json").write_text(json.dumps(alerts, indent=2), encoding="utf-8")

    cov = pd.read_csv(C.OUT_DIR / "coverage.csv")
    (C.OUT_DIR / "data_provenance.json").write_text(json.dumps({
        "active_dataset": "real Reddit, Academic Torrents monthly dump RC_2026-08 (Arctic Shift)",
        "infohash": C.INFOHASH,
        "dump_sha256": json.loads(C.FILTER_SUMMARY.read_text(encoding="utf-8")).get("input_sha256"),
        "filter": {"subreddits": 39, "cap_per_subreddit_per_day": C.FILTER_CAP, "seed": C.FILTER_SEED,
                   "window": [C.WINDOW_START, C.WINDOW_END]},
        "scoring": {"model": C.MODEL, "comments_scored": int(len(df)), "threshold": C.TOXIC_THRESHOLD},
        "comments": int(len(df)),
        "communities_analysed": int(len(m)),
        "communities_forecast": int(cov["qualifies_forecast"].sum()),
        "authors": int(df["author"].nunique()),
        "days": int(df["created_utc"].dt.date.nunique()),
        "date_range": [str(df["created_utc"].min().date()), str(df["created_utc"].max().date())],
        "note": "Every value in this folder is computed from this corpus by scripts/local/torrent_2026_08/. "
                "Comments are a capped sample (<= 1000 per subreddit per day), so churn measures author overlap "
                "within the sample. Benchmark (planted ground truth) results are not in this folder.",
    }, indent=2), encoding="utf-8")

    # ── Gate ─────────────────────────────────────────────────────────────────────────────────────────────────────────
    g = Gate("step 5: meso, clusters, alerts")
    g.check("every community passed the meso minimum", len(m) == df["subreddit"].nunique(),
            f"{len(m)} of {df['subreddit'].nunique()} (MIN_COMMENTS_PER_SUB = {MesoLayer.MIN_COMMENTS_PER_SUB})")
    g.check("total_comments sum == comments scored", int(m["total_comments"].sum()) == len(df),
            f"{int(m['total_comments'].sum()):,}")
    nan_gnn = set(m.loc[m["gnn_risk_raw"].isna(), "subreddit"])
    absent = set(m.loc[~m["in_snap_graph"], "subreddit"])
    g.check("gnn_risk missing ONLY for communities absent from the SNAP graph", nan_gnn == absent,
            f"absent from graph: {sorted(absent)}")
    present = m[m["in_snap_graph"]]
    degenerate_table(g, m.drop(columns=["gnn_risk_raw", "gnn_risk"]), "meso", zero_ok=("toxicity_trend",),
                     constant_ok=("in_snap_graph",))
    degenerate_table(g, present[["gnn_risk_raw", "gnn_risk"]], "meso[in graph]")
    for col in ("polarization", "echo_chamber_index"):
        g.check(f"E5 {col} differs per community (in graph)", present[col].nunique() > 1,
                f"{present[col].nunique()} distinct among {len(present)}")
    chi = m["community_health_index"]
    g.check("E3 CHI within 0..100 and varies", bool(chi.between(0, 100).all()) and chi.nunique() > 1,
            f"{chi.min():.2f} .. {chi.max():.2f}")
    inst = m["instability_score"]
    g.check("E4 instability not all zero and varies", bool((inst > 0).any()) and inst.nunique() > 1,
            f"{inst.min():.4f} .. {inst.max():.4f}, {int((inst > 0).sum())} > 0")
    tr = m["toxicity_trend"]
    g.check("E4 toxicity_trend varies", tr.nunique() > 1,
            f"{tr.min():+.4f} .. {tr.max():+.4f} ({int((tr > 0).sum())} rising, {int((tr < 0).sum())} falling)")
    acts = m["recommended_action"].value_counts().to_dict()
    g.check("E7 recommender gives a mix of actions", len(acts) >= 2, str(acts))
    cl = feat["cluster"].value_counts().sort_index()
    g.check("E8 five non-empty K-Means clusters", len(cl) == cfg.N_CLUSTERS and bool((cl > 0).all()), str(cl.to_dict()))
    for col in ("pca_x", "pca_y"):
        d = describe_numeric(feat[col])
        g.check(f"E8 PCA axis {col} varies", d["ok"] and feat[col].std() > 0, f"std {feat[col].std():.4f}; {d['text']}")
    degenerate_table(g, feat[["cluster", "dbscan_label", "is_outlier"]], "clusters", zero_ok=("is_outlier",),
                     constant_ok=("dbscan_label", "is_outlier"))
    lv = pd.Series([a["alert_level"] for a in alerts]).value_counts().to_dict()
    g.check("alerts: one per community", len(alerts) == len(m), str(lv))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(m.drop(columns=["timestamp"]).to_string())
    print("\nclusters:", cl.to_dict(), " DBSCAN outliers:", int(feat["is_outlier"].sum()))
    g.finish()


if __name__ == "__main__":
    main()
