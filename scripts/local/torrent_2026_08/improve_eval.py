"""
Improvement experiment A3/A4: choose each variant's threshold on DEV, then evaluate on TEST exactly once.
Implements outputs/real_torrent_2026_08/improvement/PREREGISTRATION.md, Part A.

    python scripts/local/torrent_2026_08/improve_eval.py dev    # DEV only: thresholds + DEV metrics
    python scripts/local/torrent_2026_08/improve_eval.py test   # TEST once; refuses if test_result.json exists

Weights: w = band population / 100 (rows drawn per band), the same weights as label_agreement.py; unclear labels are
excluded. Bootstrap: resample within each band, 2,000 resamples, seed 2028; percentile 95% intervals.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

IMP_DATA = C.DATA_DIR / "improvement"
IMP_OUT = C.OUT_DIR / "improvement"
VARIANTS = ("V0", "V1", "V2", "V3")
BOOT, BOOT_SEED = 2000, 2028


def load(which: str) -> pd.DataFrame:
    sc = pd.read_csv(IMP_DATA / "six_label_scores.csv", dtype={"id": str})
    if which == "dev":
        lab = pd.read_csv(IMP_DATA / "dev_sample_labels.csv", dtype={"id": str})
        design = json.loads((IMP_DATA / "dev_sample_design.json").read_text(encoding="utf-8"))
    else:
        lab = pd.read_csv(C.DATA_DIR / "label_sample_labels.csv", dtype={"id": str})
        design = json.loads((C.DATA_DIR / "label_sample_design.json").read_text(encoding="utf-8"))
    d = sc[sc["set"] == which].merge(lab, on="id", how="inner", validate="one_to_one")
    assert len(d) == 300, len(d)
    n_unclear = int((d["label"] == "unclear").sum())
    d = d[d["label"] != "unclear"].reset_index(drop=True)
    d["y"] = (d["label"] == "toxic").astype(int)
    d["w"] = d["band"].map({b: design["population_per_band"][b] / design["sample_per_band"][b]
                            for b in design["sample_per_band"]})
    d.attrs["n_unclear"] = n_unclear
    return d


def prf(y, p, w) -> tuple[float, float, float]:
    tp = float(w[(y == 1) & p].sum()); fp = float(w[(y == 0) & p].sum()); fn = float(w[(y == 1) & ~p].sum())  # noqa: E702
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and not np.isnan(prec + rec) else np.nan
    return prec, rec, f1


def metrics(d: pd.DataFrame, col: str, t: float | None) -> dict:
    y, s, w = d["y"].to_numpy(), d[col].to_numpy(), d["w"].to_numpy()
    one = np.ones(len(y))
    out = {"ap_weighted": float(average_precision_score(y, s, sample_weight=w)),
           "ap_unweighted": float(average_precision_score(y, s)),
           "roc_auc_weighted": float(roc_auc_score(y, s, sample_weight=w)),
           "roc_auc_unweighted": float(roc_auc_score(y, s))}
    if t is not None:
        p = s >= t
        out["precision_weighted"], out["recall_weighted"], out["f1_weighted"] = prf(y, p, w)
        out["precision_unweighted"], out["recall_unweighted"], out["f1_unweighted"] = prf(y, p, one)
        out["predicted_positive"] = int(p.sum())
    return out


def choose_threshold(d: pd.DataFrame, col: str) -> tuple[float, float]:
    y, s, w = d["y"].to_numpy(), d[col].to_numpy(), d["w"].to_numpy()
    best_t, best_f = None, -1.0
    for t in np.sort(np.unique(s)):                       # ascending -> ties keep the smallest t
        f = prf(y, s >= t, w)[2]
        if not np.isnan(f) and f > best_f + 1e-15:
            best_t, best_f = float(t), float(f)
    return best_t, best_f


def cmd_dev() -> None:
    d = load("dev")
    res = {"n_used": int(len(d)), "n_unclear": d.attrs["n_unclear"], "n_toxic": int(d["y"].sum()), "variants": {}}
    for v in VARIANTS:
        t, f = choose_threshold(d, v)
        res["variants"][v] = {"threshold": t, "dev_f1_weighted_at_threshold": f, "dev": metrics(d, v, t)}
    res["V0_fixed"] = {str(t): metrics(d, "V0", t) for t in (0.5, 0.6)}
    (IMP_OUT / "dev_thresholds.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def cmd_test() -> None:
    out_path = IMP_OUT / "test_result.json"
    if out_path.exists():
        sys.exit("test_result.json exists: TEST has already been used once; refusing to evaluate again")
    dev = json.loads((IMP_OUT / "dev_thresholds.json").read_text(encoding="utf-8"))
    th = {v: dev["variants"][v]["threshold"] for v in VARIANTS}
    d = load("test")
    point = {v: metrics(d, v, th[v]) for v in VARIANTS}
    fixed = {str(t): metrics(d, "V0", t) for t in (0.5, 0.6)}

    rng = np.random.default_rng(BOOT_SEED)
    band_idx = [d.index[d["band"] == b].to_numpy() for b in sorted(d["band"].unique())]
    keys = ("f1_weighted", "precision_weighted", "recall_weighted", "ap_weighted", "f1_unweighted", "ap_unweighted",
            "roc_auc_weighted")
    draws = {v: {k: [] for k in keys} for v in VARIANTS}
    for _ in range(BOOT):
        i = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in band_idx])
        b = d.loc[i].reset_index(drop=True)
        if b["y"].nunique() < 2:
            continue
        for v in VARIANTS:
            m = metrics(b, v, th[v])
            for k in keys:
                draws[v][k].append(m[k])
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    res = {"n_used": int(len(d)), "n_unclear": d.attrs["n_unclear"], "n_toxic": int(d["y"].sum()),
           "thresholds_from_dev": th, "bootstrap": BOOT, "seed": BOOT_SEED, "variants": {}, "V0_fixed": fixed}
    v0f1 = np.array(draws["V0"]["f1_weighted"])
    v0ap = np.array(draws["V0"]["ap_weighted"])
    v0_width = ci(v0f1)[1] - ci(v0f1)[0]
    for v in VARIANTS:
        e = {"point": point[v], "ci95": {k: ci(draws[v][k]) for k in keys}}
        if v != "V0":
            df1 = np.array(draws[v]["f1_weighted"]) - v0f1
            dap = np.array(draws[v]["ap_weighted"]) - v0ap
            d_point = point[v]["f1_weighted"] - point["V0"]["f1_weighted"]
            e["diff_vs_V0"] = {
                "f1_weighted_point": d_point, "f1_weighted_ci95": ci(df1),
                "ap_weighted_point": point[v]["ap_weighted"] - point["V0"]["ap_weighted"], "ap_weighted_ci95": ci(dap),
                "V0_f1_weighted_ci_width": v0_width,
                "rule1_ci_above_0": bool(ci(df1)[0] > 0),
                "rule2_diff_exceeds_V0_ci_width": bool(d_point > v0_width)}
            e["improvement"] = bool(e["diff_vs_V0"]["rule1_ci_above_0"] and e["diff_vs_V0"]["rule2_diff_exceeds_V0_ci_width"])
        res["variants"][v] = e
    res["any_improvement"] = any(res["variants"][v].get("improvement", False) for v in VARIANTS if v != "V0")
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    rows = []
    for v in VARIANTS:
        p, c = res["variants"][v]["point"], res["variants"][v]["ci95"]
        rows.append({"variant": v, "threshold_dev": th[v],
                     "f1_weighted": p["f1_weighted"], "f1_weighted_ci": c["f1_weighted"],
                     "precision_weighted": p["precision_weighted"], "recall_weighted": p["recall_weighted"],
                     "ap_weighted": p["ap_weighted"], "ap_weighted_ci": c["ap_weighted"],
                     "roc_auc_weighted": p["roc_auc_weighted"], "f1_unweighted": p["f1_unweighted"],
                     "precision_unweighted": p["precision_unweighted"], "recall_unweighted": p["recall_unweighted"],
                     "ap_unweighted": p["ap_unweighted"], "roc_auc_unweighted": p["roc_auc_unweighted"],
                     "predicted_positive": p["predicted_positive"],
                     "improvement_vs_V0": res["variants"][v].get("improvement", "")})
    pd.DataFrame(rows).to_csv(IMP_OUT / "toxicity_variants_test.csv", index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"dev": cmd_dev, "test": cmd_test}[sys.argv[1]]()
