"""
Step 6c: agreement between toxic-bert's decisions and the labels in label_sample_labels.csv.

Reported under exactly this name: "agreement with LLM-assigned labels (single labeller, not human ground truth)".

  * decision = toxicity_score >= 0.6 (the configured threshold; primary) and >= 0.5 (secondary)
  * rows labelled "unclear" are excluded from the metrics and counted
  * precision, recall, F1, confusion matrix on the sample; bootstrap 95% intervals (LABEL_BOOTSTRAP resamples, seed)
  * because the sample over-represents high scores (equal bands), inverse-probability weighted precision / recall
    (weight = band population / band sample size) are reported alongside; these estimate agreement over the scored
    corpus, still against the same single labeller

Writes outputs/real_torrent_2026_08/label_agreement.json (aggregate numbers only).

    python scripts/local/torrent_2026_08/label_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

NAME = "agreement with LLM-assigned labels (single labeller, not human ground truth)"


def prf(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict:
    w = np.ones(len(y)) if w is None else w
    tp = float(w[(y == 1) & (p == 1)].sum()); fp = float(w[(y == 0) & (p == 1)].sum())  # noqa: E702
    fn = float(w[(y == 1) & (p == 0)].sum()); tn = float(w[(y == 0) & (p == 0)].sum())  # noqa: E702
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not np.isnan(prec + rec) else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    lab = pd.read_csv(C.DATA_DIR / "label_sample_labels.csv", dtype={"id": str})
    sc = pd.read_csv(C.DATA_DIR / "label_sample_scored.csv", dtype={"id": str})
    design = json.loads((C.DATA_DIR / "label_sample_design.json").read_text(encoding="utf-8"))
    assert set(lab["label"]) <= {"toxic", "not_toxic", "unclear"}, set(lab["label"])
    j = sc.merge(lab[["id", "label"]], on="id", how="inner", validate="one_to_one")
    assert len(j) == C.LABEL_N == len(lab), (len(j), len(lab))
    clear = j[j["label"] != "unclear"].reset_index(drop=True)
    y = (clear["label"] == "toxic").astype(int).to_numpy()
    wmap = {b: design["population_per_band"][b] / design["sample_per_band"][b] for b in design["sample_per_band"]}
    w = clear["band"].map(wmap).to_numpy()
    rng = np.random.default_rng(C.LABEL_SEED)
    out = {"name": NAME, "n_sample": int(len(j)), "n_unclear": int((j["label"] == "unclear").sum()),
           "n_used": int(len(clear)), "labels": j["label"].value_counts().to_dict(),
           "labels_by_band": pd.crosstab(j["band"], j["label"]).to_dict(), "bootstrap": C.LABEL_BOOTSTRAP}
    for thr, col in ((0.6, "is_toxic"), (0.5, "is_toxic_050")):
        p = clear[col].to_numpy()
        point = prf(y, p)
        boots = []
        for _ in range(C.LABEL_BOOTSTRAP):
            i = rng.integers(0, len(y), len(y))
            boots.append(prf(y[i], p[i]))
        ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))]
              for k in ("precision", "recall", "f1")}
        wt = prf(y, p, w)
        out[f"threshold_{thr}"] = {
            "precision": point["precision"], "recall": point["recall"], "f1": point["f1"], "ci95": ci,
            "confusion_matrix": {"rows=label, cols=model": [["", "model_toxic", "model_not_toxic"],
                                                            ["label_toxic", int(point["tp"]), int(point["fn"])],
                                                            ["label_not_toxic", int(point["fp"]), int(point["tn"])]]},
            "weighted_to_corpus": {"precision": wt["precision"], "recall": wt["recall"], "f1": wt["f1"]}}
    # Likely-cause check (word list fixed before looking at which rows disagree): how much of the model's
    # disagreement with the labels is untargeted profanity, which the guidelines label not_toxic?
    import re
    prof = re.compile(r"\b(fuck\w*|shit\w*|damn\w*|ass|asses|arse\w*|bitch\w*|hell|crap\w*|dick\w*|pussy|bastard\w*|"
                      r"wtf|motherf\w*)\b", re.I)
    has = clear["body"].fillna("").map(lambda t: bool(prof.search(t)))
    cell = {"fp": (y == 0) & (clear["is_toxic"].to_numpy() == 1), "tp": (y == 1) & (clear["is_toxic"].to_numpy() == 1),
            "tn": (y == 0) & (clear["is_toxic"].to_numpy() == 0), "fn": (y == 1) & (clear["is_toxic"].to_numpy() == 0)}
    out["profanity_share_by_cell_at_0.6"] = {k: {"n": int(v.sum()), "with_profanity": int(has[v].sum())}
                                             for k, v in cell.items()}
    (C.OUT_DIR / "label_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
