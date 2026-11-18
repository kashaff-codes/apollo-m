"""
Write outputs/web_evidence.json — the findings the public web view shows.

    python scripts/export_web_evidence.py

The web view (web/, Next.js on Vercel) reads this file from GitHub rather than
recomputing anything, and the console (dashboard/) computes the same figures
with the same functions (dashboard/data.py). Generating one from the other is
what keeps the two public surfaces from quoting different numbers.
tests/test_web_evidence.py fails if the committed file is stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dashboard")]

import apollo_datasets as AD  # noqa: E402
import data as D  # noqa: E402

OUT = ROOT / "outputs" / "web_evidence.json"


def build() -> dict:
    corpora = []
    for d in AD.DATASETS.values():
        f = D.corpus_facts(d)
        corpora.append({"key": d.key, "label": d.label, "kind": d.kind, "window": d.window,
                        "communities": f["communities"], "comments": f["comments"], "days": f["days"],
                        "forecast": d.forecast, "credit": d.credit, "summary": d.summary})

    table = D.across_corpora(AD.real())
    real = [d.key for d in AD.real()]
    agreement = []
    for i, a in enumerate(real):
        for b in real[i + 1:]:
            rho, n = D.rank_agreement(table, a, b)
            if n >= 3:
                agreement.append({"a": a, "b": b, "rho": round(rho, 3), "n": n})

    backtest = {k: {m: (round(v, 5) if isinstance(v, float) else v) for m, v in D.backtest_summary(AD.get(k)).items()}
                for k in AD.DATASETS if D.backtest_summary(AD.get(k))}

    labels = D.evidence(AD.get("real_torrent_2026_08")).get("label_agreement", {})
    t6 = labels.get("threshold_0.6", {})
    fp = labels.get("profanity_share_by_cell_at_0.6", {}).get("fp", {})
    label_agreement = {"n": labels.get("n_used"), "precision": round(t6.get("precision", 0), 3),
                       "recall": round(t6.get("recall", 0), 3), "f1": round(t6.get("f1", 0), 3),
                       "precision_ci95": [round(x, 3) for x in t6.get("ci95", {}).get("precision", [])],
                       "false_positives": fp.get("n"), "false_positives_with_profanity": fp.get("with_profanity")}

    v = json.loads((ROOT / "outputs" / "validation.json").read_text(encoding="utf-8"))
    benchmark = {"chi_roc_auc": v["detection"]["chi_roc_auc"],
                 "toxicity_only_roc_auc": v["detection"]["toxicity_only_roc_auc"],
                 "tft_slope_roc_auc": v["forecast_tft"]["slope_roc_auc"]}

    fwd = ROOT / "outputs" / "forward_test" / "summary.json"
    return {"corpora": corpora, "agreement": agreement, "backtest": backtest,
            "label_agreement": label_agreement, "benchmark": benchmark,
            "forward_test": json.loads(fwd.read_text(encoding="utf-8")) if fwd.exists() else None}


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
