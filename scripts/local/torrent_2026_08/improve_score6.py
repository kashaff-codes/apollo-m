"""
Improvement experiment A3: score the 600 labelled comments (300 DEV + 300 TEST) on CPU with all six toxic-bert outputs
and compute the pre-registered variants V0–V3. Nothing else is scored.

Same call as the pipeline (top_k=None, truncation 256 tokens, first 512 characters, batches of 64 sorted by length);
device=-1 and CUDA hidden before torch is imported (the GPU produced silent errors).

Writes data/real_torrent_2026_08/improvement/six_label_scores.csv (ids and scores only, no text):
    id, set, band, stored_toxicity_score, toxic, severe_toxic, obscene, threat, insult, identity_hate, V0, V1, V2, V3

Gates: 600 = 300 DEV + 300 TEST with no overlap; CPU P(toxic) == stored score within 1e-4; no degenerate column;
DEV text equals the filtered CSV's first 512 characters for all 300 (10 printed for reading).

    python scripts/local/torrent_2026_08/improve_score6.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from gates import Gate, describe_numeric  # noqa: E402

LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
IMP_DATA = C.DATA_DIR / "improvement"
OUT = IMP_DATA / "six_label_scores.csv"


def variants(df: pd.DataFrame) -> pd.DataFrame:
    df["V0"] = df["toxic"]
    df["V1"] = df[["insult", "threat", "identity_hate", "severe_toxic"]].max(axis=1)
    df["V2"] = df[["insult", "threat", "identity_hate"]].mean(axis=1)
    df["V3"] = (df["toxic"] - df["obscene"]).clip(lower=0)
    return df


def main() -> None:
    import torch
    from transformers import pipeline
    assert not torch.cuda.is_available(), "GPU visible; refusing"
    dev = pd.read_csv(IMP_DATA / "dev_sample_scored.csv", dtype={"id": str, "body": str}, keep_default_na=False)
    test = pd.read_csv(C.DATA_DIR / "label_sample_scored.csv", dtype={"id": str, "body": str}, keep_default_na=False)
    dev["set"], test["set"] = "dev", "test"
    both = pd.concat([dev, test], ignore_index=True)

    clf = pipeline("text-classification", model=C.MODEL, top_k=None, truncation=True, max_length=C.MAX_LEN, device=-1)
    texts = [t[: C.CHAR_SLICE] for t in both["body"]]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)
    t0 = time.perf_counter()
    for s in range(0, len(order), C.BATCH):
        idx = order[s:s + C.BATCH]
        for i, r in zip(idx, clf([texts[i] for i in idx], batch_size=C.BATCH)):
            d = {x["label"].lower(): float(x["score"]) for x in r}
            out[i] = [d[k] for k in LABELS]          # KeyError if the label set ever differs
    secs = time.perf_counter() - t0
    sc = pd.DataFrame(out, columns=list(LABELS))
    res = pd.concat([both[["id", "set", "band", "toxicity_score"]].rename(
        columns={"toxicity_score": "stored_toxicity_score"}), sc], axis=1)
    res = variants(res)
    res.to_csv(OUT, index=False)
    rate = len(res) / secs
    (IMP_DATA / "cpu_rate.json").write_text(json.dumps({"comments": len(res), "seconds": round(secs, 2),
                                                         "comments_per_second": round(rate, 2), "threads":
                                                         torch.get_num_threads()}, indent=2), encoding="utf-8")

    g = Gate("A3: six-label CPU scores")
    g.check("600 = 300 DEV + 300 TEST, no overlap", len(res) == 600 and (res["set"] == "dev").sum() == 300
            and (res["set"] == "test").sum() == 300 and res["id"].is_unique,
            f"{len(res)} rows, {res['id'].nunique()} unique ids")
    diff = float((res["toxic"] - res["stored_toxicity_score"]).abs().max())
    g.check("CPU P(toxic) == stored toxicity_score (tol 1e-4)", diff <= 1e-4, f"max |diff| {diff:.2e}")
    for col in (*LABELS, "V0", "V1", "V2", "V3"):
        d = describe_numeric(res[col])
        g.check(f"{col} not degenerate", d["ok"], d["text"])
    filt = pd.read_csv(C.FILTERED_CSV, usecols=["id", "subreddit", "body"], dtype={"id": str, "body": str},
                       keep_default_na=False).set_index("id")
    blind = pd.read_csv(IMP_DATA / "dev_sample_blind.csv", dtype={"id": str, "body": str}, keep_default_na=False)
    same = all(blind.loc[i, "body"] == filt.loc[blind.loc[i, "id"], "body"][: C.CHAR_SLICE] and
               blind.loc[i, "subreddit"] == filt.loc[blind.loc[i, "id"], "subreddit"] for i in blind.index)
    g.check("DEV blind text/subreddit == filtered CSV (all 300)", same, "")
    print("\nSPOT CHECK — 10 DEV rows, blind text vs filtered CSV (seed 2027)")
    for _, r in blind.sample(n=10, random_state=2027).iterrows():
        raw = filt.loc[r["id"], "body"][: C.CHAR_SLICE]
        print(f"  [{r['id']}] r/{r['subreddit']}  identical={r['body'] == raw}")
        print(textwrap.indent(textwrap.fill(" ".join(raw.split())[:220], 110), "      "))
    print(f"\nCPU: {len(res)} comments in {secs:.1f}s = {rate:.1f}/s ({torch.get_num_threads()} threads)")
    g.finish()


if __name__ == "__main__":
    main()
