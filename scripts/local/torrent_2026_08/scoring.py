"""
Step 3: score the filtered RC_2026-08 comments with unitary/toxic-bert, within a fixed time budget.

Subcommands (run in this order):
    bench   time the model on BENCH_N comments (seeded) with exactly the batching used for scoring; writes the rate
    plan    apply the budget rule in config.py to the measured rate -> per-subreddit-per-day quota N (scoring_plan.json)
    sample  seeded per-(subreddit, day) sample of min(kept, N) comments -> scoring_sample.csv (no scores)
    score   score every sampled comment whose id is not yet in score_cache.csv; appends, never rewrites the cache
    join    scoring_sample.csv + score_cache.csv -> scored_comments.csv (a new file; nothing is overwritten in place)

The model call is the repo's (score_real_reddit.py / MicroLayer): text-classification pipeline, top_k=None,
truncation at 256 tokens, first 512 characters of the body, score = probability of the "toxic" label. The only
difference is device=0 (GPU) and length-sorted batches for speed; both change scores only at float rounding level,
which the re-score check in quality.py measures.

    python scripts/local/torrent_2026_08/scoring.py bench|plan|sample|score|join
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

PLAN = C.DATA_DIR / "scoring_plan.json"


def load_filtered() -> pd.DataFrame:
    df = pd.read_csv(C.FILTERED_CSV, dtype={"id": str, "subreddit": str, "author": str, "body": str},
                     keep_default_na=False)
    df["day"] = pd.to_datetime(df["created_utc"].astype(np.int64), unit="s", utc=True).dt.strftime("%Y-%m-%d")
    return df


def make_classifier():
    import torch
    from transformers import pipeline
    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this venv; refusing to start a multi-hour CPU run silently.")
    return pipeline("text-classification", model=C.MODEL, top_k=None, truncation=True, max_length=C.MAX_LEN,
                    device=0)


def score_texts(clf, texts: list[str]) -> list[float]:
    """Score in length-sorted batches of C.BATCH; returns scores in the input order."""
    texts = [t[: C.CHAR_SLICE] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [0.0] * len(texts)
    for s in range(0, len(order), C.BATCH):
        idx = order[s:s + C.BATCH]
        res = clf([texts[i] for i in idx], batch_size=C.BATCH)
        for i, r in zip(idx, res):
            d = {x["label"].lower(): x["score"] for x in r}
            out[i] = float(d["toxic"])       # KeyError (not a silent 0.0) if the label set ever differs
    return out


def cmd_bench(_a) -> None:
    import torch
    df = load_filtered()
    smp = df.sample(n=C.BENCH_N, random_state=C.BENCH_SEED)
    clf = make_classifier()
    score_texts(clf, df.sample(n=128, random_state=1)["body"].tolist())     # warm-up, not timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    scores = score_texts(clf, smp["body"].tolist())
    torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    rate = C.BENCH_N / secs
    res = {"n": C.BENCH_N, "seed": C.BENCH_SEED, "seconds": round(secs, 3), "comments_per_second": round(rate, 2),
           "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "batch": C.BATCH,
           "max_len": C.MAX_LEN, "char_slice": C.CHAR_SLICE,
           "score_min": round(min(scores), 6), "score_max": round(max(scores), 6),
           "score_mean": round(float(np.mean(scores)), 6)}
    C.BENCH_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def cmd_plan(_a) -> None:
    bench = json.loads(C.BENCH_JSON.read_text(encoding="utf-8"))
    rate = bench["comments_per_second"]
    budget = int(np.floor(rate * C.SCORING_TARGET_SECONDS))
    s = json.loads(C.FILTER_SUMMARY.read_text(encoding="utf-8"))
    kept = np.array([v["kept"] for v in s["per_subreddit_day"].values()])
    total = lambda n: int(np.minimum(kept, n).sum())  # noqa: E731
    n = max((k for k in range(1, C.FILTER_CAP + 1) if total(k) <= budget), default=0)
    plan = {"rate_comments_per_second": rate, "target_seconds": C.SCORING_TARGET_SECONDS, "budget_comments": budget,
            "rule": "largest N <= cap with sum_cells min(kept_cell, N) <= floor(rate * target_seconds)",
            "quota_per_subreddit_per_day": n, "comments_to_score": total(n),
            "expected_seconds": round(total(n) / rate, 1), "filtered_comments": int(kept.sum()),
            "cells": int(len(kept)), "sample_seed": C.SCORING_SAMPLE_SEED}
    PLAN.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))


def cmd_sample(_a) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    n = plan["quota_per_subreddit_per_day"]
    df = load_filtered()
    parts = []
    for i, ((sub, day), g) in enumerate(df.groupby(["subreddit", "day"], sort=True)):
        parts.append(g if len(g) <= n else g.sample(n=n, random_state=C.SCORING_SAMPLE_SEED))
    smp = pd.concat(parts).sort_values(["subreddit", "created_utc", "id"]).reset_index(drop=True)
    assert len(smp) == plan["comments_to_score"], (len(smp), plan["comments_to_score"])
    smp.drop(columns=["day"]).to_csv(C.SCORING_SAMPLE, index=False)
    print(f"scoring sample: {len(smp):,} comments (quota {n}/sub/day) -> {C.SCORING_SAMPLE}")


def read_cache(path: Path) -> pd.DataFrame:
    """id kept verbatim (no NA parsing); an empty or unparseable score becomes NaN and is reported, never dropped."""
    c = pd.read_csv(path, dtype={"id": str, "toxicity_score": str}, keep_default_na=False)
    c["toxicity_score"] = pd.to_numeric(c["toxicity_score"], errors="coerce")
    return c


def cmd_score(a) -> None:
    cache_path = C.SCORE_CACHE_PASS2 if a.pass2 else C.SCORE_CACHE
    smp = pd.read_csv(C.SCORING_SAMPLE, dtype={"id": str, "body": str}, keep_default_na=False)
    done: set[str] = set()
    if cache_path.exists():
        done = set(read_cache(cache_path)["id"])
    todo = smp[~smp["id"].isin(done)]
    print(f"[{cache_path.name}] {len(smp):,} in sample, {len(done):,} already cached, {len(todo):,} to score",
          flush=True)
    if todo.empty:
        return
    clf = make_classifier()
    t0 = time.time()
    chunk = a.chunk
    new_file = not cache_path.exists()
    for s in range(0, len(todo), chunk):
        part = todo.iloc[s:s + chunk]
        sc = score_texts(clf, part["body"].tolist())
        pd.DataFrame({"id": part["id"].values, "toxicity_score": np.round(sc, 8)}).to_csv(
            cache_path, mode="a", header=new_file, index=False)
        new_file = False
        n_done = s + len(part)
        el = time.time() - t0
        print(f"  {n_done:,}/{len(todo):,} scored, {el:,.0f}s, {n_done / el:,.1f}/s, "
              f"eta {(len(todo) - n_done) / max(n_done / el, 1e-9):,.0f}s", flush=True)
    print(f"done in {time.time() - t0:,.0f}s")


def compare_passes() -> pd.DataFrame:
    """Pass 1 vs pass 2 per comment, with the agreement flag of config.py."""
    p1 = read_cache(C.SCORE_CACHE).rename(columns={"toxicity_score": "p1"})
    p2 = read_cache(C.SCORE_CACHE_PASS2).rename(columns={"toxicity_score": "p2"})
    for name, c in (("pass 1", p1), ("pass 2", p2)):
        if not c["id"].is_unique:
            sys.exit(f"{name} cache has duplicate ids")
    m = p1.merge(p2, on="id", how="outer", validate="one_to_one")
    m["agree"] = np.isfinite(m["p1"]) & np.isfinite(m["p2"]) & ((m["p1"] - m["p2"]).abs() <= C.SCORE_AGREE_TOL)
    return m


def cmd_adjudicate(_a) -> None:
    m = compare_passes()
    dis = m[~m["agree"]]
    d = (m["p1"] - m["p2"]).abs()
    print(f"compared {len(m):,}: agree {int(m['agree'].sum()):,}, disagree {len(dis):,} "
          f"(non-finite pass1 {int((~np.isfinite(m['p1'])).sum())}, pass2 {int((~np.isfinite(m['p2'])).sum())}); "
          f"max |p1-p2| among agreeing {float(d[m['agree']].max()):.2e}", flush=True)
    done = set(read_cache(C.SCORE_CACHE_CPU)["id"]) if C.SCORE_CACHE_CPU.exists() else set()
    todo_ids = [i for i in dis["id"] if i not in done]
    if not todo_ids:
        print("nothing to adjudicate")
        return
    from transformers import pipeline
    clf = pipeline("text-classification", model=C.MODEL, top_k=None, truncation=True, max_length=C.MAX_LEN,
                   device=-1)
    smp = pd.read_csv(C.SCORING_SAMPLE, usecols=["id", "body"], dtype={"id": str, "body": str},
                      keep_default_na=False).set_index("id")
    t0 = time.time()
    for s in range(0, len(todo_ids), 2000):
        ids = todo_ids[s:s + 2000]
        sc = score_texts(clf, smp.loc[ids, "body"].tolist())
        pd.DataFrame({"id": ids, "toxicity_score": np.round(sc, 8)}).to_csv(
            C.SCORE_CACHE_CPU, mode="a", header=not C.SCORE_CACHE_CPU.exists(), index=False)
        print(f"  CPU {s + len(ids):,}/{len(todo_ids):,} in {time.time() - t0:,.0f}s", flush=True)


def cmd_join(_a) -> None:
    smp = pd.read_csv(C.SCORING_SAMPLE, dtype={"id": str, "subreddit": str, "author": str, "body": str},
                      keep_default_na=False)
    m = compare_passes()
    if len(m) != len(smp) or m["p1"].isna().all() or set(m["id"]) != set(smp["id"]):
        sys.exit("pass 1 / pass 2 caches do not both cover exactly the scoring sample")
    cpu = read_cache(C.SCORE_CACHE_CPU) if C.SCORE_CACHE_CPU.exists() else pd.DataFrame(columns=["id", "toxicity_score"])
    m = m.merge(cpu.rename(columns={"toxicity_score": "cpu"}), on="id", how="left", validate="one_to_one")
    need = ~m["agree"]
    if (need & ~np.isfinite(m["cpu"].astype(float))).any():
        sys.exit(f"{int((need & ~np.isfinite(m['cpu'].astype(float))).sum())} disagreeing comments lack a finite CPU "
                 "score; run `adjudicate` first")
    m["toxicity_score"] = np.where(m["agree"], m["p1"], m["cpu"].astype(float))
    m["score_source"] = np.where(m["agree"], "gpu_pass1_confirmed_by_pass2", "cpu_adjudicated")
    res = {"comments": int(len(m)), "agree": int(m["agree"].sum()), "cpu_adjudicated": int(need.sum()),
           "pass1_nonfinite": int((~np.isfinite(m["p1"])).sum()), "pass2_nonfinite": int((~np.isfinite(m["p2"])).sum()),
           "tolerance": C.SCORE_AGREE_TOL,
           "max_abs_p1_p2_agreeing": float((m["p1"] - m["p2"]).abs()[m["agree"]].max()),
           "cpu_vs_pass1_on_adjudicated_abs_diff": (
               (m.loc[need, "cpu"].astype(float) - m.loc[need, "p1"]).abs().describe().to_dict() if need.any() else {}),
           "cpu_vs_pass2_on_adjudicated_abs_diff": (
               (m.loc[need, "cpu"].astype(float) - m.loc[need, "p2"]).abs().describe().to_dict() if need.any() else {})}
    (C.DATA_DIR / "scoring_resolution.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=float))
    out = smp.merge(m[["id", "toxicity_score", "score_source"]], on="id", how="left", validate="one_to_one")
    if out["toxicity_score"].isna().any():
        sys.exit("unresolved scores remain")
    out["is_toxic"] = (out["toxicity_score"] >= C.TOXIC_THRESHOLD).astype(int)
    out["is_toxic_050"] = (out["toxicity_score"] >= C.TOXIC_THRESHOLD_ALT).astype(int)
    out.to_csv(C.SCORED, index=False)
    print(f"scored file: {len(out):,} rows -> {C.SCORED}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "plan", "sample", "score", "adjudicate", "join"])
    ap.add_argument("--chunk", type=int, default=20_000)
    ap.add_argument("--pass2", action="store_true", help="score into score_cache_pass2.csv (verification pass)")
    a = ap.parse_args()
    {"bench": cmd_bench, "plan": cmd_plan, "sample": cmd_sample, "score": cmd_score, "adjudicate": cmd_adjudicate,
     "join": cmd_join}[a.cmd](a)


if __name__ == "__main__":
    main()
