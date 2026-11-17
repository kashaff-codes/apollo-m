"""
Gate for step 3: are the scores complete, sane, real, and reproducible?

  * counts: scored rows == scoring sample == filtered rows; ids unique; per-subreddit counts equal the filter's;
  * no degenerate toxicity_score (finite, in [0,1], varies, not all zero);
  * spot check: 10 random scored comments (seed SPOTCHECK_SEED) printed with subreddit and score, for a human read;
    their ids are written to spotcheck_ids.txt and excluded from the labelling sample;
  * re-score: 20 random comments (seed SPOTCHECK_SEED + 1) re-scored in ONE batch of 20 on the GPU and again on the CPU
    (the repo's original device); every score must match the cache within 1e-4 (tolerance fixed before running).

    python scripts/local/torrent_2026_08/check_scores.py
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from gates import Gate, describe_numeric  # noqa: E402

TOL = 1e-4


def rescore(texts: list[str], device: int) -> list[float]:
    from transformers import pipeline
    clf = pipeline("text-classification", model=C.MODEL, top_k=None, truncation=True, max_length=C.MAX_LEN,
                   device=device)
    res = clf([t[: C.CHAR_SLICE] for t in texts], batch_size=len(texts))
    return [float({x["label"].lower(): x["score"] for x in r}["toxic"]) for r in res]


def main() -> None:
    sc = pd.read_csv(C.SCORED, dtype={"id": str, "subreddit": str, "author": str, "body": str}, keep_default_na=False)
    smp =pd.read_csv(C.SCORING_SAMPLE, usecols=["id"], dtype={"id": str})
    summ = json.loads(C.FILTER_SUMMARY.read_text(encoding="utf-8"))
    plan = json.loads((C.DATA_DIR / "scoring_plan.json").read_text(encoding="utf-8"))
    import scoring
    cache = scoring.read_cache(C.SCORE_CACHE)            # pass 1, verbatim (positions = write order)
    res = json.loads((C.DATA_DIR / "scoring_resolution.json").read_text(encoding="utf-8"))

    g = Gate("step 3: scores")
    g.check("every final score confirmed (pass1==pass2) or CPU-adjudicated",
            res["agree"] + res["cpu_adjudicated"] == len(sc) and set(sc["score_source"]) <= {
                "gpu_pass1_confirmed_by_pass2", "cpu_adjudicated"},
            f"agree {res['agree']:,}, CPU {res['cpu_adjudicated']:,}, non-finite p1 {res['pass1_nonfinite']} "
            f"p2 {res['pass2_nonfinite']}")
    g.check("scored rows == sample rows == plan", len(sc) == len(smp) == plan["comments_to_score"],
            f"{len(sc):,} / {len(smp):,} / {plan['comments_to_score']:,}")
    g.check("plan covers all filtered (N == cap)", plan["comments_to_score"] == summ["comments_kept"],
            f"{plan['comments_to_score']:,} of {summ['comments_kept']:,}")
    g.check("cache ids unique, cache == sample", cache["id"].is_unique and set(cache["id"]) == set(smp["id"]),
            f"cache {len(cache):,}")
    g.check("scored ids unique", sc["id"].is_unique, "")
    g.check("per-subreddit scored == filter kept", sc["subreddit"].value_counts().to_dict() == summ["kept_per_subreddit"],
            f"{sc['subreddit'].nunique()} subreddits")
    d = describe_numeric(sc["toxicity_score"])
    g.check("toxicity_score not degenerate", d["ok"], d["text"])
    g.check("toxicity_score within [0,1]", bool(sc["toxicity_score"].between(0, 1).all()), "")
    g.check("is_toxic == (score >= 0.6)", bool((sc["is_toxic"] == (sc["toxicity_score"] >= 0.6)).all()),
            f"toxic rate {sc['is_toxic'].mean():.4f}")
    g.check("is_toxic_050 == (score >= 0.5)", bool((sc["is_toxic_050"] == (sc["toxicity_score"] >= 0.5)).all()),
            f"toxic rate {sc['is_toxic_050'].mean():.4f}")

    # Spot check: 10 random real rows, printed in full enough to judge.
    spot = sc.sample(n=10, random_state=C.SPOTCHECK_SEED)
    (C.DATA_DIR / "spotcheck_ids.txt").write_text("\n".join(spot["id"]), encoding="utf-8")
    print("SPOT CHECK — 10 random scored comments (seed %d)" % C.SPOTCHECK_SEED)
    for _, r in spot.iterrows():
        body = " ".join(r["body"].split())
        print(f"\n  [{r['id']}] r/{r['subreddit']}  score {r['toxicity_score']:.4f}")
        print(textwrap.indent(textwrap.fill(body[:400] + ("…" if len(body) > 400 else ""), 110), "      "))
    # The five highest-scored of a random 5,000, so high scores can be read too.
    hi = sc.sample(n=5000, random_state=C.SPOTCHECK_SEED + 2).nlargest(5, "toxicity_score")
    print("\nHIGHEST 5 of a random 5,000 (seed %d)" % (C.SPOTCHECK_SEED + 2))
    for _, r in hi.iterrows():
        body = " ".join(r["body"].split())
        print(f"  r/{r['subreddit']}  {r['toxicity_score']:.4f}  {body[:160]}")

    # Re-score 20 in a single batch, GPU then CPU.
    rs = sc.sample(n=20, random_state=C.SPOTCHECK_SEED + 1)
    cached = rs["toxicity_score"].to_numpy()
    import torch
    gpu = np.array(rescore(rs["body"].tolist(), 0)) if torch.cuda.is_available() else None
    cpu = np.array(rescore(rs["body"].tolist(), -1))
    if gpu is not None:
        dg = float(np.max(np.abs(gpu - cached)))
        g.check("re-score 20 on GPU == cache (tol 1e-4)", dg <= TOL, f"max |diff| {dg:.2e}")
    dc = float(np.max(np.abs(cpu - cached)))
    g.check("re-score 20 on CPU == cache (tol 1e-4)", dc <= TOL, f"max |diff| {dc:.2e}")
    pd.DataFrame({"id": rs["id"].values, "subreddit": rs["subreddit"].values, "cached": cached,
                  "gpu_rescore": gpu if gpu is not None else np.nan, "cpu_rescore": cpu}).to_csv(
        C.DATA_DIR / "rescore_check.csv", index=False)

    # Wider CPU re-check because the GPU faulted during scoring (config.py, CPU_RECHECK_*).
    import re
    from transformers import pipeline
    log = (C.DATA_DIR / "scoring.log").read_text(encoding="utf-8", errors="replace")
    bounds = sorted({int(x.replace(",", "")) for x in re.findall(r"([\d,]+) already cached", log)} - {0})
    rng = np.random.default_rng(C.CPU_RECHECK_SEED)
    pick = {"random": rng.choice(len(cache), size=C.CPU_RECHECK_N, replace=False)}
    for b in bounds:
        lo = max(0, b - 20_000)
        pick[f"rows {lo}-{b} (before fault)"] = rng.choice(np.arange(lo, b), size=C.CPU_RECHECK_PER_CHUNK,
                                                          replace=False)
    clf_cpu = pipeline("text-classification", model=C.MODEL, top_k=None, truncation=True, max_length=C.MAX_LEN,
                       device=-1)
    body = sc.set_index("id")["body"]
    final = sc.set_index("id")["toxicity_score"]          # compare with the FINAL scores used downstream
    rows = []
    for name, idx in pick.items():
        ids = cache["id"].iloc[idx].to_numpy()
        new = np.array(scoring.score_texts(clf_cpu, body.loc[ids].tolist()))
        old = final.loc[ids].to_numpy()
        diff = float(np.max(np.abs(new - old)))
        g.check(f"CPU re-score {len(ids)} {name} (tol 1e-4)", diff <= TOL,
                f"max |diff| {diff:.2e}, mean |diff| {float(np.mean(np.abs(new - old))):.2e}")
        rows.append(pd.DataFrame({"set": name, "id": ids, "cached": old, "cpu_rescore": new}))
    pd.concat(rows).to_csv(C.DATA_DIR / "rescore_cpu_wide.csv", index=False)
    g.check("fault boundaries found in scoring.log", len(bounds) >= 2, f"cache rows at restart: {bounds}")
    g.finish()


if __name__ == "__main__":
    main()
