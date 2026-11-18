"""
Improvement experiment A2: draw the 300-row DEV labelling sample (outputs/real_torrent_2026_08/improvement/
PREREGISTRATION.md). Same design as label_sample.py (TEST): bands on the stored toxicity_score, 100 per band, spread
evenly over subreddits, remainder at random within the band; seed 2027. Excludes the TEST ids, the spot-check ids and
p1ztawp. Draws twice to prove the same seed gives the same sample. Prints counts only, never a score.

Writes (gitignored, data/real_torrent_2026_08/improvement/):
    dev_sample_scored.csv   id, subreddit, band, toxicity_score, is_toxic, is_toxic_050, body (first 512 chars)
    dev_sample_blind.csv    id, subreddit, body — shuffled with the seed
    dev_sample_design.json  band populations of the DEV pool and sample sizes (for weights)

    python scripts/local/torrent_2026_08/improve_dev_sample.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from gates import Gate  # noqa: E402
from label_sample import band_of  # noqa: E402

DEV_SEED = 2027
IMP_DATA = C.DATA_DIR / "improvement"
IMP_DATA.mkdir(parents=True, exist_ok=True)


def draw(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    per_band = C.LABEL_N // len(C.LABEL_BANDS)
    picks = []
    for _, _, name in C.LABEL_BANDS:
        b = pool[pool["band"] == name]
        subs = sorted(b["subreddit"].unique())
        k = per_band // len(subs)
        chosen = []
        for s in subs:
            ids = b.index[b["subreddit"] == s].to_numpy()
            chosen.extend(rng.choice(ids, size=min(k, len(ids)), replace=False).tolist())
        rest = np.setdiff1d(b.index.to_numpy(), np.array(chosen, dtype=b.index.dtype))
        chosen.extend(rng.choice(rest, size=per_band - len(chosen), replace=False).tolist())
        picks.extend(chosen)
    return pool.loc[picks].copy()


def main() -> None:
    sc = pd.read_csv(C.SCORED, usecols=["id", "subreddit", "body", "toxicity_score", "is_toxic", "is_toxic_050"],
                     dtype={"id": str, "subreddit": str, "body": str}, keep_default_na=False)
    test_ids = set(pd.read_csv(C.DATA_DIR / "label_sample_scored.csv", dtype={"id": str})["id"])
    spot = set((C.DATA_DIR / "spotcheck_ids.txt").read_text(encoding="utf-8").split())
    spot |= set(sc.sample(n=5000, random_state=C.SPOTCHECK_SEED + 2).nlargest(5, "toxicity_score")["id"])
    excluded = test_ids | spot | {"p1ztawp"}
    pool = sc[~sc["id"].isin(excluded)].copy()
    pool["band"] = pool["toxicity_score"].map(band_of)

    a = draw(pool, DEV_SEED)
    b = draw(pool, DEV_SEED)
    smp = a
    smp["body"] = smp["body"].str.slice(0, C.CHAR_SLICE)
    smp[["id", "subreddit", "band", "toxicity_score", "is_toxic", "is_toxic_050", "body"]].to_csv(
        IMP_DATA / "dev_sample_scored.csv", index=False)
    blind = smp[["id", "subreddit", "body"]].sample(frac=1, random_state=DEV_SEED).reset_index(drop=True)
    blind.to_csv(IMP_DATA / "dev_sample_blind.csv", index=False)
    design = {"seed": DEV_SEED, "n": int(len(smp)), "bands": [list(x) for x in C.LABEL_BANDS],
              "population_per_band": pool["band"].value_counts().to_dict(),
              "sample_per_band": smp["band"].value_counts().to_dict(),
              "excluded_ids": len(excluded), "subreddits_in_sample": int(smp["subreddit"].nunique())}
    (IMP_DATA / "dev_sample_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")

    g = Gate("A2: DEV sample")
    g.check("same seed -> same DEV sample (ids and order)", list(a["id"]) == list(b["id"]),
            hashlib.sha256("\n".join(a["id"]).encode()).hexdigest()[:16])
    g.check("300 rows, 100 per band, ids unique", len(smp) == 300 and smp["id"].is_unique
            and all(v == 100 for v in design["sample_per_band"].values()), str(design["sample_per_band"]))
    g.check("no overlap with TEST", not (set(smp["id"]) & test_ids), f"TEST {len(test_ids)} ids")
    g.check("no excluded id in DEV", not (set(smp["id"]) & excluded), f"{len(excluded)} excluded")
    print(json.dumps(design, indent=2))
    g.finish()


if __name__ == "__main__":
    main()
