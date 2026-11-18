"""
Step 6a: draw the 300-row labelling sample (design fixed in config.py before any score existed).

Stratified by toxicity-score band (low < 0.1 <= mid < 0.6 <= high; 100 rows each) and, within a band, spread evenly over
subreddits with a seeded draw; the remainder of each band is drawn at random from what is left. Spot-check ids are
excluded. The text is the first 512 characters, i.e. exactly what the model scored.

Writes (gitignored, data/real_torrent_2026_08/):
    label_sample_scored.csv  id, subreddit, band, toxicity_score, is_toxic, is_toxic_050, body   (no usernames)
    label_sample_blind.csv   id, subreddit, body   — rows SHUFFLED (seeded) so order reveals nothing about score
    label_sample_design.json band populations and sample sizes, for inverse-probability weighting

Prints counts only; never prints a score of a sampled row.

    python scripts/local/torrent_2026_08/label_sample.py
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


def band_of(s: float) -> str:
    for lo, hi, name in C.LABEL_BANDS:
        if lo <= s < hi:
            return name
    raise ValueError(s)


def main() -> None:
    sc = pd.read_csv(C.SCORED, usecols=["id", "subreddit", "body", "toxicity_score", "is_toxic", "is_toxic_050"],
                     dtype={"id": str, "subreddit": str, "body": str}, keep_default_na=False)
    spot = set((C.DATA_DIR / "spotcheck_ids.txt").read_text(encoding="utf-8").split())
    # check_scores.py also printed the 5 highest-scored of a seeded random 5,000: exclude those too
    spot |= set(sc.sample(n=5000, random_state=C.SPOTCHECK_SEED + 2).nlargest(5, "toxicity_score")["id"])
    pool = sc[~sc["id"].isin(spot)].copy()
    pool["band"] = pool["toxicity_score"].map(band_of)
    rng = np.random.default_rng(C.LABEL_SEED)
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
    smp = pool.loc[picks].copy()
    smp["body"] = smp["body"].str.slice(0, C.CHAR_SLICE)
    assert len(smp) == C.LABEL_N and smp["id"].is_unique
    smp[["id", "subreddit", "band", "toxicity_score", "is_toxic", "is_toxic_050", "body"]].to_csv(
        C.DATA_DIR / "label_sample_scored.csv", index=False)
    blind = smp[["id", "subreddit", "body"]].sample(frac=1, random_state=C.LABEL_SEED).reset_index(drop=True)
    blind.to_csv(C.DATA_DIR / "label_sample_blind.csv", index=False)
    design = {"seed": C.LABEL_SEED, "n": C.LABEL_N, "bands": [list(b) for b in C.LABEL_BANDS],
              "population_per_band": pool["band"].value_counts().to_dict(),
              "sample_per_band": smp["band"].value_counts().to_dict(),
              "excluded_spotcheck_ids": len(spot), "subreddits_in_sample": int(smp["subreddit"].nunique())}
    (C.DATA_DIR / "label_sample_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
