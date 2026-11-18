"""
Full APOLLO-M pipeline on the rebuilt corpus, reusing existing scores + embeddings cache.
Same steps as APOLLOPipeline.run_full_pipeline, but:
  * only rows that already have a toxicity_score are used, so MICRO reuses the cached
    scores instead of re-running toxic-bert on 257k comments (hours);
  * input is data/apollo_comments.csv = the rebuilt corpus (build_corpus.py ->
    apollo_comments_real.csv) with its toxicity scores merged in. The unscored _real file
    on its own has no scores and would leave zero rows;
  * existing outputs are copied to outputs_backup_before_full/ first
    (copy only, never overwrites a backup, never deletes);
  * data/apollo_daily.csv and data/ground_truth.json are never written, and the
    script checks them against git at the end.

  OMP_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES python run_full_from_cache.py
"""
import os, sys, shutil, subprocess, logging
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("APOLLO-M")
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── 0. back up current outputs (copy only) ────────────────────────────────
bk = ROOT / "outputs_backup_before_full"
bk.mkdir(parents=True, exist_ok=True)
n = 0
for f in (ROOT / "outputs").iterdir():
    if f.is_file() and not (bk / f.name).exists():
        shutil.copy2(f, bk / f.name); n += 1
log.info("Backed up %d output files -> %s", n, bk)

import pandas as pd
from main import Config, APOLLOPipeline
cfg = Config()
pipe = APOLLOPipeline(cfg)

log.info("Starting full pipeline (cached scores + embeddings)")
if not cfg.REDDIT_COMMENTS.exists():
    # The loader would silently fall back to SYNTHETIC comments. Never do that here.
    raise SystemExit(f"{cfg.REDDIT_COMMENTS} not found - refusing to continue (loader would use synthetic data).")
log.info("Comments file: %s", cfg.REDDIT_COMMENTS)
comments_df = pipe.loader.load_reddit_comments()      # cleans + converts created_utc to dates
graph = pipe.loader.load_hyperlink_graph()
comments_df = pipe.run_eda(comments_df)
before = len(comments_df)
comments_df = comments_df[comments_df["toxicity_score"].notna()].reset_index(drop=True)
log.info("Using %d scored comments (%d unscored rows left out)", len(comments_df), before - len(comments_df))
if comments_df.empty:
    raise SystemExit("No scored comments in the input file - score the corpus first (MICRO).")
comments_df = pipe.run_micro(comments_df)             # cached path: no toxic-bert inference
community_toxicity = pipe.micro.compute_community_toxicity(comments_df)
chm_results = pipe.run_meso(comments_df, graph)
features_df = pipe.run_unsupervised(chm_results)
pipe.run_cybersecurity()
macro_result = pipe.run_macro(comments_df, epochs=15)  # loads embeddings cache, TFT, multi-subreddit forecast
alerts = [pipe.alerts.evaluate(c) for c in chm_results]
pipe.alerts.save_alerts(alerts)
features_df.to_csv(cfg.OUTPUT_DIR / "community_health_report.csv", index=False)
community_toxicity.to_csv(cfg.OUTPUT_DIR / "toxicity_by_subreddit_daily.csv", index=False)

# ── benchmark guard ───────────────────────────────────────────────────────
# Compare with Kashaf's version on origin/main (local HEAD may hold older commits of ours).
_ref = "origin/main" if subprocess.run(["git", "rev-parse", "--verify", "-q", "origin/main"], cwd=ROOT, capture_output=True).returncode == 0 else "HEAD"
r = subprocess.run(["git", "diff", "--quiet", _ref, "--", "data/apollo_daily.csv", "data/ground_truth.json"], cwd=ROOT)
log.info("Benchmark files vs git: %s", "UNTOUCHED" if r.returncode == 0 else "DIFFERENT - STOP AND CHECK")

# ── quick summary ─────────────────────────────────────────────────────────
m = pd.read_csv(cfg.OUTPUT_DIR / "meso_report.csv")
for c in ["community_health_index", "toxicity_trend", "instability_score", "polarization", "echo_chamber_index", "churn_rate"]:
    log.info("  %-24s min=%.4f mean=%.4f max=%.4f std=%.4f", c, m[c].min(), m[c].mean(), m[c].max(), m[c].std())
fc = pd.read_csv(cfg.OUTPUT_DIR / "forecast_results.csv")
log.info("forecast: %d subreddits", fc["subreddit"].nunique())
log.info("Done. Next: python -m pytest tests/ -v")
