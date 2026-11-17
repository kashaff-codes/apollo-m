"""
Every fixed parameter of the RC_2026-08 torrent run, in one place.

These were written down BEFORE any comment was scored and before any pipeline output existed. Nothing here may be changed
after looking at results; if a value turns out to be a poor choice, that is reported as a finding, not tuned away.

Namespaces: inputs live in data/real_torrent_2026_08/ (gitignored: usernames and comment text), outputs in
outputs/real_torrent_2026_08/ (small aggregate files only).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # repository root (apollo-m)
DATA_DIR = ROOT / "data" / "real_torrent_2026_08"
OUT_DIR = ROOT / "outputs" / "real_torrent_2026_08"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Source ───────────────────────────────────────────────────────────────────────────────────────────────────────────
# The .zst lives outside the repo; its location is machine-specific and is passed with --input where needed.
DUMP_SIZE_BYTES = 51_525_496_195
INFOHASH = "f4848163a5fb650e4b15eee0202858447bca120a"   # Academic Torrents "Reddit comments/submissions 2026-08"

# ── Step 2: filter ───────────────────────────────────────────────────────────────────────────────────────────────────
WINDOW_START, WINDOW_END = "2026-08-01", "2026-08-31"   # inclusive UTC days; the one and only window of this run
WINDOW_DAYS = (date(2026, 8, 31) - date(2026, 8, 1)).days + 1
FILTER_CAP = 1000              # max comments per subreddit per UTC day kept by the filter (reservoir sample)
FILTER_SEED = 42
FILTERED_CSV = DATA_DIR / "RC_2026-08_filtered.csv"
FILTER_SUMMARY = DATA_DIR / "RC_2026-08_filtered.summary.json"

# ── Step 3: scoring ──────────────────────────────────────────────────────────────────────────────────────────────────
MODEL = "unitary/toxic-bert"   # the repo's micro-layer model (main.Config.NLP_MODEL)
MAX_LEN = 256                  # tokens; main.Config.MAX_SEQ_LEN and score_real_reddit.py
CHAR_SLICE = 512               # characters passed to the model; same slice as score_real_reddit.py / MicroLayer
BATCH = 64                     # score_real_reddit.py batch size
BENCH_N, BENCH_SEED = 500, 42  # timing benchmark: 500 comments drawn with this seed
SCORING_TARGET_SECONDS = 2 * 3600
SCORING_SAMPLE_SEED = 42
#   Budget rule (fixed before scoring): measured throughput r (comments/s) gives a budget B = floor(r * 7200).
#   The per-subreddit-per-day quota N is the LARGEST integer N <= FILTER_CAP with  sum_cells min(kept_cell, N) <= B,
#   where the sum runs over every (subreddit, UTC day) cell of the filtered CSV. Each cell then keeps a seeded random
#   sample of min(kept_cell, N) comments. If N == FILTER_CAP everything filtered is scored.
SCORING_SAMPLE = DATA_DIR / "scoring_sample.csv"     # the downsampled comments (no scores)
SCORE_CACHE = DATA_DIR / "score_cache.csv"           # id -> toxicity_score, append-only, never overwritten
SCORED = DATA_DIR / "scored_comments.csv"            # sample joined with cache (a new file; inputs untouched)
BENCH_JSON = DATA_DIR / "scoring_benchmark.json"

# ── Thresholds ───────────────────────────────────────────────────────────────────────────────────────────────────────
TOXIC_THRESHOLD = 0.6          # main.Config.TOXICITY_THRESHOLD: "the configured threshold" (asserted at runtime)
TOXIC_THRESHOLD_ALT = 0.5      # score_real_reddit.py's is_toxic cut, used by the August real run; reported alongside

# ── Step 4: coverage rules ───────────────────────────────────────────────────────────────────────────────────────────
MIN_COMMENTS_PER_DAY = 50      # a day "counts" for a community only with >= 50 scored comments that day.
#   Why 50: the repo's reliability floor for a community mean is 30 comments (MesoLayer.MIN_COMMENTS_PER_SUB); a daily
#   mean needs at least that. Toxic-bert scores lie in [0,1], so the standard error of a daily mean is at most
#   0.5/sqrt(50) = 0.071 and, at a typical score spread, far less. Chosen before any score existed.
MIN_CONSECUTIVE_DAYS = 24      # forecast only communities with >= 24 consecutive counting days
#   The TFT is fed exactly that run of consecutive counting days (the latest run of >= 24), never zero-filled gaps:
#   models/tft_forecaster.py renumbers time_idx by row order, so a gap inside the series would be silently closed.
#   Communities that do not qualify still get meso scores; they get no forecast.

# ── Step 5: pipeline ─────────────────────────────────────────────────────────────────────────────────────────────────
HORIZON, LOOKBACK = 5, 14      # gen_forecasts.py / main.Config
TFT_MAX_EPOCHS = 6             # gen_forecasts.py
TFT_SEEDS = (42, 43, 44)       # TFT training is stochastic; every forecast is reported across these seeds
TFT_CANONICAL_SEED = 42        # the seed whose run is written to forecast_results.csv (same shape as outputs/real/)
#   Holdout check (fixed before running): for each qualifying community, hold out the last HORIZON days of its run,
#   train on the rest (same config, each seed), and compare the p50 MAE with (a) persistence = last training day's
#   value and (b) the mean of the last LOOKBACK training days. Also report the share of actuals inside p10..p90.
#   This measures forecast error of daily mean toxicity on real data. It is NOT an instability accuracy.
#
#   Finding about the EXISTING forecaster, made before any forecast was run on this corpus (library docs of
#   TimeSeriesDataSet(predict_mode) + a tiny CPU run on data/apollo_daily.csv: a series ending at time_idx 119 was
#   predicted for time_idx 115..119): TFTForecaster.train() drops the last HORIZON steps (cutoff = max time_idx - HORIZON)
#   and predict_quantiles() on observed data predicts the LAST HORIZON OBSERVED days from the LOOKBACK days before them.
#   gen_forecasts.py and save_forecast_csv() then label those values as the HORIZON days AFTER the last observation.
#   This run does not change that code. It calls it unchanged, reads the predicted time_idx from the dataset's
#   decoded_index, and labels each prediction with the date it actually predicts. The existing path's output is therefore
#   exactly the holdout check above (out of sample whenever the predicted steps are beyond the training cutoff, which is
#   recorded per community), and the comparison with persistence and the 14-day mean uses those same predictions.
#   No forward (September) forecast is produced: the existing forecaster cannot predict past the last observed row
#   without placeholder future rows, and adding those would change its logic.

# ── Step 6: labelling sample ─────────────────────────────────────────────────────────────────────────────────────────
LABEL_N = 300
LABEL_SEED = 2026
LABEL_BANDS = ((0.0, 0.1, "low"), (0.1, 0.6, "mid"), (0.6, 1.0000001, "high"))   # toxicity_score bands, 100 rows each
#   Within a band the 100 rows are spread evenly over subreddits (seeded), remainder drawn at random from the band.
#   The text shown to the labeller is the same first 512 characters the model scored. Spot-check ids are excluded.
#   Because bands are sampled equally, raw sample metrics over-represent high scores; inverse-probability weighted
#   precision/recall (weights = band population / band sample) are reported alongside.
LABEL_BOOTSTRAP = 2000
SPOTCHECK_SEED = 7

# ── Step 3 gate addition (fixed after the GPU driver faults at 18:29:43 and 18:48:32, before any check was run) ───────
#   The RTX 2060 raised two driver-level faults during scoring (nvlddmkm events 13/14/153; "illegal memory access",
#   "illegal instruction"). Chunks are only written after they finish, but faulty hardware could also return wrong
#   numbers without crashing. So, in addition to the 20-comment re-score: re-score on CPU a random CPU_RECHECK_N comments
#   from the whole cache, plus CPU_RECHECK_PER_CHUNK from each 20,000-row cache chunk written immediately before a
#   fault; every score must match within 1e-4.
CPU_RECHECK_N = 2000
CPU_RECHECK_PER_CHUNK = 500
CPU_RECHECK_SEED = 11

# ── Step 3: second scoring pass (rule fixed 19:27, after the join found ONE NaN score and before any comparison) ──────
#   The join refused because score_cache.csv holds one empty score: id p1ztawp (r/antiwork, an ordinary 104-character
#   sentence), cache row 322,668, in the 20,000-row chunk written just before the 18:48 fault. So the faulting GPU did
#   return a wrong value without crashing, and finite values may be wrong too. Rule:
#     pass 2  re-score EVERY comment again (same model and settings) into score_cache_pass2.csv;
#     compare pass 1 and pass 2 per comment; a comment AGREES if both are finite and |p1 - p2| <= SCORE_AGREE_TOL;
#     every comment that does not agree is scored on CPU (device=-1) into score_cache_cpu.csv;
#     final score = pass-1 value where the passes agree, else the CPU value.
#   score_cache.csv is never modified. The numbers of agreements and adjudications are reported.
SCORE_CACHE_PASS2 = DATA_DIR / "score_cache_pass2.csv"
SCORE_CACHE_CPU = DATA_DIR / "score_cache_cpu.csv"
SCORE_AGREE_TOL = 1e-4
