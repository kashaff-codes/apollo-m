# Improvement experiment — pre-registration

Written 2026-11-17, before the DEV sample was drawn, before any six-label score existed, and before any blend or
backtest number was computed. Nothing below is changed after results exist; anything added later is marked
EXPLORATORY in RESULTS.md. Additive only: no existing repo code or output is changed, no model in `models/` is
retrained, the full corpus is not re-scored, nothing is applied to the pipeline outputs.

Scripts: `scripts/local/torrent_2026_08/improve_*.py`. Small results: `outputs/real_torrent_2026_08/improvement/`.
Files containing comment text (DEV sample, labels) go to `data/real_torrent_2026_08/improvement/` (gitignored), so no
comment text can be committed.

## Known limits, stated before results

* **TEST is not pristine with respect to the variants.** The existing 300-row sample (now TEST) was labelled by the
  same labeller (Claude), and its V0 result and the finding that 52 of 78 false positives contain profanity were seen
  before V1–V3 were proposed. V3 (toxic − obscene) is directly motivated by that finding, so its TEST result may be
  optimistic. TEST has not been used to evaluate V1–V3 or any threshold other than V0's 0.5 and 0.6.
* **Single LLM labeller** for DEV and TEST, same guidelines (`data/real_torrent_2026_08/labelling_guidelines.md`), same
  method (blind file: id, subreddit, first 512 characters; batches of 50; scores not looked at until all labels are
  saved). Agreement is "agreement with LLM-assigned labels (single labeller, not human ground truth)".
* 300 + 300 labelled comments with roughly 30 toxic each: intervals will be wide.

## Part A — toxicity scorer variants

### Data

* **TEST** = the existing 300-row sample (`label_sample_scored.csv` / `label_sample_labels.csv`, seed 2026). Used
  exactly once, at the end, by `improve_eval.py test`, which refuses to run a second time.
* **DEV** = a new 300-row sample, seed **2027**, same design as TEST: bands on the stored `toxicity_score`
  (low < 0.1 ≤ mid < 0.6 ≤ high), 100 per band, spread evenly over subreddits, remainder at random within the band.
  Excluded from the DEV pool: the 300 TEST ids, the 15 spot-check ids, and `p1ztawp` (its text was printed during the
  scoring incident). Same-seed redraw must give the identical DEV sample (gate).
* Unclear labels are excluded from all metrics and counted.

### Scores

All 600 comments are scored again **on CPU** (`device=-1`; the GPU produced silent errors) with the repo's call
(`unitary/toxic-bert`, `top_k=None`, truncation 256 tokens, first 512 characters), keeping all six outputs: toxic,
severe_toxic, obscene, threat, insult, identity_hate.

| Variant | Score |
|---|---|
| V0 | P(toxic) — as the pipeline now |
| V1 | max(insult, threat, identity_hate, severe_toxic) |
| V2 | mean(insult, threat, identity_hate) |
| V3 | max(0, P(toxic) − P(obscene)) |

No other variant will be evaluated on TEST. Any further idea is EXPLORATORY and DEV-only.

### Metrics

* Weights: inverse sampling probability per band, w_band = (band population in the sample's pool) / (band rows in the
  sample). DEV pool = scored corpus minus the exclusions above; TEST weights as already recorded
  (`label_sample_design.json`). Every weighted figure is labelled "weighted".
* Threshold-free: weighted and unweighted **average precision** and **ROC-AUC** (scikit-learn, `sample_weight`).
* At a threshold (positive if score ≥ t): weighted and unweighted **precision, recall, F1**.
* **Primary metric: weighted F1 at the DEV-chosen threshold, on TEST.**

### Threshold choice (DEV only)

For each variant, candidate thresholds are the distinct DEV scores of that variant; t* = the candidate with the
highest **weighted F1 on DEV**; ties → the smallest such t. V0 is also reported at the pipeline's fixed 0.6 (and 0.5).

### Intervals

Stratified bootstrap (resample within each band, so the weights stay valid), 2,000 resamples, seed 2028, percentile
95% intervals — for each variant's TEST metrics and for the paired difference (variant − V0) on the same resamples.
V0 here is V0 at its own DEV-chosen threshold (same procedure for every variant).

### Decision rule

A variant counts as an improvement over V0 only if BOTH hold on TEST:
1. the paired 95% interval of (weighted F1 variant − weighted F1 V0) lies entirely above 0, and
2. the point difference in weighted F1 exceeds the full width of V0's own 95% interval for weighted F1
   (the literal reading of "beats V0 by more than the interval width").
If no variant meets both, RESULTS.md says so and the search stops. Weighted AP differences are reported with their
paired intervals as secondary evidence and do not change the verdict.

### Cost estimate

Six-label scores cost one forward pass, the same as the current scoring, but only P(toxic) was stored, so applying a
variant to the full corpus means re-scoring 1,087,806 comments. The estimate uses the measured GPU rate (313.91/s,
`scoring_benchmark.json`) times two passes (the two-pass check the GPU needed), and the CPU rate measured while
scoring the 600.

## Part B — forecast (existing TFT outputs; baselines; TFT re-training only as stated)

### B1 — held-out 5 days (2026-08-27..31), from `forecast_seeds.csv`, no new training

* Per seed (42, 43, 44) and per community: MAE over the 5 days of TFT p50, persistence (value on 2026-08-26) and the
  14-day mean (2026-08-13..26). Count of communities where TFT MAE < persistence MAE, and < 14-day-mean MAE.
* Blends b_w = w·p50 + (1 − w)·mean14, **w ∈ {0, 0.25, 0.5, 0.75, 1}** (w = 0 is the 14-day mean, w = 1 the TFT).
  MAE over all 190 community-days, per seed. Evaluated once.
* Interval: paired bootstrap over the 38 communities (resample communities), 2,000 resamples, seed 2029, for
  MAE(b_w) − MAE(mean14).
* Rule: a blend "beats the 14-day mean" only if the upper end of that 95% interval is below 0 **for all three seeds**.
  Otherwise RESULTS.md states plainly that no blend beats the 14-day mean.

### B2 — rolling-origin backtest

* Origins (last observed day) **o ∈ {2026-08-14, 17, 20, 23, 26}**; forecast days o+1 … o+5; all 38 qualifying
  communities, daily series from `daily_series.csv` (the forecast run, no gaps).
* Persistence = value on o; 14-day mean = mean of o−13 … o. MAE per origin, per community, and pooled over origins.
  (Windows of origins 3 days apart overlap by 2 days; stated, not hidden.)
* TFT per origin: the existing `TFTForecaster`, unchanged, same configuration as `forecast.py` (6 epochs, lookback 14,
  horizon 5), seeds 42/43/44, **CPU**, fed each series truncated at o+5 so that it trains on days ≤ o and predicts
  o+1 … o+5. The existing code needs 19 training days for one training window, so the TFT can only run for origins
  with o ≥ 2026-08-19: **{20, 23, 26}**. It runs only if the total CPU time stays under 30 minutes (timed; if the first
  origin's three seeds take more than 10 minutes, the TFT part stops and is reported as not run).
* Gate: the origin-26 re-training must reproduce the existing `forecast_seeds.csv` p50 exactly (same data, same seed,
  CPU) — this checks the harness.
* TFT vs baselines is compared on the common origins {20, 23, 26}.

## Quality gates

* 600 comments scored = 300 DEV + 300 TEST, no id overlap; CPU P(toxic) matches the stored `toxicity_score` within
  1e-4 for all 600.
* No degenerate score column (six outputs and four variants: finite, varying, not all zero).
* 10 DEV rows spot-checked: blind-file text equals the first 512 characters of the comment in the filtered CSV
  (itself verified against the dump), printed for reading.
* Same seed (2027) gives the same DEV sample (two independent draws, identical ids and order).
