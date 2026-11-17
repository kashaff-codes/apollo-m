# Improvement experiment — results

Pre-registration: `PREREGISTRATION.md` (SHA-256 `4137640aeb9297432cdb2df5364d067c1722e76f7cdfe060b5ec15677d8a9553`
when written at 20:42 and again after all results — unchanged). Everything here is from code run on the real data;
nothing was applied to the pipeline outputs, no existing repo file or output was changed, no model in `models/` was
retrained, the full corpus was not re-scored. Nothing EXPLORATORY was run.

**Verdict in one line each:** no toxicity variant meets the pre-registered improvement rule (V1 comes closest: better
on the paired interval, but by less than V0's own interval width); no forecast blend beats the 14-day mean, and the
14-day mean is the best of all tested forecasters across five origins.

## What was run

| Step | Script | Gate |
|---|---|---|
| DEV sample, seed 2027, 300 rows (100 per band, 39 subreddits), 316 ids excluded (300 TEST + 15 seen + `p1ztawp`) | `improve_dev_sample.py` | 4/4 PASS; same-seed redraw identical (in-process, and a second process gave the same blind-file SHA-256 `b52e9f13…`) |
| DEV labelled blind, 6 batches of 50, same guidelines and method as TEST, **single LLM labeller (Claude), not human ground truth** | — | 300 labels, ids and order identical to the blind file; labels SHA-256 `e5eb0de8…` saved before any six-label score existed |
| 600 comments (300 DEV + 300 TEST) scored on CPU, six outputs, V0–V3 | `improve_score6.py` | 13/13 PASS: 600 = 300 + 300, no overlap; CPU P(toxic) = stored score within 1.09e-06; no degenerate column; DEV text = filtered CSV for all 300, 10 printed and read |
| Thresholds chosen on DEV only | `improve_eval.py dev` | `dev_thresholds.json` (SHA-256 `2320b7f9…` before TEST) |
| TEST evaluated once | `improve_eval.py test` | `test_result.json`; a second run is refused ("TEST has already been used once") |
| Forecast B1 + B2 | `improve_forecast.py` | 11/11 PASS (see note) |

Note on B: the first run of `improve_forecast.py` reported one FAIL — a bug in my check (TFT rows carry the `r/`
prefix, baseline rows do not, so the comparison merged zero rows). The check was fixed to strip the prefix and to
require all 114 rows; the re-run is deterministic (same seeds, same data) and passed 11/11. No result was chosen
between the two runs.

## A — toxicity scorer variants

Labels: DEV 245 not_toxic / 41 toxic / 14 unclear; TEST 260 / 31 / 9. Unclear excluded (DEV 286 used, TEST 291).
Weights = band population / 100 per band. Bootstrap: within-band, 2,000, seed 2028, percentile 95%.

### DEV (threshold choice; weighted F1 maximised)

| Variant | DEV threshold | weighted F1 | weighted P | weighted R | weighted AP | weighted ROC-AUC | unweighted F1 | unweighted AP |
|---|---|---|---|---|---|---|---|---|
| V0 P(toxic) | 0.7168 | 0.430 | 0.321 | 0.649 | 0.395 | 0.946 | 0.432 | 0.398 |
| V1 max(insult, threat, identity_hate, severe_toxic) | 0.3831 | 0.542 | 0.584 | 0.506 | 0.554 | 0.965 | 0.545 | 0.557 |
| V2 mean(insult, threat, identity_hate) | 0.1661 | 0.551 | 0.645 | 0.481 | 0.538 | 0.965 | 0.556 | 0.541 |
| V3 max(0, toxic − obscene) | 0.1349 | 0.349 | 0.240 | 0.638 | 0.232 | 0.870 | 0.349 | 0.251 |
| V0 at pipeline 0.6 (reference) | 0.6 | 0.390 | 0.278 | 0.649 | — | — | 0.391 | — |

### TEST (used once)

| Variant (DEV threshold) | weighted F1 [95%] | weighted P [95%] | weighted R [95%] | weighted AP [95%] | weighted ROC-AUC [95%] | unweighted P / R / F1 | unweighted AP / ROC-AUC | flagged |
|---|---|---|---|---|---|---|---|---|
| V0 (0.7168) | 0.325 [0.216, 0.441] | 0.253 [0.165, 0.346] | 0.453 [0.263, 0.793] | 0.167 [0.119, 0.284] | 0.790 [0.673, 0.952] | 0.253 / 0.677 / 0.368 | 0.210 / 0.759 | 83 |
| V1 (0.3831) | 0.437 [0.281, 0.622] | 0.500 [0.333, 0.667] | 0.389 [0.219, 0.685] | 0.297 [0.191, 0.539] | 0.783 [0.634, 0.978] | 0.500 / 0.581 / 0.537 | 0.408 / 0.850 | 36 |
| V2 (0.1661) | 0.362 [0.204, 0.541] | 0.452 [0.273, 0.625] | 0.302 [0.150, 0.552] | 0.286 [0.189, 0.516] | 0.802 [0.677, 0.977] | 0.452 / 0.452 / 0.452 | 0.388 / 0.852 | 31 |
| V3 (0.1349) | 0.296 [0.206, 0.395] | 0.203 [0.138, 0.274] | 0.547 [0.324, 0.929] | 0.243 [0.156, 0.434] | 0.794 [0.680, 0.961] | 0.205 / 0.806 / 0.327 | 0.327 / 0.797 | 122 |
| V0 at pipeline 0.6 (reference) | 0.289 | 0.212 | 0.453 | — | — | 0.212 / 0.677 / 0.323 | — | 99 |

V0 at 0.6 reproduces the original TEST result (`label_agreement.json`: weighted 0.212 / 0.453 / 0.289), so the
evaluation path is consistent with the earlier one.

### Decision (pre-registered rule: paired 95% interval of the weighted-F1 difference above 0 AND difference larger than the width of V0's own weighted-F1 interval, 0.225)

| Variant | Δ weighted F1 vs V0 [paired 95%] | rule 1 | rule 2 (Δ > 0.225) | Δ weighted AP [paired 95%] (secondary) | improvement? |
|---|---|---|---|---|---|
| V1 | +0.113 [+0.031, +0.222] | yes | no | +0.131 [+0.056, +0.276] | **no** |
| V2 | +0.037 [−0.065, +0.157] | no | no | +0.119 [+0.054, +0.251] | **no** |
| V3 | −0.029 [−0.126, +0.059] | no | no | +0.077 [−0.015, +0.212] | **no** |

**No variant beats V0 on TEST by more than the interval width.** The search stops here, as pre-registered.

What the numbers do show, stated as what it is: V1 and V2 consistently halve the false alarms on both DEV and TEST
(weighted precision 0.25 → 0.50 for V1 on TEST, 36 flagged instead of 83) at the cost of some recall, and their
average precision is higher with paired intervals above 0 — the label set that targets insults/threats/identity hate
ranks targeted abuse better than P(toxic). V3, the variant motivated by the profanity finding, does not help: removing
P(obscene) also removes most of the signal (weighted AP on DEV 0.232 vs 0.395). Limits: one LLM labeller for both sets;
TEST had been seen before the variants were proposed (pre-registration, "Known limits"); about 30–40 toxic comments per
set, hence the wide intervals.

### Cost of applying the best variant (V1) to the full corpus

Only P(toxic) was stored, so any variant needs one re-score of all 1,087,806 comments (all six outputs come from the
same forward pass; computing V1 from them is negligible).

| Route | Rate (measured) | Time |
|---|---|---|
| GPU, one pass | 313.91 comments/s (`scoring_benchmark.json`) | 0.96 h (the actual verification pass took 0.83 h) |
| GPU, two passes + CPU adjudication (the check this GPU needs) | as above | ≈ 1.9 h + minutes |
| CPU only, one pass | 26.4 comments/s, 8 threads (`data/…/improvement/cpu_rate.json`) | ≈ 11.4 h |

Downstream (daily series, meso, forecast, checks) re-runs in minutes. Not applied — the evidence does not meet the bar.

## B — forecast

### B1 — held-out days 2026-08-27..31 (existing `forecast_seeds.csv`, no training)

MAE over 190 community-days (38 communities × 5 days):

| seed | TFT | persistence | 14-day mean | TFT wins vs persistence | TFT wins vs 14-day mean |
|---|---|---|---|---|---|
| 42 | 0.02008 | 0.01633 | 0.01136 | 18 / 38 | 12 / 38 |
| 43 | 0.02107 | 0.01633 | 0.01136 | 18 / 38 | 8 / 38 |
| 44 | 0.01854 | 0.01633 | 0.01136 | 19 / 38 | 11 / 38 |

Blends w·TFT + (1 − w)·mean14 (w fixed in advance), MAE and paired bootstrap 95% interval of MAE(blend) − MAE(mean14):

| w | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| 0 (= 14-day mean) | 0.01136 | 0.01136 | 0.01136 |
| 0.25 | 0.01201 (+0.00065 [−0.00038, +0.00189]) | 0.01212 (+0.00076 [−0.00025, +0.00206]) | 0.01194 (+0.00058 [−0.00032, +0.00187]) |
| 0.5 | 0.01413 (+0.00277 [+0.00044, +0.00543]) | 0.01447 (+0.00311 [+0.00089, +0.00583]) | 0.01367 (+0.00231 [+0.00026, +0.00529]) |
| 0.75 | 0.01694 (+0.00558 [+0.00199, +0.00986]) | 0.01755 (+0.00619 [+0.00271, +0.01049]) | 0.01599 (+0.00462 [+0.00136, +0.00916]) |
| 1 (= TFT) | 0.02008 (+0.00872 [+0.00379, +0.01473]) | 0.02107 (+0.00971 [+0.00491, +0.01556]) | 0.01854 (+0.00718 [+0.00257, +0.01335]) |

**No blend beats the 14-day mean.** Every blend with any TFT weight has a higher MAE for every seed; from w = 0.5 up the
difference is significantly worse.

### B2 — rolling-origin backtest

Baselines, 38 communities, 5 days after each origin (windows 3 days apart overlap by 2 days):

| origin | persistence MAE | 14-day mean MAE | communities where 14-day mean is better |
|---|---|---|---|
| 2026-08-14 | 0.016544 | 0.012202 | 27 / 38 |
| 2026-08-17 | 0.013109 | 0.011277 | 22 / 38 |
| 2026-08-20 | 0.012730 | 0.011445 | 19 / 38 |
| 2026-08-23 | 0.014676 | 0.011636 | 23 / 38 |
| 2026-08-26 | 0.016326 | 0.011361 | 28 / 38 |
| pooled | 0.014677 | 0.011584 | 34 / 38 (community MAE averaged over origins) |

The 14-day mean is better than persistence at every origin, so the earlier single-split ranking was not a fluke of one
split.

TFT re-trained per origin (existing `TFTForecaster` unchanged, `forecast.py` settings, seeds 42/43/44, **CPU**,
96.4 s in total: 27.8 / 32.1 / 36.5 s per origin; the origin-26 run reproduces `forecast_seeds.csv` to 9.7e-17). Only
origins with ≥ 19 training days are possible with the existing code:

| origin | TFT MAE seed 42 / 43 / 44 | persistence | 14-day mean | TFT wins vs 14-day mean (42 / 43 / 44) |
|---|---|---|---|---|
| 2026-08-20 | 0.0309 / 0.0327 / 0.0346 | 0.0127 | 0.0114 | 7 / 6 / 5 of 38 |
| 2026-08-23 | 0.0194 / 0.0273 / 0.0210 | 0.0147 | 0.0116 | 11 / 5 / 12 of 38 |
| 2026-08-26 | 0.0201 / 0.0211 / 0.0185 | 0.0163 | 0.0114 | 12 / 8 / 11 of 38 |
| pooled (3 origins) | 0.0235 / 0.0270 / 0.0247 (mean 0.0251) | 0.0146 | 0.0115 | — |

The TFT is worst with the least training data (origin 20: 20 training days, 2 windows per community) and never beats
either baseline on average.

## Recommendation

* **Toxicity:** keep V0 in the pipeline for now — the pre-registered bar was not met. V1 is the candidate worth a proper
  test: pre-register V1 at its DEV threshold (0.383) against V0, label a fresh, larger sample (≥ 1,000 comments) by
  people rather than one LLM, and only then decide. If adopted, it costs one re-score of the corpus (≈ 1.9 h on this
  GPU with the two-pass check, or ≈ 11.4 h on CPU).
* **Forecast:** on this real month the 14-day mean is the best predictor tested (5 origins), the TFT is worse than both
  baselines at every origin it could run, and no blend helps. Any forecast shown from this corpus should be shown next to
  the 14-day mean and not described as skilful. Improving the TFT would need more history (more months) or a changed
  configuration — both outside this additive, no-tuning experiment.

## Files

`outputs/real_torrent_2026_08/improvement/`: `PREREGISTRATION.md`, `RESULTS.md`, `dev_thresholds.json`,
`test_result.json`, `toxicity_variants_test.csv`, `forecast_improvement.json`, `forecast_b1_per_community.csv`,
`forecast_b2_baselines.csv`, `forecast_b2_tft.csv` (no comment text, no usernames).
`data/real_torrent_2026_08/improvement/` (gitignored, contains comment text): `dev_sample_blind.csv`,
`dev_sample_scored.csv`, `dev_sample_labels.csv`, `dev_sample_design.json`, `six_label_scores.csv`, `cpu_rate.json`,
`improve_forecast.log`, `tft_models/`.
