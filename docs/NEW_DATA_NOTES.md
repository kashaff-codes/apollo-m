# New data notes — APOLLO-M on the Academic Torrents dump RC_2026-08

Branch `feat/torrent-2026-08` (from `origin/main` at `cd481b4`). Namespace: inputs `data/real_torrent_2026_08/`
(gitignored), outputs `outputs/real_torrent_2026_08/`. Nothing from the API-collected `real_v2` corpus is used.

---

## Part A — expectations, fixed BEFORE any comment was scored

Written after reading `docs/APOLLO-M_FYP_Mid_Evaluation_Report.docx` (§3.3 Modules, §3.5 Datasets, §3.6 Evaluation),
`docs/APOLLO_Technical_Report.md` (§5.3–5.6, §7.1, §12) and `docs/REPORT_UPDATES.md` (§3.6 "Results on real data"), and
after looking at the August committed outputs in `outputs/real/`. At the time of writing, the filter had run but no
toxicity score, daily series or pipeline output existed for this corpus. The data (39 subreddits from a monthly dump,
capped at 1,000 comments per subreddit per day) and the month differ from August (24 subreddits, API collection,
2026-07-02..2026-08-11), so the comparison judges direction and structure, not exact numbers.

### Where the report's numbers come from (checked before results)

* "Real-data spread 0.093" appears in `docs/REPORT_UPDATES.md` (§3.6). The committed August summary
  `outputs/real_reddit_summary.json` says `between_community_spread: 0.1643` (max − min of per-community mean toxicity,
  24 communities, `score_real_reddit.py`). The two do not agree with each other; 0.093 was presumably written at an
  earlier collection stage. Both are recorded; neither is used as a pass mark.
* "0.016 on the corrected benchmark" is in `docs/APOLLO_Technical_Report.md` §7.1 (0.543–0.559).
* "Political and general-news communities score highest and r/science lowest" is in `docs/REPORT_UPDATES.md` §3.6.
  In the August data r/science was actually 3rd lowest (philosophy 0.0365 and history 0.0296 were lower); neither of
  those is among this run's 39 subreddits.
* "Toxicity rate" in the meso layer (`MesoLayer.analyze_all_communities`) is the community's MEAN toxicity score, not a
  thresholded rate. The CHI formula (report eq. 3.1, `main.py` line 445) is CHI = 100 − (35T + 30P + 20C + 15E), clamped
  to 0..100, with T that mean score.

### Expectations and the test that decides each one

| # | Expectation (source) | Test fixed now |
|---|---|---|
| E1 | Communities differ measurably in toxicity (REPORT_UPDATES §3.6: 0.093 real vs 0.016 benchmark) | spread = max − min of per-community mean `toxicity_score` over communities passing the meso filter; **match** if spread ≥ 0.048 (3× the benchmark's 0.016) |
| E2a | Political and general-news communities score highest (REPORT_UPDATES §3.6) | rank the meso communities by mean toxicity; **match** if the median rank of {politics, news, worldnews, Conservative} is in the top half |
| E2b | r/science lowest (REPORT_UPDATES §3.6) | **match** if r/science has the lowest mean toxicity of all meso communities; otherwise report its rank |
| E3 | CHI bounded 0–100 and varies (report §3.3.2, eq. 3.1) | min ≥ 0, max ≤ 100, > 1 distinct value |
| E4 | Instability score and its trend vary (report §3.3.2, eq. 3.2) | `instability_score` not all zero and > 1 distinct; `toxicity_trend` > 1 distinct |
| E5 | Polarisation and echo differ per community (report §3.3.2) | > 1 distinct value of each among communities found in the SNAP graph; count communities absent from the graph (they get the global fallback) |
| E6 | Forecast intervals widen with horizon (report §3.3.4, tech report §12) | for each TFT seed, mean (p90 − p10) on day 5 > day 1 over all forecast communities; also report the share of communities whose width is non-decreasing |
| E7 | Recommender gives a mix of actions (August output: 1 EMERGENCY, 1 INCREASE, 22 WARN) | ≥ 2 distinct `recommended_action` values |
| E8 | Clustering separates communities (report §3.3.3) | 5 K-Means clusters all non-empty; both `pca_x` and `pca_y` vary |
| E9 | Forecasts are real values, not empty (report §3.3.4) | every qualifying community has 5 forecast days per seed; p50 > 0, no NaN |
| E10 | (my addition, not the report's) the same communities rank similarly in August's API corpus and this dump | Spearman ρ of per-community mean toxicity over the 14 subreddits present in both (`outputs/real_reddit_summary.json` vs this run); **match** if ρ > 0 |

### My own priors (plausibility only, not from the report, not pass/fail)

* The 39-subreddit list is not the August list. It includes r/RoastMe (insults are the format) and r/gonewild (sexual
  vocabulary, which toxic-bert's "toxic" label is known to respond to). Either may outrank the political subreddits; if
  so, that says something about the community set and the model, not about the pipeline being broken.
* r/aww and r/science should be among the least toxic.
* A day-1 forecast should sit inside the range of the community's last 14 observed daily means.
* Real-world instability accuracy is not measurable here (no labels of real destabilisation exist) and will not be
  reported.

---

## Part B — what ran, on what

All commands from the repo root with `venv\Scripts\python.exe`; `S` = `scripts\local\torrent_2026_08`. Every step ran
on the **full** filtered corpus (no subsample): the scoring budget covered every filtered comment (quota N = 1,000 = the
filter cap).

| Step | Ran on | Result |
|---|---|---|
| Filter test (`filter_arctic_dump.py --limit-lines 2000000`, twice) + `S\verify_filter.py` | first 2,000,000 lines | 8/8 PASS; identical SHA-256 across runs |
| Full filter | all 367,543,312 lines | 1,087,806 kept; gate 12/12 PASS (`S\check_filter_output.py`) |
| `S\verify_kept_rows.py` | all lines again | every kept row identical to the dump; 3/3 PASS |
| Scoring (`S\scoring.py`), two GPU passes + CPU adjudication | 1,087,806 comments | gate 17/17 PASS (`S\check_scores.py`) |
| Daily series + coverage (`S\build_daily.py`) | 1,209 subreddit-days | 13/13 PASS; 38 of 39 communities qualify |
| Meso / GNN / recommender / clusters / alerts (`S\run_pipeline.py`) | 39 communities | 28/28 PASS |
| TFT (`S\forecast.py`, CPU, seeds 42/43/44) | 38 communities × 31 days | 17/17 PASS |
| Independent recomputation + expectations (`S\verify_outputs.py`) | all outputs | 15/15 PASS; 10 of 11 expectations match |
| Investigations (`S\investigate.py`) | all outputs | `investigations.json` |
| Label sample (`S\label_sample.py`), labels, agreement (`S\label_agreement.py`) | 300 comments | see Part C |

Not run: no database load, no dashboard/API/monitoring change, nothing from `real_v2` or Aaqib's branches, no forward
(September) forecast (Part D1), no test added to `tests/`.

An INTERIM dry run of the downstream scripts (12 subreddits already scored, CPU, scratch folder outside the repo) was
used only to catch code errors before scoring finished; it found one (a crash in `verify_outputs.py` when a community
is absent) and none of its numbers are reported anywhere.

## Part C — results, compared with the expectations of Part A

Real-data results only. None of the benchmark (planted ground truth) numbers of the report are restated or compared
here as if they were the same kind of claim.

| # | Expected (fixed before results) | Measured | Verdict |
|---|---|---|---|
| E1 | spread ≥ 0.048 | spread **0.1959** over 39 communities (max r/gonewild 0.2408, min r/geopolitics 0.0449). August: 0.1643 (24 communities); REPORT_UPDATES: 0.093; benchmark 0.016 | match |
| E2a | political/news median rank in top half | median rank **14 of 39** (politics 6, news 11, worldnews 17, Conservative 22) | match |
| E2b | r/science lowest | r/science **38 of 39** (0.0452); lowest r/geopolitics (0.0449) | differs — see I1 |
| E3 | CHI in 0–100 and varies | **65.45 – 83.14**, 38 distinct values | match |
| E4 | instability and trend vary, not all zero | instability **0.0186 – 0.1118**, all 39 > 0; trend **−0.0140 … +0.0170** (25 rising, 14 falling) | match |
| E5 | polarisation and echo per community | P 38 distinct, E 37 distinct among the 38 communities in the SNAP graph; r/Superstonk absent (global fallback) | match |
| E6 | intervals widen with horizon (each seed) | mean p90−p10 day 1 → day 5: seed 42 **0.0707 → 0.0781**, 43 **0.0811 → 0.0833**, 44 **0.0641 → 0.0674** | match |
| E7 | mix of actions | WARN 27, MONITOR 7, INCREASE_MODERATION 5 | match |
| E8 | 5 non-empty clusters, both PCA axes vary | sizes {0: 8, 1: 19, 2: 5, 3: 3, 4: 4}; std pca_x 1.562, pca_y 1.196; silhouette 0.3432 | match |
| E9 | forecasts real | 38 communities × 5 days × 3 seeds, p50 0.075–0.169, no NaN | match |
| E10 | (mine) same ranking as August | Spearman **ρ = 0.916** over the 14 shared subreddits | match |

Spot check (`S\check_scores.py`, 10 random comments, seed 7): all ten are real comment text in a plausible subreddit
(and `verify_kept_rows.py` confirms every kept row's subreddit against the dump). Neutral text scores low (0.0007–
0.0095); a jab at "crazy liberals" in r/Conservative scores 0.18. Not everything looks sensible: a mild roast in
r/RoastMe ("30 going on rebellious 15.") scores 0.0008, i.e. a non-profane insult is scored as harmless, and the five
highest of a random 5,000 (all ≥ 0.998) are heavy profanity, including a supportive reply in r/depression. That
pattern is what the labelled sample below measures.

Plausibility (stated as plausibility, not accuracy): the three most toxic communities are r/gonewild, r/RoastMe and
r/PublicFreakout — the first two as the Part A prior anticipated (sexual vocabulary; insults as the format). The least
toxic are r/geopolitics, r/science, r/China, r/aww. The 14 communities shared with August keep almost the same order
(ρ 0.916) although the source, month and sampling all differ.

### Investigations (`S\investigate.py` → `investigations.json`)

* **I1 — E2b differs because of a statistical tie with a community not in the August set.** r/science 0.04519 vs
  r/geopolitics 0.04489: difference 0.0003, comment-level 95% interval [−0.00256, 0.00317], day-level bootstrap 95%
  interval [−0.00491, 0.00558]; r/science is lower in 46.3% of bootstrap resamples. In August r/science was not the
  lowest either (3rd lowest; r/philosophy and r/history were lower, and neither is in this run). Nothing was adjusted.
* **I2 — DBSCAN marks 34 of 39 communities as outliers** (August: 24 of 24). With `eps = 0.5` and `min_samples = 5`
  on the standardised 4 features, the median distance to the 4th neighbour is 1.043 and only 1 community has 4
  neighbours within eps. The outlier flag carries almost no information at this scale; this is the existing
  configuration, not changed.
* **I3 — GNN raw outputs are near zero everywhere**: 1e-06 … 1.1e-05 here (9 distinct values at 6 dp), 1e-06 … 3e-06
  in August. `gnn_risk` is a percentile rank of these numbers.
* **I4 — churn is partly a sampling effect of the cap.** Churn vs pre-cap volume: Spearman 0.377; median churn 0.7754
  for the 28 communities capped on ≥ 25 days vs 0.6911 for the other 11. Churn enters CHI with weight 20.
* **I5 — the TFT loses to simple baselines on real data** (all 190 predicted community-days out of sample; CPU;
  `forecast_eval.json`):

  | seed | TFT p50 MAE | persistence MAE | 14-day-mean MAE | beats persistence | beats 14-day mean | p10–p90 coverage |
  |---|---|---|---|---|---|---|
  | 42 | 0.0201 | 0.0163 | 0.0114 | 18 / 38 | 12 / 38 | 0.837 |
  | 43 | 0.0211 | 0.0163 | 0.0114 | 18 / 38 | 8 / 38 | 0.900 |
  | 44 | 0.0185 | 0.0163 | 0.0114 | 19 / 38 | 11 / 38 | 0.805 |

  Across seeds the p50 of the same community-day varies by 0.0113 on average (max 0.0330). Likely causes, measured:
  the model pulls every community toward the cross-community mean (slope of error on community level −0.408 / −0.375
  / −0.416; spread of community-mean p50 0.029–0.034 vs 0.044 observed); it barely tracks direction (day-1→day-5
  direction agreement 0.474 / 0.474 / 0.447); and the series are mostly day-to-day noise around a stable level
  (median within-community daily SD 0.0133, about the persistence error), which a 14-day mean averages away. Training
  is 6 epochs on about 8 windows per community (`gen_forecasts.py` settings). Not tuned.

### Agreement with LLM-assigned labels (single labeller, not human ground truth)

300 comments (100 per score band, 39 subreddits), labelled blind in 6 batches of 50 under `labelling_guidelines.md`
(written first; toxic = insult, harassment, hate, threat or abusive profanity aimed at a person or group). Labels:
260 not_toxic, 31 toxic, 9 unclear (excluded). `label_agreement.json`, 2,000 bootstrap resamples:

| toxic-bert decision | precision | recall | F1 | confusion (label toxic: TP/FN; label not toxic: FP/TN) | weighted to corpus P / R / F1 |
|---|---|---|---|---|---|
| score ≥ 0.6 | **0.212** [0.132, 0.295] | **0.677** [0.500, 0.842] | **0.323** [0.210, 0.428] | 21 / 10; 78 / 182 | 0.212 / 0.453 / 0.289 |
| score ≥ 0.5 | 0.218 [0.144, 0.297] | 0.774 [0.609, 0.909] | 0.340 [0.236, 0.437] | 24 / 7; 86 / 174 | 0.218 / 0.521 / 0.308 |

This is a poor result for the model under this definition. Likely cause (word list fixed before looking at which rows
disagree): 52 of the 78 false positives contain profanity, against 18 of 182 true negatives. toxic-bert's "toxic"
label responds to swearing whether or not it targets anyone, while these guidelines count only targeted abuse. Part of
the gap is therefore a definitional mismatch (Jigsaw's "toxic" includes rude or profane language), part model error.
Every community mean in Part C inherits this: the toxicity measured is closer to "profane or hostile language" than to
"abuse of people".

## Part D — findings about existing code (reported, not changed)

* **D1 — the forecast labels predictions of observed days as future days.** `TFTForecaster.train()` trains on
  time_idx ≤ max − 5 and `predict_quantiles()` (via `TimeSeriesDataSet.from_dataset(..., predict=True)`) predicts the
  **last 5 observed days**, per the pytorch-forecasting docs ("the last max_prediction_length samples of each time
  series as prediction samples") and a check on `data/apollo_daily.csv` (a series ending at time_idx 119 / 2026-06-28
  was predicted for time_idx 115–119 / 2026-06-24..28). `gen_forecasts.py` (`start = datetime(2026, 8, 11)`) and
  `save_forecast_csv()` (`start_date + 1 day`) then date them after the last observation; the August
  `outputs/real/forecast_results.csv` (2026-08-12..16) therefore holds values for each community's last observed days.
  The dashboard shows them as "days ahead" (`dashboard/app.py` line 533). This run dates each prediction correctly and
  uses it as an out-of-sample check (I5). The report's benchmark forecasting result (§3.6, Table 3.3) was produced by
  the same code; whether that changes its interpretation was not examined here.
* **D2 — DBSCAN** flags nearly every community (I2). **D3 — GNN** outputs are near zero (I3). **D4 — churn** depends on
  sampling (I4). **D5 — toxic-bert** is driven by profanity (Part C agreement).
* **D6 — `MesoLayer`'s "toxicity_rate"** is a mean score, not a rate; the report's eq. 3.1 calls it "mean toxicity
  rate". Correctly reproduced here; flagged so nobody compares it with `toxic_rate` in the daily series.

## Part E — incidents during the run

* **GPU faults and silent corruption.** Pass-1 scoring hit three NVIDIA driver faults (18:29:43, 18:48:32, 18:52;
  Windows `nvlddmkm` events 13/14/153; "illegal memory access", "illegal instruction"). A wrapper restarted from the
  cache. The first join then found one NaN score (`p1ztawp`), so the GPU had returned a wrong value without crashing.
  Rule fixed before any comparison: re-score everything (pass 2), accept a score only if both passes agree within
  1e-4, otherwise use a CPU score. Outcome: 1,087,796 agree; 10 CPU-adjudicated, including a NaN in pass 2
  (`p27dr90`, a pass that never crashed) and 8 pass-1 values off by up to 8.6e-4. Below the tolerance, 22 agreeing
  comments differ by > 1e-6 (max 6.9e-5); none crosses 0.5 or 0.6 and together they can move a daily mean by at most
  2.6e-6. Final scores match fresh CPU re-scores within 1.2e-6 (3,520 comments). After this, all remaining model work
  ran on CPU.
* **TFT first ran on the GPU by mistake.** In PowerShell `$env:CUDA_VISIBLE_DEVICES=''` deletes the variable instead of
  hiding the GPU; `forecast_eval.json` recorded the RTX 2060. The forecast was re-run with `CUDA_VISIBLE_DEVICES=-1`
  (recorded device `cpu`); every forecast number in this document is from the CPU run. Same-seed CPU re-runs are
  identical (max |diff| 0).
* **Filter script change.** The real dump marks some removed comments `[ Removed by Reddit ]` (66 of 78,435 eligible
  comments in the first 2,000,000 lines); the appendix script kept them as text. One line was changed (Part F).

## Part F — edits to existing files

**None.** `git diff` against `origin/main` is empty; `models/` was not written to (TFT checkpoints went to
`data/real_torrent_2026_08/tft_models/`).

New files: `scripts/local/filter_arctic_dump.py` (the appendix, with one changed line, below),
`scripts/local/torrent_2026_08/*.py` (14 scripts), `data/real_torrent_2026_08/.gitignore` (`*`, `!.gitignore` — the
root `.gitignore` only ignores `data/*.csv` one level deep), `docs/NEW_DATA_NOTES.md`,
`docs/HANDOFF_torrent_data.md`, `outputs/real_torrent_2026_08/*`.

The one code change to the appendix script, line 67:

```
- REMOVED = {"[deleted]", "[removed]", ""}
+ REMOVED = {"[deleted]", "[removed]", "", "[ Removed by Reddit ]"}  # last one: real RC_2026-08 marker, seen in testing
```

Before committing, the two machine-specific paths in its docstring usage example (lines 19–20) were replaced with
the placeholders `<RC_ZST>` and `<OUTPUT_DIR>` (docstring only; behaviour unchanged).

Environment (not repo files): the venv held only the serving stack, so the ML stack was installed **into `venv\`**
(not global), pinned to the versions of the global Python that trained `models\`: torch 2.13.0+cu132 (matching the
driver's CUDA 13.2), transformers 4.57.6, scikit-learn 1.7.2, networkx 3.3, pytorch-forecasting 1.8.0, lightning 2.6.5,
matplotlib 3.10.7, scipy 1.15.3, zstandard 0.25.0 and their dependencies. The install added packages and upgraded
setuptools 65.5.0 → 78.1.0; `pip list` after the run shows numpy 2.2.6, pandas 2.3.3, streamlit 1.65.0, fastapi
0.162.5, uvicorn 0.54.0, pydantic 2.14.3, plotly 7.1.0, anthropic 2.9.0, psycopg2-binary 2.9.13 — the same versions
as before the install. The SNAP graph and the
retrained models were present and were not rebuilt.

## Part G — NEEDS ME

1. **Forecast labelling (D1)** affects the committed August `outputs/real/forecast_results.csv`, the dashboard and,
   possibly, how the report's forecasting results are read. Fixing it is a change to existing pipeline code — your and
   Kashaf's call, not done here.
2. **Human labels.** The agreement figure is from one LLM labeller. A person labelling
   `data/real_torrent_2026_08/label_sample_blind.csv` with the same guidelines would give a human-agreement figure.
3. **GPU stability.** The RTX 2060 produced silent numeric errors under sustained load (Part E). Any future GPU result
   on this machine needs the same two-pass check.
4. **Commit** (step 9) only on your reply "commit".

## Part H — claims audit

Every number above, with the command that printed it and where. `S` = `scripts\local\torrent_2026_08`; JSON fields
are in `outputs\real_torrent_2026_08\` unless a path is given. Claims without a row here have been removed.

| Claim | Value | Command | Evidence (output line / field) |
|---|---|---|---|
| dump size | 51,525,496,195 B | `Get-Item RC_2026-08.zst` | `Length` |
| dump SHA-256 | 30da6cc4… | `filter_arctic_dump.py … --hash` | summary `input_sha256` |
| lines in dump | 367,543,312 | full filter | `Done: … from 367,543,312 lines`; summary `stats.lines_read` |
| filter speed / time | 156k–169k lines/s (2M test runs of 11.8–12.8 s); 179.6k lines/s over the full run; 2,046 s | filter test; full filter | `in 11.8s` / `in 12.8s` for 2,000,000; summary `seconds` 2046.0 |
| `[ Removed by Reddit ]` | 66 in first 2M lines | `S\verify_filter.py` (before fix) | `removal-like bodies NOT in REMOVED: {'[ Removed by Reddit ]': 66}` |
| filter test gate, determinism | 8/8; SHA `1328f360…` ×2 | `S\verify_filter.py`; `sha256sum` | `GATE RESULT: PASS`; two identical hashes |
| matched / deleted / eligible / kept | 14,834,419 / 698,538 / 14,135,881 / 1,087,806 | `S\check_filter_output.py` | gate row 1–2; volume table footer |
| sub-days, hitting cap | 1,209; 934 | `S\check_filter_output.py` | footer `sub-days 1209; sub-days hitting cap 934` |
| kept rows identical to dump | 1,087,806 found, 0 mismatched | `S\verify_kept_rows.py` | gate rows |
| scoring speed, budget, quota | 313.91/s; 2,260,152; N = 1000 | `S\scoring.py bench`, `plan` | `scoring_benchmark.json`, `scoring_plan.json` |
| GPU faults | 18:29:43, 18:48:32, 18:52 | `Get-WinEvent` (System log) | `nvlddmkm` events 13/14/153 |
| pass agreement | 1,087,796 agree; 10 CPU | `S\scoring.py adjudicate`, `join` | `data\…\scoring_resolution.json` |
| sub-tolerance differences | 22 > 1e-6, max 6.9e-5, 0 cross 0.5/0.6, ≤ 2.6e-6 | inline analysis of `scoring.compare_passes()` (session command) | printed lines `agreeing with |p1-p2| > 1e-06: 22` etc. |
| re-score checks | max 1.6e-7 (GPU 20), 2.05e-7 (CPU 20), ≤ 1.19e-6 (CPU 3,500) | `S\check_scores.py` | gate rows |
| mean toxicity; toxic rate | 0.118683; 0.0950 (≥ 0.6), 0.1072 (≥ 0.5) | `S\check_scores.py` | `toxicity_score not degenerate`, `is_toxic` rows |
| authors | 450,940 | `S\run_pipeline.py` | `data_provenance.json` `authors` |
| qualifying communities | 38 of 39 | `S\build_daily.py` | `forecast runs … 38 of 39 communities qualify` |
| daily std within communities | 0.0072 / 0.0135 / 0.0549 | `S\build_daily.py` | `toxicity varies within every community` |
| CHI, instability, trend, P, E, churn ranges | see Part C | `S\run_pipeline.py` | gate rows `meso.*` |
| actions, clusters, alerts, silhouette | see Part C; LOW 35 / MEDIUM 4; 0.3432 | `S\run_pipeline.py` | gate rows; log `K-Means silhouette score: 0.3432` |
| independent recomputation | 15/15 | `S\verify_outputs.py` | `GATE RESULT: PASS (15/15 checks passed)` |
| spread, ranks, ρ | 0.1959; ranks; 0.916 | `S\verify_outputs.py` | `expectations_check.csv`, ranking table |
| TFT metrics and spread | Part C I5 table | `set CUDA_VISIBLE_DEVICES=-1 && S\forecast.py` | `forecast_eval.json` `per_seed`, `across_seeds` |
| TFT same-seed reproducibility | max |diff| 0 | scratch script `tft_same_seed.py` (session command, outside the repo) | `run A vs run B: max |diff| … = 0.000e+00` |
| I1–I5 | Part C | `S\investigate.py` | `investigations.json` |
| label counts, agreement, profanity split | Part C | `S\label_sample.py`, `S\label_agreement.py` | `label_sample_design.json`, `label_agreement.json` |
| predict window check | time_idx 115–119 for a series ending at 119 | scratch script `check_predict_window.py` (session command, outside the repo) | `decoded index: … 'time_idx_first_prediction': 115` |
