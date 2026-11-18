# Handoff — RC_2026-08 torrent data (for Kashaf)

What exists, how to regenerate it, what each output means, what was checked, and what is **not** measured.
Nothing was loaded into the database and no service (dashboard, web app, API, monitoring) was edited.
Full record of decisions, findings and the claims audit: `docs/NEW_DATA_NOTES.md`.

## 1. Source and scope

* Academic Torrents "Reddit comments/submissions 2026-08" (Arctic Shift), infohash
  `f4848163a5fb650e4b15eee0202858447bca120a`; comments file `RC_2026-08.zst` only (51,525,496,195 bytes, SHA-256
  `30da6cc45cb9b6cad5a910888bcebeb3213715365d3373e9bcd9f1d01850deaa`, 367,543,312 lines). Never unpacked to disk.
* 39 subreddits (`DEFAULT_SUBREDDITS` in `scripts/local/filter_arctic_dump.py`), one window 2026-08-01..2026-08-31 UTC.
* Separate namespace: inputs `data/real_torrent_2026_08/` (gitignored — usernames and text), outputs
  `outputs/real_torrent_2026_08/` (aggregates only). Not mixed with `real_v2`, `outputs/real/` or the benchmark.

## 2. Fixed parameters (all in `scripts/local/torrent_2026_08/config.py`, fixed before results)

| Parameter | Value |
|---|---|
| Filter cap / seed | 1,000 comments per subreddit per UTC day, reservoir sample, seed 42 |
| Scoring | `unitary/toxic-bert`, first 512 characters, 256 tokens, P("toxic"); budget rule gave quota N = 1,000, so **all 1,087,806** filtered comments were scored |
| Toxic thresholds | `is_toxic` at 0.6 (`main.Config.TOXICITY_THRESHOLD`); `is_toxic_050` at 0.5 (the August real run's cut) |
| Coverage rule | a day counts only with ≥ 50 scored comments; a community is forecast only with ≥ 24 consecutive counting days; the TFT gets exactly that run; missing days are never filled |
| TFT | lookback 14, horizon 5, 6 epochs (as `gen_forecasts.py`), seeds 42/43/44, canonical seed 42, run on CPU |
| Label sample | 300 rows, seed 2026, 100 per score band (< 0.1, 0.1–0.6, ≥ 0.6), spread over subreddits |

**Qualifying communities (38 of 39):** every subreddit except r/worldpolitics, each with the full run 2026-08-01..31.
r/worldpolitics has 1,918 comments in the month; only 19 days reach 50 comments and its longest run is 5 days, so it
has meso scores but no forecast.

## 3. Files

### 3a. Outputs — `outputs/real_torrent_2026_08/` (small, no usernames or comment text)

| File | Rows | Columns / content | Meaning |
|---|---|---|---|
| `meso_report.csv` | 39 | subreddit, community_health_index, toxicity_rate, toxicity_recent, toxicity_trend, instability_score, polarization, echo_chamber_index, churn_rate, total_comments, timestamp, gnn_risk_raw, gnn_risk, recommended_action, in_snap_graph | Same columns as `outputs/real/meso_report.csv` plus `in_snap_graph`. `toxicity_rate` is the community's MEAN toxicity score (meso-layer definition), not a thresholded rate |
| `clusters_report.csv` | 39 | meso columns + cluster, pca_x, pca_y, dbscan_label, is_outlier | K-Means (5) + PCA + DBSCAN, as `outputs/real/clusters_report.csv` |
| `alerts.json` | 39 alerts | subreddit, alert_level, chi, message, timestamp | `main.AlertSystem.evaluate` per community |
| `daily_series.csv` | 1,209 | subreddit, date, avg_toxicity, toxic_rate (≥ 0.6), toxic_rate_050, comment_count, unique_authors, meets_min_comments, in_forecast_run, time_idx, group_id | Per-subreddit per-day series; no zero-filled days |
| `coverage.csv` | 39 | subreddit, comments_scored, days_with_data, days_meeting_min, min_comments_per_day, longest_run_days, run_start, run_end, qualifies_forecast, reason | Outcome of the coverage rule |
| `forecast_results.csv` | 190 | day, date, predicted_toxicity, p10, p50, p90, risk_level, method, subreddit, interval_width, actual, out_of_sample, kind | Canonical TFT run (seed 42), layout of `outputs/real/forecast_results.csv` plus 4 columns. **`date` is the day actually predicted (2026-08-27..31, observed days held out of training), not a future day** — see §5 |
| `forecast_seeds.csv` | 570 | seed, subreddit, day, date, time_idx, p10, p50, p90, actual, persistence, mean14, encoder_min, encoder_max, encoder_days, training_cutoff_time_idx, out_of_sample, failed | All three seeds with baselines |
| `forecast_eval.json` | — | per seed and across seeds: MAE vs persistence and 14-day mean, coverage, widths by day | TFT evaluation on real data |
| `filter_volume_by_subreddit.csv` | 39 | subreddit, days_with_data, eligible_before_cap, kept, days_hit_cap, min_kept_per_day | Real volume before the cap |
| `expectations_check.csv` | 11 | id, expected, measured, match | Report expectations (fixed before results) vs this run |
| `investigations.json` | — | I1–I5 | Causes of every "differs" and odd output |
| `label_agreement.json` | — | precision/recall/F1, intervals, confusion matrices, profanity breakdown | Agreement with LLM-assigned labels |
| `data_provenance.json` | — | source, hash, counts, date range | Same role as `outputs/real/data_provenance.json` |

### 3b. Local inputs — `data/real_torrent_2026_08/` (gitignored; regenerate with §4)

| File | Rows | Columns | Notes |
|---|---|---|---|
| `RC_2026-08_filtered.csv` | 1,087,806 | id, subreddit, author, body, score, created_utc | **usernames + text** |
| `RC_2026-08_filtered.summary.json` | — | stats, per subreddit-day eligible/kept/hit_cap, SHA-256 | |
| `scoring_sample.csv` | 1,087,806 | as the filtered CSV | identical content (quota = cap) |
| `score_cache.csv`, `score_cache_pass2.csv` | 1,087,806 each | id, toxicity_score | GPU pass 1 and pass 2 |
| `score_cache_cpu.csv` | 10 | id, toxicity_score | CPU scores for the 10 comments where the passes disagreed |
| `scored_comments.csv` | 1,087,806 | filtered columns + toxicity_score, score_source, is_toxic, is_toxic_050 | **usernames + text** |
| `label_sample_scored.csv` | 300 | id, subreddit, band, toxicity_score, is_toxic, is_toxic_050, body | no usernames; text = first 512 chars |
| `label_sample_blind.csv` | 300 | id, subreddit, body | shuffled; no scores |
| `label_sample_labels.csv` | 300 | id, label (toxic / not_toxic / unclear) | single LLM labeller |
| `labelling_guidelines.md` | — | | written before labelling |
| `PROVENANCE.md` | — | | full provenance incl. GPU faults |
| logs, `scoring_*.json`, `rescore_*.csv`, `spotcheck_ids.txt`, `tft_models/` | | | run records |

## 4. Regenerate everything from the .zst

From the repo root with the venv (`venv\Scripts\python.exe`, Python 3.10; the run used torch 2.13.0+cu132,
transformers 4.57.6, scikit-learn 1.7.2, pytorch-forecasting 1.8.0, lightning 2.6.5, networkx 3.3, zstandard 0.25.0).
Needs `data/reddit_hyperlinks.tsv` and `models/gnn_model.pt`, `models/moderation_recommender.pkl`.
`<RC_ZST>` = path to `RC_2026-08.zst`. `S` = `scripts\local\torrent_2026_08`.

```
python scripts\local\filter_arctic_dump.py --input <RC_ZST> --output data\real_torrent_2026_08\RC_2026-08_filtered.csv --start 2026-08-01 --end 2026-08-31 --cap 1000 --seed 42 --hash
python S\check_filter_output.py                 # gate + filter_volume_by_subreddit.csv
python S\verify_kept_rows.py --input <RC_ZST>   # every kept row re-read from the dump (~30 min)
python S\scoring.py bench
python S\scoring.py plan
python S\scoring.py sample
python S\scoring.py score                       # pass 1 (GPU)
python S\scoring.py score --pass2               # pass 2 (GPU)
python S\scoring.py adjudicate                  # CPU for every disagreement
python S\scoring.py join                        # scored_comments.csv
python S\check_scores.py                        # gate
python S\build_daily.py                         # daily_series.csv, coverage.csv + gate
python S\run_pipeline.py                        # meso, clusters, alerts, provenance + gate
set CUDA_VISIBLE_DEVICES=-1 && python S\forecast.py   # TFT on CPU, 3 seeds + gate
python S\verify_outputs.py                      # independent recomputation + expectations_check.csv
python S\investigate.py                         # investigations.json
python S\label_sample.py                        # label sample (blind + scored)
python S\label_agreement.py                     # needs label_sample_labels.csv
```

GPU scores are not bit-identical to CPU (differences up to ~1e-6); the TFT on CPU is identical for the same seed.

## 5. What each output means — and how not to read it

* **CHI / instability / polarisation / echo / churn** are the report's formulas (eq. 3.1, 3.2) applied to this corpus.
  `churn_rate` is computed on a capped sample, so it measures author overlap *within the sample*; it rises with
  community volume (Spearman 0.377 with pre-cap volume).
* **Forecast:** the existing `TFTForecaster` trains on all but the last 5 days and `predict_quantiles()` predicts the
  **last 5 observed days**. `gen_forecasts.py` labels those values as the 5 days after the data (in
  `outputs/real/forecast_results.csv`, values for each community's last observed days are labelled 2026-08-12..16).
  Here every prediction carries the date it really predicts (2026-08-27..31) and the observed value, so it is a real
  out-of-sample check. No forward (September) forecast exists. The dashboard's Forecast page
  (`dashboard/app.py` lines 516–542) titles it "5-day toxicity forecast" and plots `day` on an axis labelled
  "days ahead", so showing this file there unchanged would present held-out August days as the future. Whether and
  how to show it is your decision; nothing in the dashboard was changed.
* **gnn_risk** is a percentile rank of raw outputs that are all between 1e-6 and 1.1e-5; r/Superstonk is absent from
  the 2014–2017 SNAP graph and has no value.
* **DBSCAN `is_outlier`** is True for 34 of 39 communities (August: 24 of 24): with eps 0.5 almost every point is noise.
* **alert_level** uses `AlertSystem` (CHI < 70 → MEDIUM): 4 MEDIUM, 35 LOW.

## 6. Checks that passed (each script prints a gate table and exits non-zero on failure)

| Gate | Result |
|---|---|
| Filter vs full JSON recount, first 2,000,000 lines (`verify_filter.py`) | 8/8 PASS |
| Filter determinism (2 runs, same seed, 2,000,000 lines) | identical SHA-256 |
| Filtered CSV vs summary (`check_filter_output.py`) | 12/12 PASS |
| Every kept row re-read from the dump (`verify_kept_rows.py`) | 3/3 PASS (1,087,806 found, 0 mismatched) |
| Scores (`check_scores.py`) | 17/17 PASS |
| Daily series + coverage (`build_daily.py`) | 13/13 PASS |
| Meso, clusters, alerts (`run_pipeline.py`) | 28/28 PASS |
| TFT forecast (`forecast.py`) | 17/17 PASS |
| Independent recomputation (`verify_outputs.py`) | 15/15 PASS |

## 6b. Improvement experiment (additive, pre-registered)

`outputs/real_torrent_2026_08/improvement/RESULTS.md` (pre-registration in `PREREGISTRATION.md` next to it): four
toxic-bert scorer variants chosen on a new 300-comment DEV set and tested once on the existing 300 — **none met the
pre-registered improvement bar** (V1 = max of insult/threat/identity_hate/severe_toxic came closest); forecast blends
and a 5-origin backtest — **no blend beats the 14-day mean**, which is the best forecaster tested. Nothing was applied
to the outputs listed above.

## 7. NOT measured

* **Real-world instability accuracy** — there are no labels of real destabilisation; no such number exists.
* **Human-labelled toxicity accuracy on this corpus** — the 300 labels are from a single LLM labeller; agreement is
  reported as exactly that. A human labelling of `label_sample_blind.csv` (same guidelines) would turn it into a
  human-agreement figure.
* **Forward forecast skill** — no September data was used; the TFT is only evaluated on held-out August days.
* **Anything about deleted/removed comments** (698,538 in these subreddits) — their text is gone.
* **Communities' full volume** — scores describe a capped sample of ≤ 1,000 comments per day; the pre-cap volume is in
  `filter_volume_by_subreddit.csv`.
