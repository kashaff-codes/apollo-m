# APOLLO-M — Forecasting Instability in Digital Communities

A multi-layer pipeline that scores Reddit comments for toxicity, aggregates them
into a per-community health index and a trend-aware instability score, clusters
communities without labels, and forecasts each community's toxicity five days
ahead with quantile bands using a Temporal Fusion Transformer.

Final Year Project — Bahria University Karachi.
Kashaf Fatima · Rizwan Saleem · Aaqib Mehmood. Supervisor: Mr. Saghir Ahmed.

| | |
|---|---|
| Analyst console | **https://apollo-m.streamlit.app** — create an account (email, or Google once enabled); accounts are shared with the web view |
| Public web view | https://apollo-m.vercel.app — no login |
| REST API | https://apollo-api-tllm.onrender.com/docs |
| Monitoring | https://apollo-m-monitoring.onrender.com — Grafana, read-only, no login · [`/prometheus/`](https://apollo-m-monitoring.onrender.com/prometheus/) |
| Companion system | [CEREBRO](https://github.com/kashaff-codes/CEREBRO) — threat and misinformation intelligence |

Free-tier services sleep when idle: the first request after a quiet spell can take up to a minute.
Open Grafana a few minutes before showing it: Prometheus starts without history after a wake.

**For a reviewer:** [`docs/PRODUCT_BRIEF.md`](docs/PRODUCT_BRIEF.md) answers what it is
for, who would buy it, how reliable it is (with evidence), how data gets in, and
how a customer would deploy it.

---

## Accounts and access

One identity for the web view and the console: **Supabase Auth**. Passwords are
hashed by Supabase and never seen by APOLLO-M; a session is a signed JWT (about
an hour) plus a rotating refresh token, kept for 30 days when *Keep me signed in*
is ticked, and revoked on the server at sign-out. Google sign-in uses OAuth 2.0
with PKCE. Five failed sign-ins lock an email out for fifteen minutes; passwords
need at least ten characters, not only letters or only digits.

What a user may read is decided by the database, not the app
([`web/supabase/accounts.sql`](web/supabase/accounts.sql)): each user reads their
own profile and activity; an **admin** (a role in `public.profiles`, granted in
SQL, not a password in code) reads everyone's, which is what the console's
*Admin* page shows. The activity log is append-only. There are no shared
accounts and no public demo button; a developer sign-in exists only with
`APOLLO_DEV_LOGIN=1`, for local work and tests.

---

## Live — data that keeps arriving

Nobody runs a script to keep APOLLO-M current.

| Path | What happens | How fresh |
|---|---|---|
| **Scheduled ingest** — [`scripts/live_ingest.py`](scripts/live_ingest.py), run by [`.github/workflows/live-ingest.yml`](.github/workflows/live-ingest.yml) | Every six hours: the newest comments for every community in [`live/communities.txt`](live/communities.txt) from Arctic Shift, bots filtered, scored with toxic-bert, 7-day Community Health Index and a next-day forecast recomputed, aggregates committed to `outputs/live/` | Arctic Shift runs minutes behind Reddit; first run: 7,247 comments from 52 communities, newest 18 seconds old |
| **Topic watch** — console page, [`apollo_live.py`](apollo_live.py) | On demand: every post or comment mentioning a phrase in a set of communities over the last hour to week, with where and when, straight from Arctic Shift | Minutes |
| **Monthly archive** (Academic Torrents `RC_YYYY-MM`) | Every comment in a month, for exact volumes, backfill and calibration; it is how we checked the live sample measures the same thing (ρ 0.916) | Weeks; published monthly |

Adding a community is one line in `live/communities.txt` (53 tracked). Comment
text and usernames are never committed: the ingest publishes per-community
aggregates, and keeps authors only as salted hashes in the Actions cache, for
churn. Keyword search across *all* of Reddit would need Reddit's own API; Arctic
Shift's search names the communities to look in, which is why Topic watch
starts from preset, editable lists.

---

## Four corpora, one pipeline

The same pipeline has been run on four corpora. They are kept in separate
directories and never mixed; [`apollo_datasets.py`](apollo_datasets.py) names
each one, where its outputs live, and what its forecast rows mean. The console's
sidebar switches between them and every page says which one it is showing.

| Corpus | Window | Communities | Comments | Forecast | Built by |
|---|---|---|---|---|---|
| **Benchmark** — declared simulation, ground truth recorded | 120 days | 60 | 261,337 | future + backtest | team |
| **Arctic Shift API** — collected live | 2026-07-02 → 08-11 | 24 | 116,853 | future + backtest | Kashaf |
| **RC_2026-08** — the complete August monthly dump (51.5 GB, 367.5M comments), every kept row re-verified | 2026-08-01 → 08-31 | 39 | 1,087,806 | held-out evaluation | Rizwan |
| **Arctic Shift rebuild** — one contiguous window, bots and boilerplate removed, real SNAP graph | 2026-10-23 → 11-15 | 38 | 257,129 sampled · 246,239 analysed | future + backtest | Aaqib |

Why a simulation as well: no public dataset labels which real communities went on
to destabilise, so recall can only be measured where the answer was planted.
`simulate_data.py` gives 15 of 60 communities a rising trend, records it in
`data/ground_truth.json` (read by no pipeline stage), and `validate_simulation.py`
measures recovery. Toxicity there is still real: each comment carries
`unitary/toxic-bert`'s score for the exact text it contains.

## What the numbers say — including what did not work

**On the benchmark** (`outputs/validation.json`):

| | Result |
|---|---|
| TFT predicts a rising trend for the planted communities | **15/15**, slope ROC-AUC **0.732** |
| Ranking by trend-aware instability score | ROC-AUC 1.000 — a wiring check: it searches for the shape that was planted |
| Ranking by Community Health Index | ROC-AUC **0.575** |
| Ranking by raw toxicity alone | ROC-AUC 0.649 |

The composite CHI ranks the planted communities *worse than one of its own inputs*:
an index of present-tense health cannot rank communities by how fast they are
changing. That is the project's central finding, published as a negative result.

**On real Reddit:**

| Finding | Evidence |
|---|---|
| Which communities are toxic is stable across months and collection methods | Spearman ρ of community toxicity: **0.916** (API vs August dump, 14 shared), **0.892** (August dump vs Oct–Nov, 38), 0.864 (API vs Oct–Nov, 14) |
| The TFT beats repeating the last value, but not the 14-day mean | held-out MAE — API corpus 0.026 vs 0.035 vs **0.024**; RC_2026-08 0.020 vs 0.016 vs **0.011**; rebuild 0.021 vs 0.024 vs **0.019** (TFT / last value / 14-day mean). On the benchmark, where trends are planted, the TFT wins (0.054 vs 0.077 vs 0.057). |
| Its 80% band is roughly calibrated on held-out days | 73–84% of held-out actuals fall inside p10–p90, by corpus |
| **Forward test: on days that had not happened yet, the result holds** | trained on August only, predicted 1–5 September, then scored 91,101 comments actually posted on those days: MAE 14-day mean **0.0166**, TFT 0.0178 (worse by 0.0012, 95% CI 0.0003–0.0021), last value 0.0181; the 80% band covered only 67%. Same pattern on 16–17 November. [`outputs/forward_test/summary.json`](outputs/forward_test/summary.json) |
| toxic-bert largely detects swearing | against 291 independently labelled RC_2026-08 comments, precision **0.21** at the 0.6 threshold (95% CI 0.13–0.30); 52 of its 78 false positives contain profanity |
| Four scorer variants did not improve on it | Rizwan's pre-registered test: thresholds fixed on a dev set, tested once, none met the bar |

The supervised TF-IDF + logistic-regression classifier reaches accuracy 0.901 and
F1-macro 0.702 on held-out Jigsaw/Davidson labels; it is a separate model from
toxic-bert. Modules named in the architecture but **not invoked** by the
pipeline — the in-Apollo misinformation classifier, the GraphSAGE GNN, the
moderation recommender — are marked unevaluated in `outputs/metrics.json` rather
than given a number.

## How it connects

```mermaid
flowchart LR
  subgraph Sources
    AS[Arctic Shift API]
    RC[RC_2026-08 dump]
    SIM[simulate_data.py]
  end
  subgraph Pipeline["Pipeline (local, torch)"]
    MI[Micro · toxic-bert] --> ME[Meso · CHI, instability] --> UN[Unsupervised · K-Means, DBSCAN]
    ME --> MA[Macro · TFT p10/p50/p90]
  end
  Sources --> MI
  MA --> OUT[(outputs/&lt;corpus&gt;/<br/>committed)]
  UN --> OUT
  OUT --> DB[(PostgreSQL)]
  OUT --> SL[Streamlit console]
  OUT --> API[FastAPI]
  DB --> API
  OUT -. db_setup.py .-> SB[(Supabase)] --> WEB[Next.js web view]
  API -- /metrics --> PROM[Prometheus] --> GRAF[Grafana]
  OUT -- CSV via GitHub --> GRAF
  CEREBRO[CEREBRO API] -- /metrics --> PROM
  SL -. links .-> WEB & GRAF & CEREBRO
```

| Surface | Stack | Reads |
|---|---|---|
| Analyst console (`dashboard/`) | Streamlit, Plotly | committed outputs of all four corpora |
| REST API (`api/`) | FastAPI, JWT | PostgreSQL, **falling back to the committed outputs of the same corpus** when the database is unreachable — every response says which (`"source"`) |
| Public web view (`web/`) | Next.js 16 App Router, React 19, Supabase | Supabase Postgres via row-level security ([README](web/README.md)) |
| Monitoring (`monitoring/hosted/`) | Prometheus 3.16, Grafana 13.3 | both APIs' `/metrics`, and the corpus CSVs for history |

**Database status.** The Render PostgreSQL the API used was deleted when Render's
free tier expired it; that is why the API now falls back to the committed
outputs (`/health` reports `"data_source": "csv:real_api_2026_07"` and
`apollo_database_up` is 0 in Grafana). To restore it, see
[`docs/RESTORE_DATABASE.md`](docs/RESTORE_DATABASE.md). Supabase, behind the web
view, is unaffected.

## The console

Eleven pages, each opening with which corpus it shows: *Overview* (health
landscape, bands, ranking), *Dynamics* (an animated level-vs-momentum phase space
and a community × day heatmap), *Communities* (CHI gauge, a waterfall of the four
weighted penalties, component radar, history with forecast band), *Forecast*
(forecasts drawn as what they are — future or held-out — with a backtest against
naive baselines on every corpus), *Actions*, *Clusters* (3D), *Across corpora*,
*Evidence* (every result above, negative ones included), *Live* (the scheduled
ingest's latest measurements: when, how fresh, health of every tracked
community), *Topic watch* (ask Arctic Shift about a phrase now: where and when it
is being discussed, with the posts) and *Monitoring*. Charts draw themselves in
on load (and do not, under reduced-motion settings).

Alert bands are one rule shared by every surface ([`apollo_bands.py`](apollo_bands.py)):
✕ CRITICAL < 65 ≤ ▲ HIGH < 75 ≤ ◆ MEDIUM < 85 ≤ ● LOW. A test evaluates the API,
loader, exporter and Supabase view against it across the whole range.

## Quick start

```bash
pip install -r requirements.txt      # console + API, no torch
streamlit run dashboard/app.py       # console  → http://localhost:8501
uvicorn api.main:app --port 8010     # API      → http://localhost:8010/docs
```

Reproducing the pipeline (torch, transformers, the TFT):

```bash
pip install -r requirements-train.txt
python simulate_data.py              # benchmark (~15 min; scores texts with toxic-bert)
python regen_meso.py                 # micro + meso + clustering → outputs/
python gen_forecasts.py              # TFT → outputs/forecast_future.csv + forecast_backtest.csv
python gen_forecasts.py --dataset real_api_2026_07
python validate_simulation.py        # recovery → outputs/validation.json
python build_live_seed.py            # the Live page's real comments
```

The real corpora: `scripts/local/torrent_2026_08/` (Rizwan — every step gated,
regeneration in `docs/HANDOFF_torrent_data.md`) and `build_corpus.py` →
`run_full_from_cache.py` (Aaqib). Raw comment text and usernames stay out of git;
only aggregates are committed.

| Requirements file | Installs | Used by |
|---|---|---|
| `requirements.txt` | console + API, no ML | Streamlit Cloud (fixed filename) |
| `requirements-api.txt` | API only | Render |
| `requirements-train.txt` | full pipeline incl. torch | local training |

## Tests

```bash
pytest tests/                        # 229 tests, ~1 min
APOLLO_SLOW=1 pytest tests/test_micro.py      # + the real toxic-bert
APOLLO_REQUIRE_CORPUS=1 pytest tests/         # on a machine with the scored corpus
```

| Kind | Where | Checks |
|---|---|---|
| Smoke | `test_smoke.py`, `test_dashboard.py` | outputs exist; every console page renders headless on two corpora (Streamlit AppTest) |
| Sanity | `test_sanity.py` | the real corpus is not flat, zero-filled or synthetic-looking |
| Validation | `test_validation.py`, `test_datasets.py` | metric thresholds; one contract for all four corpora, incl. that each forecast file means what the registry says |
| Regression | `test_chi.py`, `test_paper_inputs.py` | the CHI `*100` overflow; alert-band parity across six implementations; SHA-256 of every input the paper is reproduced from |
| Isolation | `test_isolation_ingest.py` | the dump filter on a real `.zst` and the corpus builder on edge-case rows |
| Integration | `test_api.py`, `test_monitoring.py` | the API with its database gone; every metric a Grafana panel queries is one the API exports; every live-status field a panel reads exists |
| Live path | `test_live.py` | one CHI definition; health window and minimum-comment rule; churn as the pipeline defines it; forecast band contains its median; Topic watch reports throttled communities instead of dropping them |

Tests that need the git-ignored corpus are marked `corpus` and skip with the
reason when it is absent. CI runs the suite on every push and pull request.

## Layout

| Path | What |
|---|---|
| `main.py` | Pipeline layers: micro, meso, unsupervised, macro, alerts |
| `apollo_datasets.py` · `apollo_bands.py` · `apollo_chi.py` | The corpus registry · the alert-band rule · the CHI formula, defined once |
| `apollo_live.py` · `scripts/live_ingest.py` · `live/` | Topic watch · the scheduled ingest · the tracked-community list |
| `scripts/forward_test.py` | Forecasts of days that had not happened, scored against reality |
| `models/tft_forecaster.py` | Temporal Fusion Transformer: calendar-day index, future forecasts, backtest |
| `outputs/` · `outputs/real*/` | Benchmark outputs (paper inputs, pinned) · one directory per real corpus |
| `dashboard/` | Streamlit console: `app.py`, `charts.py`, `data.py`, `theme.py` |
| `api/` | FastAPI backend |
| `web/` | Public web view — Next.js on Supabase |
| `monitoring/hosted/` | Prometheus + Grafana for Render; `build_dashboards.py` generates the dashboards |
| `scripts/local/` | RC_2026-08 pipeline (Rizwan) |
| `docs/` | Technical report, paper and its reproduction scripts, handoffs |

## Deployment

`render.yaml` declares the API and the monitoring service; Streamlit Cloud serves
the console from `requirements.txt`; Vercel serves `web/`. No secrets are
committed: `JWT_SECRET`, `ANTHROPIC_API_KEY` (plus `ANTHROPIC_WORKSPACE_ID` for a
user-scoped `sk-ant-usr-` key), `DATABASE_URL`, `CORS_ORIGINS` and
`GF_SECURITY_ADMIN_PASSWORD` are set in each host's dashboard. The live ingest
runs in GitHub Actions and needs no secret. See
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) and
[`docs/RESTORE_DATABASE.md`](docs/RESTORE_DATABASE.md).
