"""
APOLLO-M FastAPI Backend
Exposes pipeline results via REST API for frontend/dashboard consumption.

Endpoints:
    GET  /health                    — API health, and which data source is live
    GET  /datasets                  — every corpus the pipeline has produced
    GET  /communities               — All community health scores
    GET  /communities/{subreddit}   — Single community details
    GET  /alerts                    — All alerts
    GET  /alerts/{level}            — Alerts filtered by level
    GET  /forecast/{subreddit}      — five-day forecast for a subreddit
    GET  /clusters                  — Community clusters
    GET  /recommendations           — moderation recommendations
    GET  /summary                   — Dashboard summary stats
    GET  /metrics                   — Prometheus exposition
    POST /analyze                   — Score submitted text

Data source
-----------
PostgreSQL first, then the committed pipeline outputs for the same corpus.

The deployed API used to read only from the database, and the database was a
Render free-tier instance shared with CEREBRO. Render deletes those after their
free period; when it went, every data endpoint returned 500 — including
/metrics, so the monitoring stack had nothing to scrape either. The outputs the
database was loaded from are committed (outputs/real/ by default, see
APOLLO_DATASET), so serving them when the database is unreachable is not a
degraded answer: it is the same rows. Every response says which source it came
from, and /metrics publishes apollo_database_up so the loss is visible rather
than silent.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import OAuth2PasswordRequestForm

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.auth import (ACCESS_TOKEN_EXPIRE_MINUTES, Token, User,  # noqa: E402
                      authenticate_user, create_access_token, get_current_user)
from apollo_bands import ACTION, alert_band  # noqa: E402
from apollo_datasets import DATASETS  # noqa: E402
from apollo_datasets import get as get_dataset  # noqa: E402

# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="APOLLO-M API",
    description="AI Framework for Community Instability Forecasting",
    version="1.1.0",
)

# Which browser origins may call this API. Default "*" keeps local development
# and the demo working; set CORS_ORIGINS to a comma-separated list to restrict.
# allow_credentials stays off: this API authenticates with a Bearer token.
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins or ["*"],
                   allow_methods=["*"], allow_headers=["*"])

DASHBOARD_URL = os.getenv("APOLLO_DASHBOARD_URL", "https://apollo-m.streamlit.app")


# ── Request metrics ───────────────────────────────────────────
# Counted per route template (/communities/{subreddit}, not each name), so the
# label set stays bounded no matter what callers send.
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_req_count: dict[tuple[str, str, int], int] = defaultdict(int)
_lat_bucket: dict[tuple[str, float], int] = defaultdict(int)
_lat_sum: dict[str, float] = defaultdict(float)
_lat_n: dict[str, int] = defaultdict(int)
_STARTED = time.time()


@app.middleware("http")
async def _instrument(request: Request, call_next):
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        if path != "/metrics":
            dt = time.perf_counter() - t0
            _req_count[(request.method, path, status)] += 1
            _lat_sum[path] += dt
            _lat_n[path] += 1
            for b in _BUCKETS:
                if dt <= b:
                    _lat_bucket[(path, b)] += 1


# ── Data source: database, then committed outputs ─────────────
DB_SCHEMA = os.getenv("DB_SCHEMA", "apollo")
_SEARCH_PATH = f"-c search_path={DB_SCHEMA},public"
_DS_KEY = {"real": "real_api_2026_07"}.get(os.getenv("APOLLO_DATASET", "real").lower(),
                                           os.getenv("APOLLO_DATASET", "real").lower())
DATASET = get_dataset(_DS_KEY if _DS_KEY in DATASETS else "real_api_2026_07")

# After a failed connection, skip the database for this long instead of paying
# the connection timeout on every request.
_DB_RETRY_SECONDS = 60
_db_down_until = 0.0
_db_last_error = ""


class DatabaseUnavailable(RuntimeError):
    pass


def get_db():
    url = os.getenv("DATABASE_URL", "").strip()
    # 'prefer', not 'require': Render's internal hostname serves plaintext inside
    # its private network. Set PGSSLMODE=require where the network is untrusted.
    if url:
        return psycopg2.connect(url, options=_SEARCH_PATH, connect_timeout=5,
                                sslmode=os.getenv("PGSSLMODE", "prefer"))
    return psycopg2.connect(
        options=_SEARCH_PATH, connect_timeout=5,
        dbname=os.getenv("DB_NAME", "apollo_db"), user=os.getenv("DB_USER", "apollo"),
        password=os.getenv("DB_PASS", "apollo_pass"), host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"))


def query_db(sql: str, params=None) -> list:
    """Run a query, or raise DatabaseUnavailable if there is no database to run it on."""
    global _db_down_until, _db_last_error
    if time.time() < _db_down_until:
        raise DatabaseUnavailable(_db_last_error)
    try:
        conn = get_db()
    except psycopg2.OperationalError as exc:
        _db_down_until = time.time() + _DB_RETRY_SECONDS
        _db_last_error = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise DatabaseUnavailable(_db_last_error) from exc
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def database_up() -> bool:
    try:
        query_db("SELECT 1 AS ok")
        return True
    except Exception:  # noqa: BLE001
        return False


def _csv(name: str) -> pd.DataFrame:
    p = DATASET.file(name)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"{name} not available for {DATASET.key}")
    return pd.read_csv(p)


def _r(sub: str) -> str:
    return sub if sub.startswith("r/") else f"r/{sub}"


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _meso() -> pd.DataFrame:
    m = _csv("meso_report.csv")
    m["subreddit"] = m["subreddit"].map(_r)
    m["alert_level"] = m["community_health_index"].map(alert_band)
    return m


def _source() -> str:
    return f"csv:{DATASET.key}"


# ── Root ──────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "APOLLO-M API", "status": "ok", "interactive_docs": "/docs",
        "endpoints": ["/health", "/datasets", "/summary", "/communities", "/communities/{subreddit}",
                      "/alerts", "/alerts/{level}", "/forecast/{subreddit}", "/clusters",
                      "/recommendations", "/metrics"],
        "frontend_dashboard": DASHBOARD_URL,
    }


@app.get("/health")
def health():
    up = database_up()
    return {
        "status": "healthy",
        "database": "connected" if up else f"unavailable — {_db_last_error or 'not configured'}",
        "data_source": "database" if up else _source(),
        "dataset": DATASET.key,
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
    }


@app.get("/datasets")
def datasets():
    """Every corpus the pipeline has produced, and which one this API serves."""
    return {
        "serving": DATASET.key,
        "datasets": [{"key": d.key, "label": d.label, "kind": d.kind, "window": d.window,
                      "communities": d.communities, "forecast": d.forecast, "credit": d.credit,
                      "available": d.available()} for d in DATASETS.values()],
    }


# ── Authentication ────────────────────────────────────────────
@app.post("/auth/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token(data={"sub": user["username"], "role": user["role"]},
                                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return Token(access_token=token, token_type="bearer", role=user["role"],
                 expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)


@app.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current logged-in user info."""
    return current_user


# ── Communities ───────────────────────────────────────────────
# Per-community routes take {subreddit:path} so both "AskReddit" and
# "r/AskReddit" work: the slash in the second used to split the path, and the
# conventional way to name a community answered 404.
COMMUNITY_COLS = ["subreddit", "community_health_index", "toxicity_rate", "polarization",
                  "echo_chamber_index", "churn_rate", "total_comments"]


@app.get("/communities")
def get_communities():
    """Latest CHI scores for all communities."""
    try:
        rows = query_db("""
            SELECT DISTINCT ON (subreddit)
                subreddit, community_health_index, toxicity_rate,
                polarization, echo_chamber_index, churn_rate,
                total_comments, timestamp
            FROM community_health
            ORDER BY subreddit, timestamp DESC
        """)
        return {"communities": rows, "count": len(rows), "source": "database"}
    except DatabaseUnavailable:
        m = _meso()
        cols = [c for c in COMMUNITY_COLS + ["alert_level"] if c in m]
        return {"communities": _records(m[cols]), "count": len(m), "source": _source()}


@app.get("/communities/{subreddit:path}")
def get_community(subreddit: str):
    """Details for one subreddit."""
    subreddit = _r(subreddit)
    try:
        rows = query_db("SELECT * FROM community_health WHERE subreddit = %s "
                        "ORDER BY timestamp DESC LIMIT 1", (subreddit,))
        source = "database"
    except DatabaseUnavailable:
        m = _meso()
        rows, source = _records(m[m["subreddit"] == subreddit]), _source()
    if not rows:
        raise HTTPException(status_code=404, detail=f"Community {subreddit} not found")
    return {**rows[0], "source": source}


# ── Alerts ────────────────────────────────────────────────────
_LEVEL = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def _alerts_from_csv(level: str | None = None) -> list[dict]:
    m = _meso()
    if level:
        m = m[m["alert_level"] == level]
    out = [{"subreddit": r.subreddit, "alert_level": r.alert_level,
            "chi": round(float(r.community_health_index), 2),
            "recommended_action": ACTION[r.alert_level],
            "message": f"[{r.alert_level}] {r.subreddit} CHI={r.community_health_index:.1f} "
                       f"toxicity={r.toxicity_rate:.2%}"} for r in m.itertuples()]
    return sorted(out, key=lambda x: _LEVEL[x["alert_level"]])


@app.get("/alerts")
def get_alerts():
    """All alerts, most severe first."""
    try:
        rows = query_db("""SELECT DISTINCT ON (subreddit) subreddit, alert_level, chi, message, timestamp
                           FROM alerts ORDER BY subreddit, timestamp DESC""")
        rows.sort(key=lambda x: _LEVEL.get(x["alert_level"], 4))
        return {"alerts": rows, "count": len(rows), "source": "database"}
    except DatabaseUnavailable:
        rows = _alerts_from_csv()
        return {"alerts": rows, "count": len(rows), "source": _source()}


@app.get("/alerts/{level}")
def get_alerts_by_level(level: str):
    """Alerts filtered by level (CRITICAL/HIGH/MEDIUM/LOW)."""
    level = level.upper()
    if level not in _LEVEL:
        raise HTTPException(status_code=400, detail="Level must be CRITICAL, HIGH, MEDIUM, or LOW")
    try:
        rows = query_db("""SELECT DISTINCT ON (subreddit) subreddit, alert_level, chi, message, timestamp
                           FROM alerts WHERE alert_level = %s ORDER BY subreddit, timestamp DESC""", (level,))
        source = "database"
    except DatabaseUnavailable:
        rows, source = _alerts_from_csv(level), _source()
    return {"alerts": rows, "count": len(rows), "level": level, "source": source}


# ── Forecasts ─────────────────────────────────────────────────
@app.get("/forecast/{subreddit:path}")
def get_forecast(subreddit: str):
    """
    Five-day toxicity forecast for a subreddit.

    From the CSVs this prefers forecast_future.csv — the days after the data —
    over forecast_results.csv, which for this corpus is a hindcast of the last
    observed days (see gen_forecasts.py). The response says which it is.
    """
    subreddit = _r(subreddit)
    try:
        rows = query_db("""SELECT subreddit, forecast_date, predicted_toxicity, risk_level, method, created_at
                           FROM forecasts WHERE subreddit = %s ORDER BY forecast_date ASC""", (subreddit,))
        source, kind = "database", "as loaded"
    except DatabaseUnavailable:
        rows, source = [], _source()
    if not rows:
        fut = DATASET.future_file
        name, kind = (fut.name, "future") if fut else ("forecast_results.csv", DATASET.forecast)
        df = _csv(name)
        # Compare whole names. The old fallback used .lstrip("r/"), which strips
        # the characters r and / from the left — "r/rust" became "ust".
        df = df[df["subreddit"].astype(str).map(_r).str.lower() == subreddit.lower()]
        rows, source = _records(df), _source()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No forecast for {subreddit}")
    return {"subreddit": subreddit, "forecast": rows, "days": len(rows), "kind": kind, "source": source}


# ── Clusters ──────────────────────────────────────────────────
@app.get("/clusters")
def get_clusters():
    """Community cluster assignments."""
    df = _csv("clusters_report.csv")
    col = "cluster_label" if "cluster_label" in df.columns else "cluster"
    if col not in df.columns:
        raise HTTPException(status_code=422, detail="clusters_report.csv has no cluster column")
    clusters = {}
    for label in df[col].dropna().unique():
        sub = df[df[col] == label]
        clusters[str(label)] = {"communities": [_r(s) for s in sub["subreddit"]], "count": len(sub),
                                "avg_chi": round(sub["community_health_index"].mean(), 2),
                                "avg_toxicity": round(sub["toxicity_rate"].mean(), 4)}
    return {"clusters": clusters, "total_communities": len(df), "source": _source()}


# ── Moderation recommendations ────────────────────────────────
def _recommendations() -> pd.DataFrame:
    m = _meso()
    m["recommended_action"] = m["alert_level"].map(ACTION)
    return m[["subreddit", "community_health_index", "alert_level", "recommended_action", "toxicity_rate"]]


@app.get("/recommendations")
def get_recommendations():
    """
    A recommended action per community: a deterministic mapping from the alert
    band (apollo_bands.ACTION). This endpoint used to read
    outputs/moderation_recommendations.csv, which no stage writes, so it
    returned 404 on every call.
    """
    df = _recommendations()
    return {"recommendations": _records(df), "count": len(df), "source": _source()}


@app.get("/recommendations/{subreddit:path}")
def get_recommendation(subreddit: str):
    df = _recommendations()
    row = df[df["subreddit"] == _r(subreddit)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"{_r(subreddit)} not found")
    return _records(row)[0]


# ── Summary ───────────────────────────────────────────────────
@app.get("/summary")
def get_summary():
    """Dashboard summary — key stats at a glance."""
    try:
        alerts = query_db("SELECT alert_level, COUNT(*) as count FROM alerts GROUP BY alert_level")
        health = query_db("""SELECT AVG(community_health_index) as avg_chi, AVG(toxicity_rate) as avg_toxicity,
                                    COUNT(DISTINCT subreddit) as total_communities FROM community_health""")
        tox = query_db("""SELECT COUNT(*) as total, SUM(CASE WHEN is_toxic THEN 1 ELSE 0 END) as toxic_count
                          FROM toxicity_scores""")
        summary = {
            "total_communities": health[0]["total_communities"] if health else 0,
            "avg_chi": round(float(health[0]["avg_chi"] or 0), 2) if health else 0,
            "avg_toxicity": round(float(health[0]["avg_toxicity"] or 0), 4) if health else 0,
            "alert_distribution": {r["alert_level"]: r["count"] for r in alerts},
            "total_comments_analyzed": tox[0]["total"] if tox else 0,
            "toxic_comment_count": tox[0]["toxic_count"] if tox else 0,
        }
        source = "database"
    except DatabaseUnavailable:
        m = _meso()
        summary = {
            "total_communities": int(len(m)),
            "avg_chi": round(float(m["community_health_index"].mean()), 2),
            "avg_toxicity": round(float(m["toxicity_rate"].mean()), 4),
            "alert_distribution": {k: int(v) for k, v in m["alert_level"].value_counts().items()},
            "total_comments_analyzed": int(m["total_comments"].sum()),
        }
        source = _source()
    return {"summary": summary, "dataset": DATASET.key, "source": source,
            "timestamp": datetime.now().isoformat()}


# ── Prometheus ────────────────────────────────────────────────
@app.get("/metrics", response_class=PlainTextResponse,
         summary="Prometheus metrics (text exposition format)")
def metrics():
    """
    APOLLO-M's metrics in Prometheus exposition format, for any Prometheus
    (the hosted stack in monitoring/hosted scrapes this every 30 s).

    Never fails because a data source is missing: this is what reports the
    failure. Hand-formatted rather than pulling in prometheus_client — the
    format is a few lines of text and the deployed image stays small.
    """
    m: list[str] = []

    def add(name, helptext, value, mtype="gauge", labels: dict | None = None, header=True):
        if value is None:
            return
        if header:
            m.append(f"# HELP {name} {helptext}")
            m.append(f"# TYPE {name} {mtype}")
        lab = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}" if labels else ""
        m.append(f"{name}{lab} {value}")

    up = database_up()
    add("apollo_database_up", "1 if PostgreSQL answered, 0 if the API is serving committed outputs", int(up))
    add("apollo_info", "Which corpus and source this API serves", 1,
        labels={"dataset": DATASET.key, "source": "database" if up else "csv", "version": app.version})
    add("apollo_uptime_seconds", "Seconds since this process started", round(time.time() - _STARTED, 1))

    try:
        if up:
            rows = query_db("SELECT community_health_index AS chi, toxicity_rate AS tox FROM community_health")
            chi = [float(r["chi"]) for r in rows if r["chi"] is not None]
            tox = [float(r["tox"]) for r in rows if r["tox"] is not None]
        else:
            mm = _meso()
            chi, tox = mm["community_health_index"].tolist(), mm["toxicity_rate"].tolist()
        bands = [alert_band(v) for v in chi]
        add("apollo_communities_total", "Communities analysed", len(chi))
        add("apollo_critical_alerts", "Communities at CRITICAL alert level", bands.count("CRITICAL"))
        add("apollo_communities_by_band", "Communities in each alert band", bands.count("CRITICAL"),
            labels={"band": "CRITICAL"})
        for b in ("HIGH", "MEDIUM", "LOW"):
            add("apollo_communities_by_band", "", bands.count(b), labels={"band": b}, header=False)
        if chi:
            add("apollo_avg_chi", "Mean Community Health Index (0-100)", round(sum(chi) / len(chi), 4))
            add("apollo_min_chi", "Lowest Community Health Index", round(min(chi), 4))
        if tox:
            add("apollo_avg_toxicity", "Mean toxicity across communities", round(sum(tox) / len(tox), 4))
    except Exception:  # noqa: BLE001 — report what we can
        pass

    try:
        s = json.loads(DATASET.file("forecast_summary.json").read_text(encoding="utf-8"))
        add("apollo_forecast_mae", "Backtest MAE on held-out days", s["mae_tft"], labels={"method": "tft"})
        add("apollo_forecast_mae", "", s["mae_last_value"], labels={"method": "last_value"}, header=False)
        add("apollo_forecast_mae", "", s["mae_recent_mean"], labels={"method": "recent_mean"}, header=False)
        add("apollo_forecast_coverage", "Share of held-out actuals inside the p10-p90 band", s["coverage_p10_p90"])
    except Exception:  # noqa: BLE001
        pass
    try:
        mj = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
        t = mj.get("Toxicity (TF-IDF+LR, Jigsaw)", {})
        add("apollo_toxicity_f1_micro", "Toxicity classifier micro F1", float(t["F1 micro"]))
        add("apollo_toxicity_f1_macro", "Toxicity classifier macro F1", float(t["F1 macro"]))
    except Exception:  # noqa: BLE001
        pass
    try:
        vj = json.loads((OUTPUTS / "validation.json").read_text(encoding="utf-8"))
        add("apollo_forecast_slope_roc_auc", "TFT slope ROC-AUC against planted ground truth",
            vj["forecast_tft"]["slope_roc_auc"])
        add("apollo_instability_roc_auc", "Instability-score ROC-AUC against planted ground truth",
            vj["detection"]["instability_score_roc_auc"])
    except Exception:  # noqa: BLE001
        pass

    # Request counters and latency histogram, per route template.
    m.append("# HELP apollo_api_requests_total API requests by method, route and status")
    m.append("# TYPE apollo_api_requests_total counter")
    for (meth, path, status), n in sorted(_req_count.items()):
        m.append(f'apollo_api_requests_total{{method="{meth}",route="{path}",status="{status}"}} {n}')
    m.append("# HELP apollo_api_request_seconds API request latency")
    m.append("# TYPE apollo_api_request_seconds histogram")
    for path in sorted(_lat_n):
        for b in _BUCKETS:
            m.append(f'apollo_api_request_seconds_bucket{{route="{path}",le="{b}"}} {_lat_bucket[(path, b)]}')
        m.append(f'apollo_api_request_seconds_bucket{{route="{path}",le="+Inf"}} {_lat_n[path]}')
        m.append(f'apollo_api_request_seconds_sum{{route="{path}"}} {round(_lat_sum[path], 6)}')
        m.append(f'apollo_api_request_seconds_count{{route="{path}"}} {_lat_n[path]}')
    return "\n".join(m) + "\n"


# ── Analyze ───────────────────────────────────────────────────
@app.post("/analyze")
def analyze(payload: dict):
    """
    Score submitted text. Body: {"text": "comment text here"}.
    Returns the CEREBRO misinformation signal and a keyword toxicity flag.
    """
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text field required")
    try:
        from modules.cerebro_detector import CEREBRODetector
        misinfo_result = CEREBRODetector().analyze(text)
    except Exception:  # noqa: BLE001
        misinfo_result = {"misinformation_score": 0, "misinfo_category": "UNKNOWN"}
    toxic_words = ["hate", "kill", "stupid", "idiot", "garbage", "attack"]
    return {
        "text": text[:100],
        "misinformation_score": misinfo_result.get("misinformation_score", 0),
        "misinfo_category": misinfo_result.get("misinfo_category", "UNKNOWN"),
        "coordinated_behaviour": misinfo_result.get("coordinated_behaviour", False),
        "keyword_toxicity_flag": any(w in text.lower() for w in toxic_words),
        "timestamp": datetime.now().isoformat(),
    }
