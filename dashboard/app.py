"""APOLLO-M — community-instability console (Streamlit).

Login → corpus + page in the sidebar → views built from dashboard/charts.py.
Every number comes from a committed output directory, chosen through
apollo_datasets, and every page says which corpus it is showing.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import os as _os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUTPUTS = ROOT / "outputs"
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import apollo_datasets as AD  # noqa: E402
import charts as C  # noqa: E402
import data as D  # noqa: E402
import theme as T  # noqa: E402
from apollo_bands import ACTION, COLOR, ICON, ORDER  # noqa: E402

st.set_page_config(page_title="APOLLO-M", page_icon="●", layout="wide")


def _secrets_to_env() -> None:
    """
    Publish Streamlit secrets as environment variables.

    The LLM layer reads os.getenv("ANTHROPIC_API_KEY"), which works locally
    because llm/__init__.py loads the git-ignored .env. A Streamlit Cloud
    deployment has no .env, so the key was never visible and every explanation
    silently degraded to "[template fallback — no LLM reachable]".

    setdefault, not assignment: a real environment variable always wins.
    """
    try:
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "CLAUDE_MODEL", "LLM_PROVIDER", "OLLAMA_HOST",
                    "OLLAMA_MODEL", "APOLLO_API_URL", "GRAFANA_URL", "PROMETHEUS_URL"):
            if key in st.secrets:
                _os.environ.setdefault(key, str(st.secrets[key]))
    except Exception:
        pass


_secrets_to_env()

# Where the deployed services live. Anything with a public URL points at it.
API_URL = _os.getenv("APOLLO_API_URL", "https://apollo-api-tllm.onrender.com").rstrip("/")
CEREBRO_URL = _os.getenv("CEREBRO_URL", "https://cerebro-sandy-beta.vercel.app").rstrip("/")
# Production alias: per-deployment Vercel URLs sit behind deployment protection.
WEB_URL = _os.getenv("APOLLO_WEB_URL", "https://apollo-m.vercel.app").rstrip("/")
# The hosted monitoring stack (monitoring/hosted, deployed from render.yaml).
# Grafana serves the dashboard read-only to anonymous visitors, so these links
# work for anyone — the earlier defaults pointed at localhost, which on any
# other device is the visitor's own machine.
GRAFANA_URL = _os.getenv("GRAFANA_URL", "https://apollo-m-monitoring.onrender.com").rstrip("/")
PROMETHEUS_URL = _os.getenv("PROMETHEUS_URL", f"{GRAFANA_URL}/prometheus/")

st.markdown(T.CSS, unsafe_allow_html=True)


# ── cached data access ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load(kind: str, key: str):
    ds = AD.get(key)
    return {
        "meso": D.meso, "clusters": D.clusters, "daily": D.daily, "future": D.future,
        "held_out": D.held_out, "backtest": D.backtest, "summary": D.backtest_summary,
        "facts": D.corpus_facts, "evidence": D.evidence,
    }[kind](ds)


@st.cache_data(show_spinner=False)
def load_phase(key: str) -> pd.DataFrame:
    return D.phase_space(load("daily", key))


@st.cache_data(show_spinner=False)
def load_corpora() -> pd.DataFrame:
    return D.across_corpora(AD.real())


def show(fig, table: pd.DataFrame | None = None, key: str | None = None):
    """Every chart has a table twin, so no value is reachable only by hovering."""
    st.plotly_chart(fig, use_container_width=True, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})
    if table is not None and not table.empty:
        with st.expander("View as table"):
            st.dataframe(table, use_container_width=True, hide_index=True)


def band_chip(band: str) -> str:
    return (f'<span class="band" style="border-color:{COLOR[band]};background:{T.rgba(COLOR[band], .14)}">'
            f'{ICON[band]} {band}</span>')


safe = T.safe


def finding(html: str):
    st.markdown(f'<div class="finding">{html}</div>', unsafe_allow_html=True)


# ── Authentication ──────────────────────────────────────────────────────────
# Accounts live in Supabase Auth, shared with the public web view (auth.py has
# the protocol, web/supabase/accounts.sql the database rules). There are no
# passwords in this file and no public demo button: a hard-coded "admin /
# apollo_admin" shown on the login page is a door with the key taped to it.
import auth as A  # noqa: E402
import streamlit.components.v1 as _components  # noqa: E402

APP_URL = _os.getenv("APP_URL", "https://apollo-m.streamlit.app").rstrip("/")
DEV_LOGIN = _os.getenv("APOLLO_DEV_LOGIN") == "1"   # local development and tests only
COOKIE = "apollo_session"


def _remember(token: str | None) -> None:
    """Keep (or forget) the refresh token in a first-party cookie for 30 days.
    Streamlit cannot set cookies itself; this tiny component writes it on the
    app's own origin. The token is single-use (Supabase rotates it) and revoked
    on sign-out."""
    if token:
        js = (f"parent.document.cookie='{COOKIE}={token}; Max-Age=2592000; Path=/; "
              "SameSite=Lax; Secure';")
    else:
        js = f"parent.document.cookie='{COOKIE}=; Max-Age=0; Path=/; SameSite=Lax; Secure';"
    _components.html(f"<script>{js}</script>", height=0)


def _start(s: A.Session, how: str) -> None:
    s = A.profile(s)
    st.session_state.session = s
    st.session_state.user = s.email
    st.session_state.remember = st.session_state.get("remember_me", True)
    A.log(s, "sign_in", {"method": how})


st.session_state.setdefault("session", None)
st.session_state.setdefault("user", None)

# 1. Returning from Google (OAuth code + our nonce in the query string).
_q = st.query_params
if not st.session_state.session and _q.get("auth") == "google" and _q.get("code"):
    try:
        _start(A.google_exchange(_q["code"], _q.get("n", "")), "google")
        st.session_state.remember_me = True
    except A.AuthError as exc:
        st.session_state.auth_error = str(exc)
    st.query_params.clear()

# 2. A remembered session: exchange the saved refresh token for a fresh one.
if not st.session_state.session and not st.session_state.get("signed_out"):
    try:
        _saved = (st.context.cookies or {}).get(COOKIE)
    except Exception:  # noqa: BLE001
        _saved = None
    if _saved and A.config():
        try:
            _start(A.refresh(_saved), "remembered")
        except A.AuthError:
            st.session_state.forget_cookie = True

# 3. An active session whose access token has expired: refresh it quietly.
if st.session_state.session:
    try:
        st.session_state.session = A.ensure_fresh(st.session_state.session)
    except A.AuthError:
        st.session_state.session = None
        st.session_state.user = None

if st.session_state.session and st.session_state.get("remember", True):
    _remember(st.session_state.session.refresh_token)
if st.session_state.pop("forget_cookie", False):
    _remember(None)


def _login_page() -> None:
    st.markdown("""<style>
    .stApp .block-container{padding-top:7vh;max-width:1180px}
    section[data-testid="stSidebar"]{display:none}
    div[data-testid="stForm"]{background:#1A1430;border:1px solid rgba(201,191,227,.14);
      border-radius:16px;padding:26px 26px 10px;}
    .auth-side{padding:8px 34px 0 0}
    .auth-points{margin:26px 0 0;padding:0;list-style:none}
    .auth-points li{display:flex;gap:12px;margin:0 0 14px;color:#C9BFE3;font-size:.98rem;line-height:1.45}
    .auth-points b{color:#F4F1EC;font-weight:500}
    .auth-points span.k{flex:0 0 26px;color:#A58CF0;font-family:'JetBrains Mono',monospace;font-size:.8rem;padding-top:3px}
    .google-btn{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;
      padding:11px 0;border-radius:10px;border:1px solid rgba(201,191,227,.25);background:#F4F1EC;
      color:#110C1F !important;font-weight:600;text-decoration:none !important;margin:4px 0 6px}
    .google-btn:hover{background:#fff}
    .or{display:flex;align-items:center;gap:12px;color:#7c83b3;font-size:.8rem;margin:14px 0 10px}
    .or:before,.or:after{content:"";flex:1;height:1px;background:rgba(201,191,227,.16)}
    /* Inputs inside the card: a darker well with a visible edge, or they vanish
       into the card (same colour as the theme's secondary background). */
    div[data-testid="stForm"] div[data-testid="stTextInputRootElement"]{background:#110C1F !important;
      border:1px solid rgba(201,191,227,.26) !important;border-radius:10px !important}
    div[data-testid="stForm"] div[data-testid="stTextInputRootElement"]:focus-within{
      border-color:#A58CF0 !important}
    div[data-testid="stForm"] input{background:transparent !important;color:#F4F1EC !important}
    .auth-side .hook{margin-bottom:14px}
    </style>""", unsafe_allow_html=True)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        _real = [load("meso", d.key) for d in AD.real()]
        st.markdown('<div class="auth-side">' + T.status_strip([
            ("real comments read", f"{int(sum(m['total_comments'].sum() for m in _real)):,}"),
            ("communities", str(len(set().union(*[set(m["subreddit"]) for m in _real]))))]) +
            '<div class="hook" style="margin-top:22px">Moderation reacts to the worst day.<br>'
            'APOLLO-M watches the five before it.</div>'
            "<p class='apollo-sub'>Community health, early warning and forecasts for Reddit "
            "communities, measured on real comments and checked against what happened next.</p>"
            '<ul class="auth-points">'
            '<li><span class="k">01</span><span><b>One health score per community</b>, and the four '
            'measurements behind it.</span></li>'
            '<li><span class="k">02</span><span><b>Live.</b> New comments are fetched and scored every '
            'six hours; any topic can be searched now.</span></li>'
            '<li><span class="k">03</span><span><b>Honest forecasts.</b> Tested on days that had not '
            'happened yet, and reported when a simple baseline wins.</span></li>'
            "</ul></div>", unsafe_allow_html=True)
    with right:
        if err := st.session_state.pop("auth_error", None):
            st.error(err)
        if not A.config():
            st.warning("Accounts are not configured on this deployment (SUPABASE_URL and "
                       "SUPABASE_PUBLISHABLE_KEY are missing).")
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in:
            if A.config() and A.google_enabled():
                try:
                    st.markdown(f'<a class="google-btn" href="{A.google_url(APP_URL)}" target="_self">'
                                '<b>G</b>&nbsp; Continue with Google</a>', unsafe_allow_html=True)
                except A.AuthError:
                    pass
                st.markdown('<div class="or">or with email</div>', unsafe_allow_html=True)
            with st.form("sign_in"):
                email = st.text_input("Email", autocomplete="email")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                st.checkbox("Keep me signed in on this device", value=True, key="remember_me")
                if st.form_submit_button("Sign in", use_container_width=True, type="primary"):
                    try:
                        _start(A.sign_in(email, password), "password")
                        st.session_state.signed_out = False
                        st.rerun()
                    except A.AuthError as exc:
                        st.error(str(exc))
        with tab_up:
            with st.form("sign_up"):
                name = st.text_input("Full name", autocomplete="name")
                email2 = st.text_input("Work email", autocomplete="email")
                pw2 = st.text_input("Password", type="password", autocomplete="new-password",
                                    help=f"At least {A.MIN_PASSWORD} characters, not only letters or only numbers.")
                st.caption("Your activity in APOLLO-M (sign-ins, pages opened, searches) is recorded "
                           "against your account and visible to your organisation's administrators.")
                if st.form_submit_button("Create account", use_container_width=True, type="primary"):
                    try:
                        out = A.sign_up(email2, pw2, name)
                        if isinstance(out, str):
                            st.success(out)
                        else:
                            _start(out, "sign_up")
                            st.rerun()
                    except A.AuthError as exc:
                        st.error(str(exc))
        if DEV_LOGIN and st.button("Developer sign-in (local only)", use_container_width=True):
            st.session_state.session = A.Session("dev", "", 9e12, "dev", "developer@localhost",
                                                 "Developer", "dev", role="admin")
            st.session_state.user = "developer@localhost"
            st.rerun()


if not st.session_state.session:
    _login_page()
    st.stop()

SESSION: A.Session = st.session_state.session
IS_ADMIN = SESSION.role == "admin"


def track(action: str, **detail) -> None:
    """One activity row for this user (best effort; skipped for the dev session)."""
    if SESSION.provider != "dev":
        A.log(SESSION, action, detail)


# ── Sidebar: corpus, page, connected systems ────────────────────────────────
AVAILABLE = AD.available()
KEYS = [d.key for d in AVAILABLE]
LABELS = {d.key: d.label for d in AD.DATASETS.values()}

with st.sidebar:
    st.markdown('<div class="apollo-title" style="font-size:1.55rem"><span class="mark">●</span> APOLLO-M</div>',
                unsafe_allow_html=True)
    st.caption(f"Signed in as **{safe(SESSION.name or SESSION.email)}**"
               + (" · admin" if IS_ADMIN else "") + f"  \n{safe(SESSION.email)}")
    key = st.selectbox("Corpus", KEYS, index=KEYS.index(AD.DEFAULT) if AD.DEFAULT in KEYS else 0,
                       format_func=lambda k: LABELS[k],
                       help="Four corpora, one pipeline. The benchmark is a declared simulation with "
                            "recorded ground truth; the other three are real Reddit comments.")
    page = st.radio("Navigate", ["Overview", "Dynamics", "Communities", "Forecast", "🎯 Actions", "Clusters",
                                 "Across corpora", "Evidence", "🔴 Live", "📡 Topic watch", "Monitoring"]
                    + (["🛡 Admin"] if IS_ADMIN else []))
    provider = st.radio("LLM provider", ["claude", "ollama"], horizontal=True,
                        help="Claude = cloud API. Ollama = local and free, needs `ollama serve`.")
    st.divider()
    st.markdown("**Connected systems**")
    st.link_button("🌍 Public web view", WEB_URL, use_container_width=True)
    st.link_button("🛰️ Open CEREBRO", CEREBRO_URL, use_container_width=True)
    st.link_button("📊 Grafana", GRAFANA_URL, use_container_width=True)
    st.link_button("🔌 API docs", f"{API_URL}/docs", use_container_width=True)
    st.divider()
    if st.button("Sign out", use_container_width=True, key="sidebar_signout"):
        track("sign_out")
        A.sign_out(SESSION)          # revokes the refresh token on the server
        _remember(None)              # and forgets it in this browser
        st.session_state.session = None
        st.session_state.user = None
        st.session_state.signed_out = True
        st.rerun()

# One activity row per page opened (and per corpus switched), not per rerun.
if (page, key) != st.session_state.get("_last_view"):
    st.session_state._last_view = (page, key)
    track("page_view", page=page.split(" ", 1)[-1] if page[0] in "🎯🔴📡🛡" else page, corpus=key)

DS = AD.get(key)
meso = load("meso", key)
daily = load("daily", key)
future = load("future", key)
held = load("held_out", key)
FC = future if not future.empty else held          # what the forecast pages draw
BANDS = dict(zip(meso["subreddit"], meso["band"])) if not meso.empty else {}


def provenance():
    """Say which corpus every number below comes from, before any chart."""
    f = load("facts", key)
    cls = "prov bench" if DS.kind == "benchmark" else "prov"
    kind = "Declared simulation" if DS.kind == "benchmark" else "Real Reddit"
    st.markdown(
        f'<div class="{cls}"><b>{DS.label}</b> &nbsp;<span class="chip">{kind}</span>'
        f'<span class="chip">{DS.window}</span><span class="chip">{f["communities"]} communities</span>'
        f'<span class="chip">{f["comments"]:,} comments</span><span class="chip">by {DS.credit}</span>'
        f'<div class="meta">{DS.summary}</div><div class="meta">Source: {DS.source}</div></div>',
        unsafe_allow_html=True)


def header(title: str, sub: str = "", strip: bool = True):
    """Page title in Fraunces, under the mono status strip for the selected corpus.
    Pages that are not about the selected corpus (Live, Topic watch) pass strip=False,
    so a live page never shows the August corpus's numbers above it."""
    import time as _t
    if strip and not meso.empty:
        f = load("facts", key)
        attention = int(meso["band"].isin(["CRITICAL", "HIGH"]).sum())
        st.markdown(T.status_strip([
            ("UTC", _t.strftime("%H:%M", _t.gmtime())), ("comments read", f"{f['comments']:,}"),
            ("communities", str(f["communities"])), ("need attention", str(attention)),
            ("window", DS.window)]), unsafe_allow_html=True)
    st.markdown(f'<div class="apollo-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="apollo-sub">{sub}</div>', unsafe_allow_html=True)
    st.write("")


def fc_p50_last(sub: str):
    g = FC[FC["subreddit"] == sub] if not FC.empty else FC
    return round(float(g.sort_values("day")["p50"].iloc[-1]), 3) if not g.empty else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# Pages
# ═════════════════════════════════════════════════════════════════════════════

if page == "Overview":
    header("Community health", "One number per community, and the four measurements behind it.")
    provenance()
    if meso.empty:
        st.info("No outputs for this corpus.")
        st.stop()
    n_crit = int((meso["band"] == "CRITICAL").sum())
    n_high = int((meso["band"] == "HIGH").sum())
    f = load("facts", key)
    worst = meso.iloc[0]
    st.markdown(T.kpi_row([
        T.kpi("Communities", int(len(meso)), f"{f['days']} days observed", count=True),
        T.kpi("Comments analysed", int(f["comments"]), "scored by toxic-bert", count=True),
        T.kpi("Critical + high", n_crit + n_high, f"{n_crit} critical · {n_high} high", count=True),
        T.kpi("Median CHI", f"{meso['community_health_index'].median():.1f}",
              f"range {meso['community_health_index'].min():.1f} – {meso['community_health_index'].max():.1f}"),
        T.kpi("Lowest health", worst["subreddit"], f"CHI {worst['community_health_index']:.1f} · {worst['band']}"),
    ]), unsafe_allow_html=True)
    st.write("")
    show(C.band_strip(meso), key="ov_strip")
    show(C.landscape(meso), meso[["subreddit", "community_health_index", "band", "toxicity_rate",
                                  "polarization", "churn_rate", "echo_chamber_index", "total_comments"]],
         key="ov_land")
    st.markdown("#### Ranking")
    tbl = meso[["subreddit", "community_health_index", "band", "toxicity_rate", "polarization",
                "total_comments"]].copy()
    tbl["band"] = tbl["band"].map(lambda b: f"{ICON[b]} {b}")
    tbl["forecast p50 (day 5)"] = tbl["subreddit"].map(fc_p50_last)
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=420, column_config={
        "community_health_index": st.column_config.ProgressColumn("CHI", min_value=0, max_value=100, format="%.1f"),
        "toxicity_rate": st.column_config.ProgressColumn("toxicity", min_value=0,
                                                         max_value=float(max(.01, meso["toxicity_rate"].max())),
                                                         format="%.3f"),
        "polarization": st.column_config.NumberColumn("polarisation", format="%.3f"),
        "total_comments": st.column_config.NumberColumn("comments", format="%d"),
    })
    st.caption("Bands: ✕ CRITICAL < 65 ≤ ▲ HIGH < 75 ≤ ◆ MEDIUM < 85 ≤ ● LOW — one rule, shared by "
               "this console, the API, the exporter and the public web view (apollo_bands.py).")

elif page == "Dynamics":
    header("Dynamics", "Where each community was heading, day by day. Press play.")
    provenance()
    if daily.empty:
        st.info("No daily series for this corpus.")
        st.stop()
    ps = load_phase(key)
    show(C.phase_space(ps, BANDS), ps.assign(date=ps["date"].dt.date), key="dyn_phase")
    finding("<b>Reading it:</b> each point is a community. Its x-position is how toxic it has been over "
            "the last five days, its y-position whether that is rising or falling. Press <b>▶ Play</b> — a "
            "community drifting up and to the right is the early-warning case the forecaster exists for.")
    show(C.heatmap(daily, meso["subreddit"].tolist()),
         daily.pivot_table(index="subreddit", columns=daily["date"].dt.strftime("%m-%d"),
                           values="avg_toxicity").round(3).reset_index(), key="dyn_heat")

elif page == "Communities":
    header("One community", "The score, what it is made of, and what happened every day.")
    provenance()
    if meso.empty:
        st.stop()
    sel = st.selectbox("Community", meso["subreddit"].tolist(),
                       format_func=lambda s: f"{ICON[BANDS[s]]} {s}  ·  CHI "
                                             f"{meso.set_index('subreddit').loc[s, 'community_health_index']:.1f}")
    r = meso[meso["subreddit"] == sel].iloc[0]
    chi = float(r["community_health_index"])
    a, b = st.columns([1, 1.4])
    with a:
        show(C.gauge(chi, r["band"]), key="cm_gauge")
        st.markdown(f"Recommended action: **{ACTION[r['band']].replace('_', ' ').title()}** &nbsp; "
                    f"{band_chip(r['band'])}", unsafe_allow_html=True)
        st.markdown(f"**{int(r['total_comments']):,}** comments analysed")
        if "insufficient_history" in r and bool(r.get("insufficient_history")):
            st.warning("Fewer than 7 days of data — churn and trend here mean *not enough data*, not *healthy*.")
    with b:
        steps = D.chi_waterfall(r)
        show(C.waterfall(steps), steps, key="cm_water")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        show(C.profile(r, meso.median(numeric_only=True), D.COMPONENTS), key="cm_profile")
    with c2:
        dsub = daily[daily["subreddit"] == sel]
        show(C.history(dsub, future[future["subreddit"] == sel] if not future.empty else None,
                       held[held["subreddit"] == sel] if not held.empty else None,
                       title=f"{sel} — daily toxicity and volume"),
             dsub[["date", "avg_toxicity", "comment_count"]], key="cm_hist")
    if st.button(f"🧠 Explain {sel} with {provider}", type="primary"):
        track("explain", community=sel, provider=provider)
        with st.spinner(f"Asking {provider}…"):
            from llm.explain import explain_community
            txt = explain_community({
                "subreddit": sel, "chi": round(chi, 1),
                "toxicity": f"{r['toxicity_rate']:.0%}", "polarization": f"{r['polarization']:.0%}",
                "alert_level": r["band"], "forecast_p50": D.nan_to_none(fc_p50_last(sel)),
            }, provider_name=provider)
        st.success(txt)
        if "template fallback" in txt and provider == "ollama":
            st.caption("Ollama isn't running — start it (`ollama serve`) or switch the provider to **claude**.")

elif page == "Forecast":
    header("Five-day forecast", "What the model expects next — and how often it has been right.")
    provenance()
    if FC.empty:
        st.info("No forecast for this corpus.")
        st.stop()
    if not future.empty:
        st.markdown(f"These are forecasts of the **five days after each community's last observation**. "
                    f"Below them, the same model is scored on days it never trained on.")
    else:
        finding("<b>Held-out evaluation, not a forecast of the future.</b> RC_2026-08 trains on all but the "
                "last five days of August and predicts those five, so each prediction can be shown next to "
                "what actually happened. No forward (September) forecast was made for this corpus.")
    subs = sorted(FC["subreddit"].unique(), key=lambda s: meso.set_index("subreddit")
                  .reindex([s])["community_health_index"].fillna(100).iloc[0])
    sel = st.selectbox("Community", subs)
    dsub = daily[daily["subreddit"] == sel]
    fsub = FC[FC["subreddit"] == sel]
    show(C.history(dsub.tail(35), fsub if not future.empty else None, fsub if future.empty else None,
                   title=f"{sel}"),
         fsub[[c for c in ("date", "p10", "p50", "p90", "actual") if c in fsub]], key="fc_hist")
    trend = float(fsub.sort_values("day")["p50"].iloc[-1] - fsub.sort_values("day")["p50"].iloc[0])
    arrow = "▲ rising" if trend > .005 else "▼ falling" if trend < -.005 else "→ flat"
    st.markdown(f"**Five-day direction: {arrow}** ({trend:+.3f}) · mean band width "
                f"{(fsub['p90'] - fsub['p10']).mean():.3f}")

    st.markdown("#### How good is it?")
    s = load("summary", key)
    if s:
        better = "beats" if s["mae_tft"] < s["mae_recent_mean"] else "does not beat"
        st.markdown(T.kpi_row([
            T.kpi("TFT error (MAE)", f"{s['mae_tft']:.4f}", f"{s['days']} held-out days"),
            T.kpi("Repeat last value", f"{s['mae_last_value']:.4f}", f"TFT better in {s['beats_last']}/{s['communities']}"),
            T.kpi("14-day mean", f"{s['mae_recent_mean']:.4f}", f"TFT better in {s['beats_mean']}/{s['communities']}"),
            T.kpi("Band coverage", f"{s['coverage']:.0%}", "of actuals inside p10–p90 (80% nominal)"),
        ]), unsafe_allow_html=True)
        finding(f"On this corpus the TFT <b>{better}</b> simply repeating the recent 14-day mean. "
                + ("That is the honest headline: on real Reddit, daily community toxicity is close to "
                   "mean-reverting over five days, and every corpus the team tested agrees."
                   if better == "does not beat" else
                   "This is the benchmark, where trends were planted deliberately — the case a "
                   "trend-aware model is built for."))
        bt = load("backtest", key)
        show(C.backtest_dots(bt), bt.round(4), key="fc_dots")
    show(C.backtest_across({k: load("summary", k) for k in KEYS}, LABELS),
         pd.DataFrame({LABELS[k]: load("summary", k) for k in KEYS}).T.round(4).reset_index()
         .rename(columns={"index": "corpus"}), key="fc_across")
    st.markdown("#### Highest-risk communities")
    show(C.small_multiples(daily, FC, subs[:6]), key="fc_sm")

elif page == "🎯 Actions":
    header("Actions", "Every community gets the same rule. No moderator has to guess.")
    provenance()
    if meso.empty:
        st.stop()
    counts = meso["band"].value_counts()
    st.markdown(T.kpi_row([T.kpi(f"{ICON[b]} {ACTION[b].replace('_', ' ').title()}", int(counts.get(b, 0)), b, count=True)
                           for b in ORDER]), unsafe_allow_html=True)
    st.write("")
    v = meso[["subreddit", "community_health_index", "band", "action", "toxicity_rate"]].copy()
    v["forecast p50 (day 5)"] = v["subreddit"].map(fc_p50_last)
    v["band"] = v["band"].map(lambda b: f"{ICON[b]} {b}")
    st.dataframe(v, use_container_width=True, hide_index=True, height=480, column_config={
        "community_health_index": st.column_config.ProgressColumn("CHI", min_value=0, max_value=100, format="%.1f")})
    st.caption("Actions are a deterministic mapping from the band (apollo_bands.ACTION), not a model "
               "prediction. The forecast column is the proactive signal.")

elif page == "Clusters":
    header("Clusters", "Communities grouped by how they are unhealthy, not by topic.")
    provenance()
    cl = load("clusters", key)
    if cl.empty:
        st.stop()
    show(C.clusters_3d(cl), cl[["subreddit", "cluster", "pca_x", "pca_y", "toxicity_rate",
                                "community_health_index"]].round(3), key="cl_3d")
    prof = cl.groupby("cluster")[["community_health_index", "toxicity_rate", "polarization", "churn_rate",
                                  "echo_chamber_index"]].mean().round(3)
    prof.insert(0, "communities", cl.groupby("cluster").size())
    st.markdown("**What each cluster has in common**")
    st.dataframe(prof, use_container_width=True)
    if "is_outlier" in cl:
        st.caption(f"DBSCAN marks {int(cl['is_outlier'].sum())} of {len(cl)} as outliers. With eps = 0.5 almost "
                   "every point is noise — a parameter issue the team has documented, not a finding.")

elif page == "Across corpora":
    header("Across corpora", "Measure the same communities again, months later. Do they rank the same?")
    t = load_corpora()
    real = [d.key for d in AD.real()]
    pairs = [(real[i], real[j]) for i in range(len(real)) for j in range(i + 1, len(real))]
    tiles = []
    for a, b in pairs:
        rho, n = D.rank_agreement(t, a, b)
        if n >= 3:
            # ρ goes in the value: labels are upper-cased, and an upper-case ρ
            # renders as "P".
            tiles.append(T.kpi(f"rank agreement · {n} shared", f"ρ {rho:.3f}",
                               f"{LABELS[a].split('·')[-1].strip()} vs {LABELS[b].split('·')[-1].strip()}"))
    st.markdown(T.kpi_row(tiles), unsafe_allow_html=True)
    finding("<b>Spearman rank correlation of community toxicity between corpora.</b> Three collections — an "
            "API sample in July, the complete August dump, and a rebuilt window in October–November — rank "
            "the communities they share almost identically. Which communities are toxic is a stable property, "
            "not an artifact of one sample.")
    show(C.corpora_dots(t, real, LABELS), t.pivot_table(index="subreddit", columns="label", values="toxicity")
         .round(3).reset_index(), key="ac_dots")

elif page == "Evidence":
    header("Evidence", "What we measured, how, and what it showed. Including what did not work.")
    m = D.metrics()
    st.markdown("### Supervised toxicity classifier (TF-IDF + logistic regression)")
    st.caption("Trained and tested on labelled Jigsaw/Davidson data. A separate model from toxic-bert, which "
               "scores the Reddit corpora — its agreement with labels on real comments is the next section.")
    tox = m.get("Toxicity (TF-IDF+LR, Jigsaw)", {})
    st.markdown(T.kpi_row([T.kpi("Accuracy", tox.get("accuracy", "—"), "Jigsaw held-out"),
                           T.kpi("F1 macro", tox.get("F1 macro", "—")), T.kpi("F1 micro", tox.get("F1 micro", "—")),
                           T.kpi("Precision", tox.get("precision", "—")), T.kpi("Recall", tox.get("recall", "—"))]),
                unsafe_allow_html=True)

    ev = load("evidence", "real_torrent_2026_08")
    la = ev.get("label_agreement", {})
    if la:
        st.markdown("### Does toxic-bert agree with independent labels on real comments?")
        t6 = la.get("threshold_0.6", {})
        st.markdown(T.kpi_row([
            T.kpi("Precision @ 0.6", f"{t6.get('precision', 0):.2f}",
                  "95% CI {:.2f}–{:.2f}".format(*t6.get("ci95", {}).get("precision", [0, 0]))),
            T.kpi("Recall @ 0.6", f"{t6.get('recall', 0):.2f}",
                  "95% CI {:.2f}–{:.2f}".format(*t6.get("ci95", {}).get("recall", [0, 0]))),
            T.kpi("F1 @ 0.6", f"{t6.get('f1', 0):.2f}"),
            T.kpi("Labelled sample", la.get("n_used", 0), f"{la.get('n_unclear', 0)} unclear excluded"),
        ]), unsafe_allow_html=True)
        prof_fp = la.get("profanity_share_by_cell_at_0.6", {}).get("fp", {})
        finding(f"<b>The most important caveat in the project.</b> On 291 RC_2026-08 comments, most comments "
                f"toxic-bert calls toxic are not toxic by the labelling guidelines — and "
                f"{prof_fp.get('with_profanity', 0)} of its {prof_fp.get('n', 0)} false positives contain "
                f"profanity. It largely detects <i>swearing</i>. Labels come from a single LLM labeller, so this is "
                f"agreement, not accuracy against humans. Community <i>rankings</i> are still consistent "
                f"across corpora (see Across corpora).")
    imp = ev.get("improvement", {})
    if imp:
        st.markdown("### Pre-registered attempt to improve the scorer")
        rows = []
        for v, d in imp.get("variants", {}).items():
            diff = d.get("diff_vs_V0", {})
            rows.append({"variant": v, "F1 (weighted)": round(d["point"].get("f1_weighted", float("nan")), 3),
                         "AP (weighted)": round(d["point"].get("ap_weighted", float("nan")), 3),
                         "ΔF1 vs V0": round(diff.get("f1_weighted_point", 0), 3) if diff else float("nan"),
                         "met the pre-registered bar": "yes" if d.get("improvement") else "no"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Thresholds chosen on a separate 300-comment dev set, tested once on the existing 300. "
                   f"Any variant improved: **{'yes' if imp.get('any_improvement') else 'no'}**. "
                   "Recorded as a negative result rather than tuned until something passed.")
    exp = ev.get("expectations", pd.DataFrame())
    if not exp.empty:
        st.markdown("### Expectations fixed before the RC_2026-08 results existed")
        e = exp.copy()
        e["match"] = e["match"].map(lambda x: "✓ match" if x == "match" else "✗ " + str(x))
        st.dataframe(e, use_container_width=True, hide_index=True)

    st.markdown("### Benchmark: recovering planted instability")
    v = load("evidence", "benchmark").get("validation", {})
    if v:
        det, fct = v.get("detection", {}), v.get("forecast_tft", {})
        st.markdown(T.kpi_row([
            T.kpi("Instability score ROC-AUC", det.get("instability_score_roc_auc", "—")),
            T.kpi("CHI ROC-AUC", det.get("chi_roc_auc", "—"), "the composite index"),
            T.kpi("Toxicity alone ROC-AUC", det.get("toxicity_only_roc_auc", "—"), "a single input"),
            T.kpi("TFT slope ROC-AUC", fct.get("slope_roc_auc", "—")),
        ]), unsafe_allow_html=True)
        finding("<b>The published negative result.</b> On 60 communities with 15 planted destabilising "
                "trajectories, the composite CHI separates them <i>worse</i> than raw toxicity on its own "
                f"(ROC-AUC {det.get('chi_roc_auc')} vs {det.get('toxicity_only_roc_auc')}). "
                + str(v.get("caveat", "")))

    st.markdown("### Forecaster against baselines, every corpus")
    show(C.backtest_across({k: load("summary", k) for k in KEYS}, LABELS), key="ev_across")
    st.markdown("### Model status")
    st.json({k: v for k, v in m.items() if k != "Toxicity (TF-IDF+LR, Jigsaw)"}, expanded=False)


# ── Live replay ─────────────────────────────────────────────────────────────
@st.cache_data
def _live_seed() -> pd.DataFrame:
    try:
        return pd.read_csv(OUTPUTS / "live_seed.csv")
    except Exception:
        return pd.DataFrame()


def _hosted_stream():
    """
    Drive the live feed from the wall clock instead of a background process,
    so every visitor sees it advance with no server-side state. Comments and
    scores are the real scored corpus; only the arrival timing is simulated.
    """
    import time as _time
    seed = _live_seed()
    if seed.empty:
        return None, None, None
    step = 2
    pos = int(_time.time() // step) % len(seed)
    window = seed.iloc[max(0, pos - 250):pos + 1]
    if window.empty:
        window = seed.iloc[:1]
    recent = window.tail(8).iloc[::-1].copy()
    recent["ts"] = [_time.strftime("%H:%M:%S", _time.localtime(_time.time() - i * step)) for i in range(len(recent))]
    trail = seed.iloc[max(0, pos - 400):pos + 1]["is_toxic"].astype(float)
    curve = trail.rolling(60, min_periods=10).mean().dropna()
    stats = {
        "processed": pos + 1,
        "rolling_toxic_rate": float(window["is_toxic"].mean()),
        "rolling_avg_toxicity": float(window["toxicity"].mean()),
        "hottest_community": (window.groupby("subreddit")["toxicity"].mean().idxmax() if len(window) else "—"),
        "updated": _time.strftime("%H:%M:%S"),
    }
    return stats, recent, (list(range(pos + 1 - len(curve), pos + 1)), curve.tolist())


@st.fragment(run_every="2s")
def live_panel():
    sp = OUTPUTS / "live_stats.json"
    hosted = not sp.exists()
    curve = None
    if not hosted:
        s, feed = json.loads(sp.read_text()), None
    else:
        s, feed, curve = _hosted_stream()
        if s is None:
            st.info("Live feed unavailable — outputs/live_seed.csv is missing.")
            return
    st.markdown(T.kpi_row([T.kpi("Processed", f"{s.get('processed', 0):,}"),
                           T.kpi("Rolling toxic rate", f"{s.get('rolling_toxic_rate', 0):.0%}"),
                           T.kpi("Rolling avg toxicity", f"{s.get('rolling_avg_toxicity', 0):.2f}"),
                           T.kpi("Hottest community", str(s.get("hottest_community") or "—"))]),
                unsafe_allow_html=True)
    if curve and len(curve[0]) > 2:
        # A stable key updates the chart in place; a per-tick key remounted it
        # every two seconds, so it spent most of its life blank.
        st.plotly_chart(C.live_line(*curve), use_container_width=True, key="live_curve")
    try:
        rows = feed if feed is not None else pd.read_csv(OUTPUTS / "live_feed.csv").tail(8).iloc[::-1]
        reveal = st.session_state.get("live_reveal", False)
        for _, r in rows.iterrows():
            toxic = bool(r["is_toxic"])
            col = COLOR["CRITICAL"] if toxic else COLOR["LOW"]
            mark = "✕" if toxic else "●"
            sub = str(r["subreddit"])
            sub = sub if sub.startswith("r/") else f"r/{sub}"
            # Real Reddit carries real slurs. A comment the model flags is shown
            # as its score unless the viewer asks to see it.
            body = (safe(str(r["body"])[:140]) if (reveal or not toxic)
                    else "<i style='color:#898781'>text hidden — flagged toxic</i>")
            st.markdown(f"<p class='feed'><span style='color:{col}'>{mark}</span> <b>{safe(sub)}</b> "
                        f"<span class='t'>{r['ts']} · tox {float(r['toxicity']):.2f}</span><br>"
                        f"{body}</p>", unsafe_allow_html=True)
    except Exception:
        pass
    st.caption(f"updated {s.get('updated', '')} · " + (LIVE_NOTE if hosted else "real corpus, real scores, live"))


def _live_note() -> str:
    try:
        p = json.loads((OUTPUTS / "live_seed.provenance.json").read_text())
        a, b = (x[:10] for x in p["posted_between"])
        return (f"replaying {p['comments']:,} real Reddit comments from {p['communities_with_comments']} "
                f"communities, posted {a} → {b}, scored by {p['model']} — the comments and scores are real, "
                f"only the arrival timing is simulated. Authors are not stored.")
    except Exception:
        return "replaying scored Reddit comments — arrival timing is simulated"


LIVE_NOTE = _live_note()

if page == "🔴 Live":
    header("Live", "Measured every six hours by a scheduled job — no one runs a script.", strip=False)
    LIVE = OUTPUTS / "live"
    status = json.loads((LIVE / "status.json").read_text()) if (LIVE / "status.json").exists() else None
    if not status:
        st.info("The scheduled live ingest has not published yet (outputs/live/status.json).")
    else:
        import datetime as _dt
        ran = _dt.datetime.fromisoformat(status["run_at"])
        age_min = (_dt.datetime.now(_dt.timezone.utc) - ran).total_seconds() / 60
        ago = f"{age_min:.0f} min" if age_min < 90 else f"{age_min / 60:.1f} h"
        st.markdown(T.kpi_row([
            T.kpi("Last measured", f"{ago} ago", ran.strftime("%d %b %H:%M UTC")),
            T.kpi("Communities", status["communities_with_data"], f"of {status['communities_tracked']} tracked"),
            T.kpi("Comments this run", f"{int(status['comments_this_run']):,}",
                  f"last {status['window_hours']:g} h, scored by toxic-bert"),
            T.kpi("Comments to date", f"{int(status['comments_total']):,}",
                  f"{status['runs_total']} run(s) since {str(status['first_run'])[:10]}"),
            T.kpi("Freshness", f"{status['newest_comment_age_min']:.0f} min"
                  if status.get("newest_comment_age_min") is not None else "—",
                  "newest comment vs the run"),
        ]), unsafe_allow_html=True)
        st.write("")
        lm = pd.read_csv(LIVE / "meso_report.csv") if (LIVE / "meso_report.csv").exists() else pd.DataFrame()
        if not lm.empty:
            lm = lm.rename(columns={"alert": "band"})
            show(C.band_strip(lm), key="live_strip")
            show(C.landscape(lm.assign(volume=lm["total_comments"])),
                 lm[["subreddit", "community_health_index", "band", "toxicity_rate", "churn_rate",
                     "total_comments", "last_seen"]], key="live_land")
            finding(f"<b>Rolling {status['chi_window_days']}-day health, recomputed every run</b> with the "
                    "pipeline's own formula (apollo_chi): toxicity and author churn from the live comments, "
                    "polarisation and echo from each community's place in the SNAP hyperlink graph. "
                    f"Forecast: {safe(status['forecast_method'])}.")
        if status.get("slices_failed"):
            st.caption(f"{status['slices_failed']} fetch slices failed on the last run (Arctic Shift "
                       "throttling); those communities keep their previous values until the next run.")
    st.caption("Adding a community: one line in live/communities.txt. Schedule: "
               ".github/workflows/live-ingest.yml (every 6 hours, and on demand).")
    with st.expander("Replay demo — real scored comments, simulated arrival timing"):
        st.toggle("Show text of comments flagged toxic", key="live_reveal",
                  help="Off by default: real comments include slurs. Scores are always shown.")
        live_panel()

elif page == "📡 Topic watch":
    import apollo_live as AL
    import charts_live as CL
    header("Topic watch", "Ask Reddit a question now. Each search goes straight to Arctic Shift, "
                          "which mirrors Reddit within minutes.", strip=False)
    c1, c2, c3, c4 = st.columns([2.2, 1.6, 1, 1])
    q = c1.text_input("Keyword or phrase", value="remote", placeholder="e.g. remote hiring, layoffs, nvidia")
    preset = c2.selectbox("Where to look", list(AL.TOPICS), index=0)
    hours = c3.selectbox("In the last", [1, 6, 24, 72, 168], index=2,
                         format_func=lambda h: f"{h} h" if h < 48 else f"{h // 24} days")
    kind = c4.selectbox("Search", ["posts", "comments"], index=0)
    subs = st.multiselect("Communities searched (edit freely)", AL.TOPICS[preset] + ["AskReddit", "news"],
                          default=AL.TOPICS[preset])
    if st.button("📡 Search now", type="primary", use_container_width=True) and q.strip() and subs:
        with st.spinner(f"asking Arctic Shift about “{q}” in {len(subs)} communities…"):
            res = AL.topic_watch(q.strip(), subs, hours=hours, kind=kind)
        st.session_state["topic_res"] = res
        track("topic_search", query=q.strip()[:120], communities=len(subs), hours=hours,
              kind=kind, matches=len(res["rows"]))
    res = st.session_state.get("topic_res")
    if res:
        rows = pd.DataFrame(res["rows"])
        n = len(rows)
        st.markdown(T.kpi_row([
            T.kpi("Matches", f"{n:,}", f"{res['kind']} mentioning “{safe(res['query'])}”"),
            T.kpi("Communities", int(rows["subreddit"].nunique()) if n else 0,
                  f"of {len(res['searched'])} searched"),
            T.kpi("Newest", f"{res['newest_age_min']:.0f} min ago" if res["newest_age_min"] is not None else "—",
                  "Arctic Shift lag included"),
            T.kpi("Window", f"{res['hours']} h", "asked just now"),
        ]), unsafe_allow_html=True)
        if res["failed"]:
            st.caption("Not answered this time (throttled — try again in a minute): "
                       + ", ".join(f"{k} ({v})" for k, v in res["failed"].items()))
        if n:
            show(CL.where(rows), rows.groupby("subreddit").size().rename("matches").reset_index(), key="tw_where")
            show(CL.when(rows, res["hours"]), None, key="tw_when")
            st.markdown("#### The matches, newest first")
            for _, r in rows.head(40).iterrows():
                extra = (f" · {int(r['comments'])} replies" if pd.notna(r.get("comments")) else "")
                st.markdown(
                    f"<div class='feed'><span class='chip'>{r['time']:%d %b %H:%M} UTC</span> "
                    f"<b>{safe(r['subreddit'])}</b>{extra} — "
                    f"<a href='{safe(r['link'] or '#')}' target='_blank'>{safe(str(r['text'])[:180])}</a></div>",
                    unsafe_allow_html=True)
        else:
            st.info("Nothing matched in that window. Widen the window or add communities.")
    st.caption("Why a list of communities: Arctic Shift's keyword search must name the subreddits to search. "
               "Searching all of Reddit at once needs Reddit's own API (a registered app: "
               "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET). Nothing here is stored.")

elif page == "🛡 Admin":
    header("Admin", "Who is using APOLLO-M, and how. Visible to administrators only; "
                    "the database enforces it.", strip=False)
    if not IS_ADMIN:
        st.stop()
    days = st.select_slider("Window", options=[1, 7, 30, 90], value=30, format_func=lambda d: f"last {d} days")
    users, acts = A.admin_activity(SESSION, since_days=days)
    U = pd.DataFrame(users)
    E = pd.DataFrame(acts)
    if U.empty:
        st.info("No accounts yet, or the accounts schema (web/supabase/accounts.sql) is not applied.")
        st.stop()
    if not E.empty:
        E["at"] = pd.to_datetime(E["at"], utc=True)
        E = E.merge(U[["id", "email"]], left_on="user_id", right_on="id", how="left")
    active = int(E["user_id"].nunique()) if not E.empty else 0
    st.markdown(T.kpi_row([
        T.kpi("Accounts", f"{len(U):,}", f"{int((U['role'] == 'admin').sum())} admin"),
        T.kpi("Active users", f"{active:,}", f"in the last {days} days"),
        T.kpi("Sign-ins", f"{int((E['action'] == 'sign_in').sum()) if not E.empty else 0:,}", "password, Google or remembered"),
        T.kpi("Topic searches", f"{int((E['action'] == 'topic_search').sum()) if not E.empty else 0:,}", "on Topic watch"),
    ]), unsafe_allow_html=True)
    if not E.empty:
        import plotly.graph_objects as _go
        per_day = E.assign(day=E["at"].dt.floor("D")).groupby(["day", "action"]).size().reset_index(name="n")
        fig = _go.Figure()
        for i, (act, g) in enumerate(per_day.groupby("action")):
            fig.add_trace(_go.Bar(x=g["day"], y=g["n"], name=act, marker_color=T.SERIES[i % len(T.SERIES)]))
        fig.update_layout(barmode="stack", height=360, title_text="Activity per day, by kind")
        show(fig, per_day, key="admin_daily")
        by_user = (E.groupby("email").agg(actions=("action", "size"), last=("at", "max"),
                                           searches=("action", lambda s: int((s == "topic_search").sum())))
                     .reset_index().sort_values("actions", ascending=False))
        st.markdown("#### Per account")
        st.dataframe(U.merge(by_user, on="email", how="left").fillna({"actions": 0, "searches": 0})
                     [["email", "full_name", "role", "created_at", "last_seen_at", "actions", "searches"]],
                     use_container_width=True, hide_index=True)
        st.markdown("#### Recent activity")
        recent = E.head(200).copy()
        recent["detail"] = recent["detail"].map(lambda d: ", ".join(f"{k}: {v}" for k, v in (d or {}).items()))
        st.dataframe(recent[["at", "email", "action", "detail"]], use_container_width=True, hide_index=True)
    st.caption("Rows come from public.activity through row-level security: a non-admin asking the same "
               "question gets only their own rows. Roles are changed in the database, not here.")

elif page == "Monitoring":
    header("Monitoring", "Both APIs, scraped every thirty seconds. Public, read-only.")
    if meso.empty:
        st.stop()
    st.markdown(T.kpi_row([
        T.kpi("apollo_communities_total", int(len(meso))),
        T.kpi("apollo_critical_alerts", int((meso["band"] == "CRITICAL").sum())),
        T.kpi("apollo_avg_chi", f"{meso['community_health_index'].mean():.1f}"),
        T.kpi("apollo_avg_toxicity", f"{meso['toxicity_rate'].mean():.3f}"),
    ]), unsafe_allow_html=True)
    st.write("")
    oc = st.columns(3)
    oc[0].link_button("📊 Open Grafana", GRAFANA_URL, use_container_width=True)
    oc[1].link_button("📈 Prometheus (PromQL)", PROMETHEUS_URL, use_container_width=True)
    oc[2].link_button("🔎 Live /metrics", f"{API_URL}/metrics", use_container_width=True)
    st.caption("Grafana is public and read-only: the APOLLO-M dashboard opens without an account. "
               "Prometheus scrapes the deployed API's /metrics every 30 seconds and keeps the history Grafana "
               "charts. The hosted services sleep when idle on the free tier, so the first open can take "
               "about a minute.")
    try:
        import requests as _rq
        txt = _rq.get(f"{API_URL}/metrics", timeout=8).text
        with st.expander("Live scrape output (Prometheus exposition format)"):
            st.code(txt, language="text")
    except Exception:
        st.caption("Scrape endpoint unreachable right now — the API may be waking from sleep.")
    with st.expander("PromQL used by the Grafana dashboard"):
        for q, desc in {
            "apollo_avg_chi": "Average Community Health Index (0-100)",
            "apollo_critical_alerts": "Communities currently CRITICAL",
            "apollo_avg_toxicity": "Mean toxicity across communities",
            "apollo_communities_total": "Communities analysed",
            "rate(apollo_api_requests_total[5m])": "API request rate by route",
            "histogram_quantile(0.95, sum by (le) (rate(apollo_api_request_seconds_bucket[5m])))":
                "API p95 latency",
        }.items():
            st.markdown(f"- `{q}` — {desc}")
