"""
Visual system for the APOLLO-M console.

Brand colours (violet, cyan) are for chrome only — titles, borders, glows.
Data marks use roles with fixed rules, so a colour always means one thing:

  SERIES    categorical identity, fixed order, never cycled. Validated on the
            chart surface below (dark, adjacent pairs): worst CVD ΔE 8.4,
            normal-vision ΔE 19.3, every slot ≥ 3:1.
  SEQ       one hue, magnitude. On a dark surface the near-zero end recedes
            toward the surface, so the ramp runs dark → light.
  STATUS    the four alert bands only. Red and green separate by ΔE 4.1 under
            deuteranopia, so a band is never shown by colour alone: it always
            carries its marker shape and its name (see apollo_bands).
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio

PAGE = "#070714"
SURFACE = "#10112a"
INK = "#f0efec"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#212246"
AXIS = "#34365e"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
SEQ = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
       "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
SEQ_SCALE = [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


pio.templates["apollo"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, color=INK_2, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    colorway=SERIES,
    title=dict(font=dict(color=INK, size=15), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=INK_2, size=12))),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=INK_2, size=12))),
    # Title top-left, legend top-right: anchored to opposite corners so they
    # cannot overlap however long either one is.
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=12), orientation="h",
                yanchor="bottom", y=1.01, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="#1b1c3d", bordercolor=AXIS, font=dict(color=INK, family=FONT)),
    margin=dict(l=16, r=16, t=64, b=16),
    transition=dict(duration=500, easing="cubic-in-out"),
))
pio.templates.default = "apollo"


CSS = f"""
<style>
@property --num {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes count {{ from {{ --num: 0; }} to {{ --num: var(--to); }} }}
@keyframes rise  {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes drift {{ from {{ background-position: 0 0, 0 0, 0 0; }}
                   to   {{ background-position: 600px 300px, -400px 500px, 200px -300px; }} }}
@keyframes sweep {{ from {{ background-position: 0% 50%; }} to {{ background-position: 200% 50%; }} }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 {rgba(VIOLET, .0)}; }}
                   50% {{ box-shadow: 0 0 28px 2px {rgba(VIOLET, .28)}; }} }}

.stApp {{
  background:
    radial-gradient(1px 1px at 20px 30px, {rgba('#ffffff', .55)}, transparent 60%),
    radial-gradient(1px 1px at 140px 90px, {rgba('#ffffff', .35)}, transparent 60%),
    radial-gradient(1.5px 1.5px at 260px 160px, {rgba('#c4b5fd', .45)}, transparent 60%),
    radial-gradient(1200px 620px at 12% -10%, {rgba(VIOLET, .22)}, transparent 60%),
    radial-gradient(900px 520px at 92% 0%, {rgba(CYAN, .14)}, transparent 55%),
    {PAGE};
  background-size: 300px 220px, 380px 260px, 520px 340px, auto, auto, auto;
  animation: drift 160s linear infinite;
  color: {INK};
}}
@media (prefers-reduced-motion: reduce) {{
  .stApp, .kpi, .kpi .v, .apollo-title, .prov {{ animation: none !important; }}
}}
h1, h2, h3 {{ color: {INK} !important; letter-spacing: -.01em; }}
.apollo-title {{
  font-size: 2.3rem; font-weight: 800; line-height: 1.1;
  background: linear-gradient(90deg, {VIOLET}, {CYAN}, {VIOLET}); background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: sweep 9s linear infinite;
}}
.apollo-sub {{ color: {INK_2}; margin-top: -4px; font-size: .95rem; }}

.kpi {{
  background: linear-gradient(180deg, {rgba('#ffffff', .055)}, {rgba('#ffffff', .02)});
  border: 1px solid {rgba(VIOLET, .32)}; border-radius: 16px; padding: 16px 18px;
  animation: rise .55s ease-out both, pulse 6s ease-in-out 1s infinite;
}}
.kpi:nth-child(2) {{ animation-delay: .07s, 1.4s; }}
.kpi .v {{ font-size: 2rem; font-weight: 800; color: {INK}; line-height: 1.15; }}
.kpi .v.count {{ animation: count 1.6s cubic-bezier(.2,.8,.2,1) both; counter-reset: num var(--num); }}
.kpi .v.count::after {{ content: counter(num); }}
.kpi .v.text {{ font-size: 1.15rem; line-height: 1.35; overflow-wrap: anywhere; padding: 6px 0 3px; }}
.kpi .l {{ font-size: .72rem; color: {INK_2}; text-transform: uppercase; letter-spacing: .09em; }}
.kpi .s {{ font-size: .78rem; color: {MUTED}; margin-top: 2px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}

.prov {{
  border-radius: 14px; padding: 12px 16px; margin: 4px 0 14px;
  border: 1px solid {rgba(CYAN, .28)}; background: {rgba(CYAN, .05)};
  animation: rise .5s ease-out both;
}}
.prov.bench {{ border-color: {rgba('#fab219', .35)}; background: {rgba('#fab219', .05)}; }}
.prov b {{ color: {INK}; }}
.prov .meta {{ color: {INK_2}; font-size: .84rem; margin-top: 4px; }}
.chip {{ display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .74rem;
        border: 1px solid {AXIS}; color: {INK_2}; margin-right: 6px; }}

.band {{ display: inline-flex; gap: 6px; align-items: center; padding: 2px 10px;
        border-radius: 999px; font-weight: 700; font-size: .8rem; color: {INK};
        border: 1px solid; }}

.stPlotlyChart {{ border-radius: 16px; overflow: hidden; border: 1px solid {rgba('#ffffff', .07)};
                 animation: rise .6s ease-out both; }}
section[data-testid="stSidebar"] {{ background: #0b0b1e; border-right: 1px solid {rgba(VIOLET, .28)}; }}
div[data-testid="stExpander"] details {{ border-color: {rgba('#ffffff', .08)}; }}
.finding {{ border-left: 3px solid {VIOLET}; padding: 8px 14px; margin: 6px 0 12px;
           background: {rgba(VIOLET, .06)}; border-radius: 0 12px 12px 0; color: {INK_2}; }}
.finding b {{ color: {INK}; }}
</style>
"""


def kpi(label: str, value, sub: str = "", count: bool = False) -> str:
    """
    A stat tile. Small integers count up on load. A CSS counter cannot group
    digits, so anything from 10,000 up renders static with separators rather
    than animating into "1087806". Text values get a smaller face so a long
    community name does not break mid-word.
    """
    if count and isinstance(value, int) and 0 <= value < 10_000:
        v = f'<div class="v count" style="--to:{value}" aria-label="{value:,}"></div>'
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        v = f'<div class="v">{value:,}</div>'
    elif isinstance(value, str) and len(value) > 9 and not value[:1].isdigit():
        v = f'<div class="v text">{value}</div>'
    else:
        v = f'<div class="v">{value}</div>'
    s = f'<div class="s">{sub}</div>' if sub else ""
    return f'<div class="kpi">{v}<div class="l">{label}</div>{s}</div>'


def kpi_row(tiles: list[str]) -> str:
    return f'<div class="kpi-row">{"".join(tiles)}</div>'


_MD_SPECIALS = str.maketrans({c: f"&#{ord(c)};" for c in "*_[]()#`~|!>+-="})


def safe(text) -> str:
    """
    Third-party text, made inert before it reaches st.markdown.

    Comment bodies and usernames come from Reddit. They are rendered inside
    markup with unsafe_allow_html=True (needed for the coloured markers), so
    unescaped they were interpreted as Markdown and as raw HTML — a comment
    containing [text](url) became a live link in our page, and one containing
    tags would have become tags. HTML-escape first, then neutralise the Markdown
    metacharacters, so the text shows exactly as written.
    """
    return html.escape(str(text), quote=True).translate(_MD_SPECIALS)
