"""
Visual system for the APOLLO-M console — the Occuris family, in night mode.

APOLLO-M shares Occuris's design language (https://occuris.pages.dev):
Fraunces for titles, Geist for everything a reader works with, JetBrains Mono
for timestamps, identifiers and the status strip; ink, paper and violet. An
analyst console is used for long sessions, so it runs on Occuris's own `night`
token rather than the marketing site's paper ground.

Brand colour is for chrome only — titles, actions, rules. Data marks use roles
with fixed rules, so a colour always means one thing:

  SERIES    categorical identity, fixed order, never cycled. Validated on SURFACE
            below (dark, adjacent pairs): worst CVD ΔE 8.4, normal-vision ΔE 19.3,
            every slot ≥ 3:1.
  SEQ       one hue, magnitude; dark → light so near-zero recedes into the surface.
  STATUS    the four alert bands only, always with marker shape and name — red and
            green are ΔE 4.1 apart under deuteranopia (apollo_bands).
  FORECAST  forecasts and their bands. Occuris violet #6E4DC9 is 3.3:1 on night —
            enough for a button, not for a line or text — so marks use #A58CF0 (6.9:1).

Text on night: paper 17:1, mist 11:1, muted 6.5:1 — all above WCAG AA.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio

# Occuris tokens
NIGHT = "#110C1F"
PAPER = "#F4F1EC"
MIST = "#C9BFE3"
VIOLET = "#6E4DC9"
AMBER = "#E8A317"

PAGE = NIGHT
SURFACE = "#1A1430"
INK = PAPER
INK_2 = MIST
MUTED = "#9A91B4"
GRID = "#2A2343"
AXIS = "#3B3358"
FORECAST = "#A58CF0"
LINK = "#A58CF0"

SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
SEQ = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
       "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
SEQ_SCALE = [[i / (len(SEQ) - 1), c] for i, c in enumerate(SEQ)]
SERIF = '"Fraunces", Georgia, serif'
FONT = '"Geist", system-ui, -apple-system, "Segoe UI", sans-serif'
MONO = '"JetBrains Mono", ui-monospace, "Cascadia Mono", Consolas, monospace'


def rgba(hex_: str, a: float) -> str:
    h = hex_.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


pio.templates["apollo"] = go.layout.Template(layout=go.Layout(
    font=dict(family=FONT, color=INK_2, size=13),
    paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
    colorway=SERIES,
    title=dict(font=dict(family=SERIF, color=INK, size=17), x=0.01, xanchor="left"),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=INK_2, size=12))),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
               title=dict(font=dict(color=INK_2, size=12))),
    # Title top-left, legend top-right: anchored to opposite corners so they
    # cannot overlap however long either one is.
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2, size=12), orientation="h",
                yanchor="bottom", y=1.01, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="#251D40", bordercolor=AXIS, font=dict(color=INK, family=FONT)),
    margin=dict(l=16, r=16, t=64, b=16),
    transition=dict(duration=500, easing="cubic-in-out"),
))
pio.templates.default = "apollo"

_FONTS = ("https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300..600"
          "&family=Geist:wght@400..700&family=JetBrains+Mono:wght@400;500&display=swap")

CSS = f"""
<style>
@import url("{_FONTS}");
@property --num {{ syntax: '<integer>'; initial-value: 0; inherits: false; }}
@keyframes count {{ from {{ --num: 0; }} to {{ --num: var(--to); }} }}
@keyframes rise  {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .25; }} }}

.stApp {{
  background: radial-gradient(1100px 560px at 10% -12%, {rgba(VIOLET, .16)}, transparent 62%), {PAGE};
  color: {INK}; font-family: {FONT};
}}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp td, .stApp th {{ font-family: {FONT}; }}
@media (prefers-reduced-motion: reduce) {{
  .kpi, .kpi .v, .prov, .strip .dot, .stPlotlyChart {{ animation: none !important; }}
}}
h1, h2, h3, h4 {{ color: {INK} !important; font-family: {SERIF} !important; font-weight: 400 !important;
                 letter-spacing: -.02em; }}
code, kbd, .mono {{ font-family: {MONO} !important; }}
a {{ color: {LINK}; }}

.apollo-title {{ font-family: {SERIF}; font-size: 2.6rem; font-weight: 400; line-height: 1.05;
                letter-spacing: -.025em; color: {INK}; }}
.apollo-title .mark {{ color: {LINK}; }}
.apollo-sub {{ color: {INK_2}; margin-top: 6px; font-size: 1.02rem; max-width: 62ch; }}
.hook {{ font-family: {SERIF}; font-size: 2.9rem; font-weight: 400; line-height: 1.02;
         letter-spacing: -.03em; color: {INK}; }}

.strip {{ font-family: {MONO}; font-size: .74rem; color: {MUTED}; display: inline-flex; flex-wrap: wrap;
          gap: 14px; align-items: center; padding: 6px 14px; border: 1px solid {rgba(MIST, .14)};
          border-radius: 999px; background: {rgba(MIST, .04)}; margin-bottom: 18px; }}
.strip b {{ color: {INK}; font-weight: 500; }}
.strip .dot {{ width: 6px; height: 6px; border-radius: 50%; background: {AMBER}; display: inline-block;
               animation: blink 2.4s ease-in-out infinite; }}

.kpi {{ background: {SURFACE}; border: 1px solid {rgba(MIST, .12)}; border-radius: 14px; padding: 16px 18px;
        animation: rise .5s ease-out both; }}
.kpi:nth-child(2) {{ animation-delay: .05s; }} .kpi:nth-child(3) {{ animation-delay: .1s; }}
.kpi:nth-child(4) {{ animation-delay: .15s; }} .kpi:nth-child(5) {{ animation-delay: .2s; }}
.kpi .v {{ font-size: 2rem; font-weight: 600; color: {INK}; line-height: 1.15; letter-spacing: -.01em; }}
.kpi .v.count {{ animation: count 1.4s cubic-bezier(.2,.8,.2,1) both; counter-reset: num var(--num); }}
.kpi .v.count::after {{ content: counter(num); }}
.kpi .v.text {{ font-size: 1.15rem; line-height: 1.35; overflow-wrap: anywhere; padding: 6px 0 3px; }}
.kpi .l {{ font-family: {MONO}; font-size: .68rem; color: {INK_2}; text-transform: uppercase;
           letter-spacing: .08em; margin-top: 2px; }}
.kpi .s {{ font-size: .8rem; color: {MUTED}; margin-top: 3px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }}

.prov {{ border-radius: 14px; padding: 14px 18px; margin: 4px 0 16px; background: {SURFACE};
         border: 1px solid {rgba(MIST, .12)}; border-left: 3px solid {LINK}; animation: rise .45s ease-out both; }}
.prov.bench {{ border-left-color: {AMBER}; }}
.prov b {{ color: {INK}; font-family: {SERIF}; font-weight: 400; font-size: 1.15rem; }}
.prov .meta {{ color: {INK_2}; font-size: .9rem; margin-top: 6px; }}
.chip {{ display: inline-block; padding: 1px 9px; border-radius: 999px; font-family: {MONO}; font-size: .7rem;
         border: 1px solid {AXIS}; color: {INK_2}; margin: 0 6px 4px 0; vertical-align: middle; }}

.band {{ display: inline-flex; gap: 6px; align-items: center; padding: 2px 10px;
        border-radius: 999px; font-weight: 600; font-size: .8rem; color: {INK}; border: 1px solid; }}

.stPlotlyChart {{ border-radius: 14px; overflow: hidden; border: 1px solid {rgba(MIST, .10)};
                 animation: rise .55s ease-out both; }}
section[data-testid="stSidebar"] {{ background: #0C0816; border-right: 1px solid {rgba(MIST, .10)}; }}
div[data-testid="stExpander"] details {{ border-color: {rgba(MIST, .10)}; }}
.finding {{ border-left: 3px solid {LINK}; padding: 10px 16px; margin: 8px 0 14px;
           background: {rgba(VIOLET, .08)}; border-radius: 0 12px 12px 0; color: {INK_2}; }}
.finding b {{ color: {INK}; }}
.feed {{ font-size: .92rem; line-height: 1.5; margin: 0 0 8px; }}
.feed .t {{ font-family: {MONO}; font-size: .72rem; color: {MUTED}; }}
</style>
"""


def status_strip(items: list[tuple[str, str]], live: bool = True) -> str:
    """Occuris's mono status line: `06:58 · 1,087,806 comments read · …`."""
    dot = '<span class="dot"></span>' if live else ""
    parts = "".join(f"<span><b>{v}</b> {k}</span>" for k, v in items)
    return f'<div class="strip">{dot}{parts}</div>'


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
