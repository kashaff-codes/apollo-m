"""
The alert bands every APOLLO-M surface shows, defined once.

A Community Health Index (0-100, higher is healthier) maps to a band:

    CHI >= 85  LOW        no action
    CHI >= 75  MEDIUM     warn
    CHI >= 65  HIGH       increase moderation
    CHI <  65  CRITICAL   emergency intervention

The same rule is also written out in the API, the database loader, the
Prometheus exporter, compute_real_metrics.py and the Supabase view (SQL cannot
import Python). tests/test_chi.py evaluates every one of those copies across the
whole 0-100 range and fails if any disagrees, so a threshold changed in one place
cannot leave the others behind.

Not to be confused with main.AlertSystem, which writes alerts.json during a
pipeline run. That is a different signal — it combines CHI with absolute
toxicity overrides (CRITICAL if CHI < 25 or toxicity > 0.8, …) — and it is not
what any dashboard, API response or web page displays.
"""
from __future__ import annotations

BANDS: tuple[tuple[float, str], ...] = ((85, "LOW"), (75, "MEDIUM"), (65, "HIGH"))
ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW")

ACTION = {
    "CRITICAL": "EMERGENCY_INTERVENTION",
    "HIGH": "INCREASE_MODERATION",
    "MEDIUM": "WARN",
    "LOW": "NO_ACTION",
}

# Status colours. Red and green are ΔE 4.1 apart for deuteranopes, so a band is
# never shown by colour alone: every chart also uses SYMBOL, every label ICON+name.
COLOR = {"CRITICAL": "#d03b3b", "HIGH": "#ec835a", "MEDIUM": "#fab219", "LOW": "#0ca30c"}
SYMBOL = {"CRITICAL": "x", "HIGH": "triangle-up", "MEDIUM": "diamond", "LOW": "circle"}
ICON = {"CRITICAL": "✕", "HIGH": "▲", "MEDIUM": "◆", "LOW": "●"}
FLOOR = {"CRITICAL": 0, "HIGH": 65, "MEDIUM": 75, "LOW": 85}


def alert_band(chi: float) -> str:
    """Band for a CHI on the 0-100 scale."""
    for floor, name in BANDS:
        if chi >= floor:
            return name
    return "CRITICAL"
