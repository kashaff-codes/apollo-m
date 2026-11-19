"""
The Community Health Index, defined once.

MesoLayer (main.py) computes it inside the full pipeline; the scheduled live
ingest (scripts/live_ingest.py) computes it every few hours without loading
torch. Both call this function, so there is one definition. Two definitions
existed once (compute_real_metrics.py used 70/30/5/5), and published numbers
depended on which script had run last.

Every input is 0-1 and the weights sum to 100, so the penalty is already on a
0-100 scale: CHI = 100 - penalty, higher is healthier.
"""
from __future__ import annotations

WEIGHTS = {"toxicity": 35, "polarization": 30, "churn": 20, "echo_chamber": 15}


def community_health_index(toxicity_rate: float, polarization: float,
                           churn_rate: float, echo_chamber: float) -> float:
    penalty = (toxicity_rate * WEIGHTS["toxicity"] + polarization * WEIGHTS["polarization"]
               + churn_rate * WEIGHTS["churn"] + echo_chamber * WEIGHTS["echo_chamber"])
    return round(max(0.0, min(100.0, 100 - penalty)), 2)
