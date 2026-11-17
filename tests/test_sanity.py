"""
Sanity tests — verify the data is not flat/synthetic/broken.
These are the specific checks Kashaf flagged as "weak tests" that would
pass even on completely wrong data.

Run from project root: python -m pytest tests/test_sanity.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest


DATA    = ROOT / "data"
OUTPUTS = ROOT / "outputs"


# ─── apollo_comments.csv ────────────────────────────────────────────────────

def test_comments_has_many_rows():
    df = pd.read_csv(DATA / "apollo_comments.csv")
    assert len(df) >= 1000, (
        f"Only {len(df)} comments — expected at least 1,000 real comments"
    )


def test_comments_multiple_subreddits():
    df = pd.read_csv(DATA / "apollo_comments.csv")
    n = df["subreddit"].nunique()
    assert n >= 10, f"Only {n} unique subreddits — expected at least 10"


def test_comments_has_real_toxicity_scores():
    """Scores must be numeric floats in [0,1], not all NaN and not all 0."""
    df = pd.read_csv(DATA / "apollo_comments.csv")
    if "toxicity_score" not in df.columns:
        pytest.skip("toxicity_score column not present — micro not yet run")
    scored = df["toxicity_score"].dropna()
    assert len(scored) >= len(df) * 0.9, (
        f"Only {len(scored)}/{len(df)} rows have a toxicity_score — run micro layer"
    )
    assert scored.between(0, 1).all(), "Some toxicity_score values outside [0, 1]"
    # Must NOT be all-zero (the old synthetic placeholder)
    nonzero = (scored > 0).sum()
    assert nonzero >= len(scored) * 0.01, (
        "Fewer than 1% of toxicity scores are nonzero — looks like placeholder data"
    )


def test_comments_toxicity_rate_realistic():
    """Realistic Reddit toxic rate: 1%–50%."""
    df = pd.read_csv(DATA / "apollo_comments.csv")
    if "is_toxic" not in df.columns or df["is_toxic"].isna().all():
        pytest.skip("is_toxic not set — run micro layer first")
    rate = df["is_toxic"].mean()
    assert 0.01 < rate < 0.50, (
        f"Toxic rate {rate:.2%} outside realistic range 1%–50%"
    )


def test_comments_varied_bodies():
    """Bodies must NOT all be the same placeholder string."""
    df = pd.read_csv(DATA / "apollo_comments.csv", nrows=200)
    top_body_pct = df["body"].value_counts(normalize=True).iloc[0]
    assert top_body_pct < 0.05, (
        f"Most common comment body is {top_body_pct:.0%} of rows — looks synthetic"
    )


# ─── apollo_daily.csv ────────────────────────────────────────────────────────

def test_daily_has_real_variation():
    """
    The flat-daily bug caused avg_toxicity=0.0981 for EVERY row.
    This checks that toxicity varies meaningfully across days/communities.
    """
    df = pd.read_csv(DATA / "apollo_daily.csv")
    std = df["avg_toxicity"].std()
    assert std > 0.001, (
        f"avg_toxicity std={std:.6f} — data is effectively flat (fillna(0) or hardcoded value)"
    )


def test_daily_no_fillna_zeros():
    """
    If the pipeline zero-filled missing days, there will be many exact 0.0
    rows that look healthy when they should be absent.
    Guard: at most 5% of rows should have avg_toxicity == 0.0 exactly.
    """
    df = pd.read_csv(DATA / "apollo_daily.csv")
    zero_pct = (df["avg_toxicity"] == 0.0).mean()
    assert zero_pct < 0.05, (
        f"{zero_pct:.1%} of rows have avg_toxicity=0.0 — likely zero-filled missing days"
    )


def test_daily_multiple_subreddits():
    df = pd.read_csv(DATA / "apollo_daily.csv")
    n = df["subreddit"].nunique() if "subreddit" in df.columns else 0
    assert n >= 10, f"Only {n} subreddits in apollo_daily.csv — expected at least 10"


def test_daily_contiguous_dates():
    """Each subreddit must have no gaps larger than 1 day inside its window."""
    df = pd.read_csv(DATA / "apollo_daily.csv")
    if "subreddit" not in df.columns or "date" not in df.columns:
        pytest.skip("Missing subreddit/date columns")
    df["date"] = pd.to_datetime(df["date"])
    max_gap = 0
    for sub, grp in df.groupby("subreddit"):
        dates = sorted(grp["date"].unique())
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            if gap > max_gap:
                max_gap = gap
    assert max_gap <= 1, (
        f"Largest date gap across subreddits is {max_gap} days "
        f"— pipeline sampled non-contiguous dates"
    )


# ─── meso_report.csv ────────────────────────────────────────────────────────

def test_meso_has_communities():
    df = pd.read_csv(OUTPUTS / "meso_report.csv")
    assert len(df) >= 10, f"meso_report.csv has only {len(df)} communities"


def test_meso_polarization_varies():
    """
    The old synthetic-graph bug made polarization identical for every row.
    With a real SNAP graph, values must differ across communities.
    """
    df = pd.read_csv(OUTPUTS / "meso_report.csv")
    if "polarization" not in df.columns:
        pytest.skip("polarization column not in meso_report.csv")
    std = df["polarization"].std()
    assert std > 1e-6, (
        f"polarization std={std:.8f} — all subreddits have identical polarization "
        f"(synthetic graph or global-only calculation)"
    )


def test_meso_chi_varies():
    """CHI scores must not all be the same value."""
    df = pd.read_csv(OUTPUTS / "meso_report.csv")
    col = "chi" if "chi" in df.columns else ("community_health_index" if "community_health_index" in df.columns else None)
    if col is None:
        pytest.skip("No CHI column found")
    std = df[col].std()
    assert std > 0.1, f"CHI std={std:.4f} — scores are suspiciously flat"


# ─── clusters_report.csv ───────────────────────────────────────────────────

def test_clusters_report_valid():
    df = pd.read_csv(OUTPUTS / "clusters_report.csv")
    assert len(df) >= 5, "clusters_report.csv has fewer than 5 rows"
    cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
    assert cluster_col is not None, "No cluster column found in clusters_report.csv"
    assert df[cluster_col].nunique() >= 2, "Only 1 cluster — clustering failed"
