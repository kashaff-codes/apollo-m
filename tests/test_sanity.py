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
# Rebuilt daily series. data/apollo_daily.csv is the protected benchmark and is not tested here.
DAILY   = OUTPUTS / "timeseries_with_embeddings.csv"


# ─── apollo_comments.csv ────────────────────────────────────────────────────

def test_comments_has_many_rows():
    df = pd.read_csv(DATA / "apollo_comments.csv")
    assert len(df) >= 1000, (
        f"Only {len(df)} comments — expected at least 1,000 real comments"
    )


def test_rebuilt_corpus_matches_pipeline_input():
    """apollo_comments.csv (what the pipeline reads) must be the rebuilt corpus plus scores,
    i.e. contain exactly the comment ids in apollo_comments_real.csv (build_corpus.py output)."""
    real_path = DATA / "apollo_comments_real.csv"
    assert real_path.exists(), "apollo_comments_real.csv missing - run build_corpus.py"
    real = pd.read_csv(real_path, usecols=["id", "created_utc"])
    used = pd.read_csv(DATA / "apollo_comments.csv", usecols=["id"])
    assert len(real) >= 1000, f"Rebuilt corpus has only {len(real)} rows"
    assert set(real["id"]) == set(used["id"]), (
        "apollo_comments.csv and apollo_comments_real.csv contain different comments"
    )


def test_comments_multiple_subreddits():
    df = pd.read_csv(DATA / "apollo_comments.csv")
    n = df["subreddit"].nunique()
    assert n >= 10, f"Only {n} unique subreddits — expected at least 10"


def test_comments_has_real_toxicity_scores():
    """Scores must be numeric floats in [0,1], not all NaN and not all 0."""
    df = pd.read_csv(DATA / "apollo_comments.csv")
    assert "toxicity_score" in df.columns, "toxicity_score column missing — run the micro layer"
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
    assert "is_toxic" in df.columns and df["is_toxic"].notna().any(), "is_toxic missing or empty — run the micro layer"
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
    df = pd.read_csv(DAILY)
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
    df = pd.read_csv(DAILY)
    zero_pct = (df["avg_toxicity"] == 0.0).mean()
    assert zero_pct < 0.05, (
        f"{zero_pct:.1%} of rows have avg_toxicity=0.0 — likely zero-filled missing days"
    )


def test_daily_multiple_subreddits():
    df = pd.read_csv(DAILY)
    n = df["subreddit"].nunique() if "subreddit" in df.columns else 0
    assert n >= 10, f"Only {n} subreddits in apollo_daily.csv — expected at least 10"


def test_daily_contiguous_dates():
    """The corpus window must be one contiguous run of >=24 days. Individual
    subreddits may miss days (missing days are never zero-filled)."""
    df = pd.read_csv(DAILY)
    df["date"] = pd.to_datetime(df["date"])
    days = sorted(df["date"].dt.normalize().unique())
    gaps = [(days[k] - days[k - 1]).days for k in range(1, len(days))]
    assert max(gaps, default=0) <= 1, f"Window has a gap of {max(gaps)} days"
    assert len(days) >= 24, f"Window spans only {len(days)} days (need >= 24)"


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
    assert "polarization" in df.columns, "polarization column missing from meso_report.csv"
    std = df["polarization"].std()
    assert std > 1e-6, (
        f"polarization std={std:.8f} — all subreddits have identical polarization "
        f"(synthetic graph or global-only calculation)"
    )


def test_meso_chi_varies():
    """CHI scores must not all be the same value."""
    df = pd.read_csv(OUTPUTS / "meso_report.csv")
    col = "chi" if "chi" in df.columns else ("community_health_index" if "community_health_index" in df.columns else None)
    assert col is not None, "No CHI column in meso_report.csv"
    std = df[col].std()
    assert std > 0.1, f"CHI std={std:.4f} — scores are suspiciously flat"


# ─── clusters_report.csv ───────────────────────────────────────────────────

def test_clusters_report_valid():
    df = pd.read_csv(OUTPUTS / "clusters_report.csv")
    assert len(df) >= 5, "clusters_report.csv has fewer than 5 rows"
    cluster_col = next((c for c in df.columns if "cluster" in c.lower()), None)
    assert cluster_col is not None, "No cluster column found in clusters_report.csv"
    assert df[cluster_col].nunique() >= 2, "Only 1 cluster — clustering failed"
