"""
Validation tests — verify metrics meet minimum quality thresholds.

Run from project root: python -m pytest tests/test_validation.py -v
"""
import sys, json, re
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from apollo_datasets import get as _dataset

# metrics.json describes the toxicity classifier and is shared by every corpus.
OUTPUTS = ROOT / "outputs"
# Forecast, backtest, meso and cluster checks are about the Arctic Shift rebuild,
# which now lives in its own directory instead of over the benchmark.
REBUILD = _dataset("real_arctic_2026_10").path


def _load_metrics():
    p = OUTPUTS / "metrics.json"
    assert p.exists(), "metrics.json not found — run pipeline first"
    return json.loads(p.read_text())


def test_toxicity_accuracy():
    metrics = _load_metrics()
    tox = metrics.get("Toxicity (TF-IDF+LR, Jigsaw)", {})
    raw = tox.get("accuracy", "0%")
    val = float(re.sub(r"[^0-9.]", "", str(raw))) / (100 if "%" in str(raw) else 1)
    assert val >= 0.80, f"Toxicity accuracy {val:.1%} < 80% threshold"


def test_toxicity_f1():
    metrics = _load_metrics()
    tox = metrics.get("Toxicity (TF-IDF+LR, Jigsaw)", {})
    raw = tox.get("F1 macro", "0")
    val = float(re.sub(r"[^0-9.]", "", str(raw)))
    if val > 1:
        val /= 100
    assert val >= 0.60, f"Toxicity F1 {val:.3f} < 0.60 threshold"


def test_tft_forecast_exists():
    p = REBUILD / "forecast_results.csv"
    assert p.exists(), "forecast_results.csv not found — run MACRO layer"
    df = pd.read_csv(p)
    assert len(df) > 0, "forecast_results.csv is empty"


def test_forecast_has_multiple_subreddits():
    p = REBUILD / "forecast_results.csv"
    assert p.exists(), "forecast_results.csv not found — run MACRO layer"
    df = pd.read_csv(p)
    col = next((c for c in df.columns if "subreddit" in c.lower() or "group" in c.lower()), None)
    assert col is not None, "No subreddit/group column in forecast_results.csv"
    n = df[col].nunique()
    assert n >= 5, f"Forecast covers only {n} subreddits — expected at least 5"


def test_forecast_dates_are_after_the_history():
    """A forecast must start AFTER the last observed day of that subreddit, not re-predict it."""
    fc = pd.read_csv(REBUILD / "forecast_results.csv")
    ts = pd.read_csv(REBUILD / "timeseries_with_embeddings.csv", usecols=["subreddit", "date"])
    ts["date"] = pd.to_datetime(ts["date"], utc=True).dt.tz_localize(None).dt.normalize()
    fc["date"] = pd.to_datetime(fc["date"])
    last = ts.groupby("subreddit")["date"].max()
    fc["bare"] = fc["subreddit"].str.replace(r"^r/", "", regex=True)
    first_fc = fc.groupby("bare")["date"].min()
    joined = pd.concat([first_fc.rename("first_forecast"), last.rename("last_obs")], axis=1).dropna()
    assert len(joined) >= 5, "forecast and history share fewer than 5 subreddits"
    bad = joined[joined["first_forecast"] <= joined["last_obs"]]
    assert bad.empty, f"forecasts overlap observed days for: {list(bad.index)[:5]}"


def test_forecast_backtest_exists_and_is_sane():
    p = REBUILD / "forecast_backtest.csv"
    assert p.exists(), "forecast_backtest.csv not found - run the MACRO layer"
    bt = pd.read_csv(p)
    assert len(bt) >= 10, f"Backtest covers only {len(bt)} subreddits"
    for c in ("mae_tft", "mae_last_value", "mae_recent_mean", "inside_p10_p90"):
        assert c in bt.columns and bt[c].notna().all(), f"{c} missing or has NaN in forecast_backtest.csv"
    assert bt["mae_tft"].mean() < 0.15, "Backtest MAE is implausibly large"


def test_tiny_subreddits_are_flagged_not_treated_as_healthy():
    """Subreddits seen on fewer than 7 days must carry insufficient_history=True."""
    m = pd.read_csv(REBUILD / "meso_report.csv")
    for c in ("observed_days", "insufficient_history"):
        assert c in m.columns, f"{c} missing from meso_report.csv - rerun run_full_from_cache.py"
    flagged = m[m["insufficient_history"].astype(bool)]
    assert len(flagged) >= 1, "no subreddit is flagged, but some have only 1-2 days of data"
    assert (flagged["observed_days"] < 7).all()
    assert (m.loc[~m["insufficient_history"].astype(bool), "observed_days"] >= 7).all()


def test_silhouette_score():
    df = pd.read_csv(REBUILD / "clusters_report.csv")
    assert len(df) > 0, "clusters_report.csv is empty"
    # clusters_report.csv has no silhouette column, so compute it from the PCA coordinates.
    for c in ("cluster", "pca_x", "pca_y"):
        assert c in df.columns, f"{c} missing from clusters_report.csv"
    from sklearn.metrics import silhouette_score
    assert df["cluster"].nunique() >= 2, "Need at least 2 clusters to score"
    mean_sil = silhouette_score(df[["pca_x", "pca_y"]], df["cluster"])
    assert mean_sil > -0.5, f"Mean silhouette {mean_sil:.3f} is very poor"


def test_benchmark_files_untouched():
    """
    Critical: the pipeline must NEVER overwrite the benchmark data.
    Verify apollo_daily.csv is not empty and ground_truth.json (if present) is intact.
    """
    daily = ROOT / "data" / "apollo_daily.csv"
    assert daily.exists(), "apollo_daily.csv (benchmark) is missing"
    df = pd.read_csv(daily)
    assert len(df) > 0, "apollo_daily.csv (benchmark) is empty — was it overwritten?"

    gt = ROOT / "data" / "ground_truth.json"
    if gt.exists():
        data = json.loads(gt.read_text())
        assert len(data) > 0, "ground_truth.json is empty — was it overwritten?"


def test_benchmark_files_match_git():
    """Benchmark files must be byte-identical to the committed versions."""
    import subprocess
    ref = "origin/main"  # the original benchmark lives on the upstream main
    if subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=ROOT, capture_output=True).returncode != 0:
        ref = "HEAD"
    r = subprocess.run(["git", "diff", "--quiet", ref, "--",
                        "data/apollo_daily.csv", "data/ground_truth.json"],
                       cwd=ROOT, capture_output=True)
    if r.returncode not in (0, 1):
        pytest.skip("git unavailable")
    assert r.returncode == 0, "Benchmark file differs from git HEAD - it was overwritten"
