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

OUTPUTS = ROOT / "outputs"


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
    p = OUTPUTS / "forecast_results.csv"
    assert p.exists(), "forecast_results.csv not found — run MACRO layer"
    df = pd.read_csv(p)
    assert len(df) > 0, "forecast_results.csv is empty"


def test_forecast_has_multiple_subreddits():
    p = OUTPUTS / "forecast_results.csv"
    if not p.exists():
        pytest.skip("forecast_results.csv not found")
    df = pd.read_csv(p)
    col = next((c for c in df.columns if "subreddit" in c.lower() or "group" in c.lower()), None)
    if col is None:
        pytest.skip("No subreddit/group column in forecast_results.csv")
    n = df[col].nunique()
    assert n >= 5, f"Forecast covers only {n} subreddits — expected at least 5"


def test_silhouette_score():
    df = pd.read_csv(OUTPUTS / "clusters_report.csv")
    assert len(df) > 0, "clusters_report.csv is empty"
    # At minimum it must exist and be non-trivial
    sil_col = next((c for c in df.columns if "silhouette" in c.lower()), None)
    if sil_col:
        mean_sil = df[sil_col].mean()
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
