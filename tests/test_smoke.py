"""
Smoke tests — verify outputs exist and are non-empty after a pipeline run.
Run from the project root: python -m pytest tests/test_smoke.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

OUTPUTS = ROOT / "outputs"
DATA    = ROOT / "data"


@pytest.mark.corpus
def test_comments_csv_exists():
    p = DATA / "apollo_comments.csv"
    assert p.exists(), f"apollo_comments.csv not found at {p}"


def test_daily_csv_exists():
    p = DATA / "apollo_daily.csv"
    assert p.exists(), f"apollo_daily.csv not found at {p}"


def test_outputs_dir_exists():
    assert OUTPUTS.exists(), "outputs/ directory missing"


def test_meso_report_exists():
    p = OUTPUTS / "meso_report.csv"
    assert p.exists(), "meso_report.csv not found in outputs/"


def test_clusters_report_exists():
    p = OUTPUTS / "clusters_report.csv"
    assert p.exists(), "clusters_report.csv not found in outputs/"


def test_metrics_json_exists():
    p = OUTPUTS / "metrics.json"
    assert p.exists(), "metrics.json not found in outputs/"


@pytest.mark.corpus
def test_comments_csv_loads():
    df = pd.read_csv(DATA / "apollo_comments.csv", nrows=50)
    assert len(df) >= 10, "apollo_comments.csv has fewer than 10 rows"


def test_daily_csv_loads():
    df = pd.read_csv(DATA / "apollo_daily.csv")
    assert len(df) >= 1, "apollo_daily.csv is empty"
