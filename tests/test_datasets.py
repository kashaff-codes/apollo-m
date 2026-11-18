"""
Every dataset the dashboard can show must satisfy one contract.

The dashboard, API and exporter render whatever a dataset directory contains
through the same code, so a dataset that breaks an assumption breaks a page —
and the four corpora were produced by three people with different scripts.
These checks run against all of them, so the contract is enforced where the
data is produced rather than discovered in front of an audience.
"""
from __future__ import annotations

import pandas as pd
import pytest

import apollo_datasets as ad
from apollo_bands import alert_band

DATASETS = list(ad.DATASETS.values())
IDS = [d.key for d in DATASETS]
MESO_COLS = {"subreddit", "community_health_index", "toxicity_rate",
             "polarization", "echo_chamber_index", "churn_rate", "total_comments"}


@pytest.fixture(params=DATASETS, ids=IDS)
def ds(request):
    return request.param


def _meso(ds):
    return pd.read_csv(ds.file("meso_report.csv"))


# ── registry ────────────────────────────────────────────────────────────────

def test_registry_keys_are_unique_and_paths_distinct():
    assert len({d.path for d in DATASETS}) == len(DATASETS), "two datasets share a directory"
    assert ad.DEFAULT in ad.DATASETS


def test_exactly_one_benchmark_and_it_is_top_level_outputs():
    bench = [d for d in DATASETS if d.kind == "benchmark"]
    assert len(bench) == 1 and bench[0].path == ad.OUTPUTS


def test_every_dataset_is_present(ds):
    assert ds.available(), f"{ds.file('meso_report.csv')} missing"
    assert ds.daily.exists(), f"daily series {ds.daily} missing"


# ── meso report ─────────────────────────────────────────────────────────────

def test_meso_schema(ds):
    missing = MESO_COLS - set(_meso(ds).columns)
    assert not missing, f"{ds.key}: meso_report.csv lacks {sorted(missing)}"


def test_meso_one_row_per_community(ds):
    m = _meso(ds)
    assert m["subreddit"].is_unique
    assert len(m) == ds.communities, (
        f"{ds.key}: registry says {ds.communities} communities, file has {len(m)}")


def test_chi_bounded_and_not_flat(ds):
    chi = _meso(ds)["community_health_index"]
    assert chi.between(0, 100).all()
    assert chi.nunique() > 1 and chi.std() > 0.1


def test_component_ranges(ds):
    m = _meso(ds)
    for c in ("toxicity_rate", "polarization", "echo_chamber_index", "churn_rate"):
        assert m[c].dropna().between(0, 1).all(), f"{ds.key}: {c} outside [0,1]"
    assert (m["total_comments"] > 0).all()


def test_bands_are_assignable(ds):
    bands = _meso(ds)["community_health_index"].map(alert_band)
    assert bands.isin(["LOW", "MEDIUM", "HIGH", "CRITICAL"]).all()


def test_benchmark_directory_holds_exactly_the_benchmark():
    """
    The benchmark is identified by its ground truth: its communities are exactly
    the ones data/ground_truth.json records. Names alone cannot tell the corpora
    apart — the simulation deliberately reuses real names such as r/politics —
    but a real corpus written into outputs/ (as #2 did) fails this immediately.
    """
    import json
    truth = set(json.loads((ad.ROOT / "data" / "ground_truth.json").read_text()))
    bench = set(_meso(ad.get("benchmark"))["subreddit"].str.replace(r"^r/", "", regex=True))
    assert bench == truth, (f"outputs/ is not the benchmark: {len(bench ^ truth)} communities "
                            f"differ from ground_truth.json, e.g. {sorted(bench ^ truth)[:3]}")


# ── clusters ────────────────────────────────────────────────────────────────

def test_clusters_align_with_meso(ds):
    c = pd.read_csv(ds.file("clusters_report.csv"))
    for col in ("subreddit", "cluster", "pca_x", "pca_y"):
        assert col in c.columns, f"{ds.key}: clusters_report.csv lacks {col}"
    assert set(c["subreddit"]) == set(_meso(ds)["subreddit"])
    assert c["cluster"].nunique() >= 2


# ── forecasts ───────────────────────────────────────────────────────────────

def _fc(ds):
    return pd.read_csv(ds.file("forecast_results.csv"))


def test_forecast_quantiles_ordered(ds):
    f = _fc(ds)
    assert {"subreddit", "day", "date", "p10", "p50", "p90"} <= set(f.columns)
    assert (f["p10"] <= f["p50"] + 1e-9).all() and (f["p50"] <= f["p90"] + 1e-9).all()
    assert f[["p10", "p50", "p90"]].stack().between(0, 1).all()


def test_forecast_five_days_per_community(ds):
    counts = _fc(ds).groupby("subreddit")["day"].nunique()
    assert (counts == 5).all(), f"{ds.key}: {counts[counts != 5].to_dict()}"


def test_forecast_communities_exist_in_meso(ds):
    assert set(_fc(ds)["subreddit"]) <= set(_meso(ds)["subreddit"])


def test_forecast_semantics_match_the_files(ds):
    """The registry's description of the forecasts must be true of the data."""
    f = _fc(ds)
    daily = pd.read_csv(ds.daily, usecols=["subreddit", "date"])
    daily["subreddit"] = daily["subreddit"].astype(str).str.replace(r"^r/", "", regex=True)
    last_obs = pd.to_datetime(daily["date"], utc=True).dt.tz_localize(None).groupby(daily["subreddit"]).max()
    f["bare"] = f["subreddit"].str.replace(r"^r/", "", regex=True)
    first = pd.to_datetime(f["date"]).groupby(f["bare"]).min()
    joined = pd.concat([first.rename("first"), last_obs.rename("last")], axis=1).dropna()
    assert len(joined) >= 5
    if ds.forecast == "future":
        assert (joined["first"] > joined["last"]).all(), "a 'future' forecast overlaps observed days"
    elif ds.forecast == "held_out":
        assert {"actual", "out_of_sample"} <= set(f.columns)
        assert f["out_of_sample"].all() and f["actual"].notna().all()
        assert (joined["first"] <= joined["last"]).all()
    else:  # hindcast: values are for observed days, but the file's dates were stamped after them
        assert ds.forecast == "hindcast"
        assert "actual" not in f.columns
