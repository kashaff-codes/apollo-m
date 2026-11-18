"""
The serving API, with the database in the state production is actually in.

The Render PostgreSQL the deployed API used was deleted when its free period
ended, and every data endpoint — /metrics included — answered 500. These tests
point the API at a database that does not exist and check that it serves the
committed outputs instead, says so, and reports the outage in /metrics.
"""
from __future__ import annotations

import importlib
import re

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _client(monkeypatch, dataset: str = "real"):
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/missing")
    monkeypatch.setenv("APOLLO_DATASET", dataset)
    import api.main as api
    api = importlib.reload(api)
    return TestClient(api.app), api


@pytest.fixture
def client(monkeypatch):
    return _client(monkeypatch)[0]


def test_health_reports_the_fallback_instead_of_failing(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["data_source"] == "csv:real_api_2026_07"
    assert body["database"].startswith("unavailable")


@pytest.mark.parametrize("path", ["/communities", "/alerts", "/alerts/high", "/summary", "/clusters",
                                  "/recommendations", "/datasets", "/communities/AskReddit",
                                  "/forecast/AskReddit", "/forecast/r/AskReddit",
                                  "/communities/r/AskReddit", "/recommendations/r/AskReddit", "/"])
def test_every_data_endpoint_answers_without_a_database(client, path):
    r = client.get(path)
    assert r.status_code == 200, (path, r.text[:200])


def test_fallback_serves_the_same_corpus_the_database_held(client):
    import pandas as pd
    from apollo_datasets import get
    want = set(pd.read_csv(get("real_api_2026_07").file("meso_report.csv"))["subreddit"])
    got = {c["subreddit"] for c in client.get("/communities").json()["communities"]}
    assert got == want


def test_forecast_is_the_future_not_the_hindcast(client):
    body = client.get("/forecast/r/AskReddit").json()
    assert body["kind"] == "future" and body["days"] == 5
    assert all(row["date"] > "2026-08-11" for row in body["forecast"])


def test_forecast_names_are_matched_whole_not_character_stripped(monkeypatch):
    """`.lstrip("r/")` turned r/relationship_advice into "elationship_advice"."""
    client, _ = _client(monkeypatch, "real_torrent_2026_08")
    r = client.get("/forecast/relationship_advice")
    assert r.status_code == 200, r.text[:200]
    assert {row["subreddit"] for row in r.json()["forecast"]} == {"r/relationship_advice"}
    assert client.get("/forecast/elationship_advice").status_code == 404


def test_unknown_community_is_404_not_someone_elses_rows(client):
    assert client.get("/forecast/definitely_not_a_community").status_code == 404
    assert client.get("/communities/definitely_not_a_community").status_code == 404


def test_metrics_report_the_outage_and_the_traffic(client):
    client.get("/communities")
    client.get("/communities")
    client.get("/communities/AskReddit")
    text = client.get("/metrics").text
    assert re.search(r"^apollo_database_up 0$", text, re.M)
    assert re.search(r'^apollo_info\{dataset="real_api_2026_07",source="csv"', text, re.M)
    assert re.search(r"^apollo_communities_total 24$", text, re.M)
    assert re.search(r'^apollo_api_requests_total\{method="GET",route="/communities",status="200"\} 2$', text, re.M)
    # bounded labels: the route template, not every name requested
    assert 'route="/communities/{subreddit:path}"' in text and "AskReddit" not in text.split("apollo_api")[-1]
    assert re.search(r'apollo_api_request_seconds_bucket\{route="/communities",le="\+Inf"\} 2', text)


def test_metrics_is_valid_exposition_format(client):
    """Every non-comment line is `name{labels} number` — what Prometheus parses."""
    line = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.e+-]+$')
    for row in client.get("/metrics").text.strip().splitlines():
        if not row.startswith("#"):
            assert line.match(row), row


def test_recommendations_no_longer_read_a_file_nothing_writes(client):
    body = client.get("/recommendations").json()
    assert body["count"] == 24
    assert {r["recommended_action"] for r in body["recommendations"]} <= {
        "EMERGENCY_INTERVENTION", "INCREASE_MODERATION", "WARN", "NO_ACTION"}
