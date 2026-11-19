"""
The live path: the scheduled ingest (scripts/live_ingest.py) and on-demand
Topic watch (apollo_live.py). Offline: the network is never called here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import apollo_live as AL  # noqa: E402
import live_ingest as LI  # noqa: E402
from apollo_chi import community_health_index  # noqa: E402


def test_one_chi_definition():
    """The live job and the pipeline share apollo_chi; the formula is pinned here."""
    assert community_health_index(0.2, 0.25, 0.8, 0.3) == round(100 - (0.2 * 35 + 0.25 * 30 + 0.8 * 20 + 0.3 * 15), 2)
    assert community_health_index(5, 5, 5, 5) == 0.0          # clamped
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "from apollo_chi import community_health_index" in src
    assert "toxicity_rate * 35" not in src, "a second CHI formula crept back into main.py"


def test_communities_list_parses_comments_and_prefixes(tmp_path, monkeypatch):
    f = tmp_path / "c.txt"
    f.write_text("# header\nr/news\nnews  # duplicate\nAskReddit\n\n")
    monkeypatch.setattr(LI, "LIST", f)
    assert LI.communities() == ["news", "AskReddit"]


def test_the_tracked_list_covers_every_real_corpus():
    tracked = set(LI.communities())
    for p in ["outputs/real_torrent_2026_08", "outputs/real_arctic_2026_10", "outputs/real"]:
        subs = set(pd.read_csv(ROOT / p / "meso_report.csv")["subreddit"].str.removeprefix("r/"))
        assert subs <= tracked, f"{p}: {sorted(subs - tracked)} not tracked live"


def _pulse(days: int, subs=("news", "tiny")) -> pd.DataFrame:
    now = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
    rows = []
    for d in range(days):
        for h in (0, 6, 12, 18):
            t = now - timedelta(days=d, hours=h)
            rows.append({"run_at": t.isoformat(), "subreddit": "news", "comments": 100,
                         "mean_toxicity": 0.10 + 0.002 * d, "p90_toxicity": 0.4, "share_toxic": 0.08,
                         "authors": 90})
            if "tiny" in subs:
                rows.append({"run_at": t.isoformat(), "subreddit": "tiny", "comments": 1,
                             "mean_toxicity": 0.9, "p90_toxicity": 0.9, "share_toxic": 1.0, "authors": 1})
    return pd.DataFrame(rows), now


def test_health_uses_the_window_and_the_minimum_comment_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(LI, "CACHE", tmp_path)
    pulse, now = _pulse(10)
    rep = LI.health(pulse, now)
    assert list(rep["subreddit"]) == ["r/news"], "a community under 30 comments in the window must be skipped"
    row = rep.iloc[0]
    assert row["total_comments"] == 7 * 4 * 100          # only the last 7 days
    assert 0 <= row["community_health_index"] <= 100
    assert row["alert"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert row["structure_from"] == "snap"               # r/news has SNAP polarisation from a corpus


def test_churn_is_the_pipelines_half_window_definition(tmp_path, monkeypatch):
    monkeypatch.setattr(LI, "CACHE", tmp_path)
    store = tmp_path / "authors"
    store.mkdir()
    now = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)
    early = [f"a{i}" for i in range(40)]
    late = early[:10] + [f"b{i}" for i in range(30)]
    for d in range(7):
        day = (now - timedelta(days=d)).date().isoformat()
        (store / f"{day}.json").write_text(json.dumps({"news": late if d < 3 else early}))
    assert LI.churn("news", now) == 0.75                 # 30 of the 40 earlier authors gone
    assert LI.churn("unknown", now) is None


def test_daily_series_and_recent_mean_forecast():
    pulse, _ = _pulse(9, subs=("news",))
    daily, fc = LI.daily_and_forecast(pulse)
    assert set(daily.columns) >= {"subreddit", "date", "avg_toxicity", "comment_count"}
    assert daily["subreddit"].str.startswith("r/").all()
    row = fc.iloc[0]
    assert row["method"] == "recent_mean_7d"
    assert row["p10"] <= row["p50"] <= row["p90"]
    assert row["date"] > daily["date"].max()             # a forecast of the next day


def test_topic_watch_counts_and_reports_failures(monkeypatch):
    calls = []

    def fake_get(path, params, tries=3):
        calls.append(params["subreddit"])
        if params["subreddit"] == "broken":
            return [], "Timeout"
        return ([{"id": "abc", "subreddit": params["subreddit"], "created_utc": 1795040000,
                  "title": f"Hiring remote dev ({params['subreddit']})", "num_comments": 3}], "")

    monkeypatch.setattr(AL, "_get", fake_get)
    res = AL.topic_watch("remote", ["r/forhire", "remotework", "broken"], hours=6)
    assert sorted(calls) == ["broken", "forhire", "remotework"]
    assert len(res["rows"]) == 2
    assert res["failed"] == {"r/broken": "Timeout"}
    assert res["rows"][0]["link"] == "https://redd.it/abc"


@pytest.mark.parametrize("name", list(AL.TOPICS))
def test_topic_presets_are_bare_subreddit_names(name):
    assert all(s and "/" not in s and " " not in s for s in AL.TOPICS[name])
