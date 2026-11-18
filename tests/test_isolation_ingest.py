"""
Isolation tests for the two new ingest stages, on small synthetic inputs.

  * scripts/local/filter_arctic_dump.py (Rizwan) — streams a monthly .zst dump
    and keeps a capped, seeded sample per subreddit per day.
  * build_corpus.py (Aaqib) — cleans the Arctic Shift CSV and cuts one
    contiguous window with a capped sample per subreddit per day.

Both ran on hundreds of thousands to a million real comments, which no CI can
hold. These tests feed each stage a handful of rows constructed so every rule is
exercised at least once — including the edge cases Sourcery raised on #2 — and
check the stage in isolation from everything downstream.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DAY0 = datetime(2026, 8, 1, tzinfo=timezone.utc)


def _ts(day: int, hour: int = 12) -> int:
    return int((DAY0 + timedelta(days=day, hours=hour)).timestamp())


# ═════════════════════════════════════════════════════════════════════════════
# filter_arctic_dump.py
# ═════════════════════════════════════════════════════════════════════════════

zstd = pytest.importorskip("zstandard")


@pytest.fixture(scope="module")
def fad():
    spec = importlib.util.spec_from_file_location(
        "filter_arctic_dump", ROOT / "scripts" / "local" / "filter_arctic_dump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(tmp_path: Path, rows: list) -> Path:
    raw = b"".join((r if isinstance(r, bytes) else json.dumps(r).encode()) + b"\n" for r in rows)
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "RC_test.zst"
    p.write_bytes(zstd.ZstdCompressor().compress(raw))
    return p


def _c(cid, sub="news", day=0, author="alice", body="a real comment", hour=12, **extra):
    return {"id": cid, "subreddit": sub, "author": author, "body": body,
            "score": 1, "created_utc": _ts(day, hour), **extra}


def _run(fad, tmp_path, rows, cap=1000, seed=42, start="2026-08-01", end="2026-08-03"):
    out = tmp_path / "out.csv"
    summary = fad.filter_dump(_dump(tmp_path, rows), out, ["news", "aww"], cap, seed,
                              fad.parse_date(start), fad.parse_date(end, end=True),
                              progress_every=0)
    kept = list(csv.DictReader(out.open(encoding="utf-8")))
    return summary, kept


def test_keeps_only_requested_subreddits_case_insensitively(fad, tmp_path):
    s, kept = _run(fad, tmp_path, [_c("a", "News"), _c("b", "aww"), _c("c", "gaming")])
    assert sorted(r["id"] for r in kept) == ["a", "b"]
    assert {r["subreddit"] for r in kept} == {"news", "aww"}       # canonical spelling


def test_drops_deleted_removed_and_counts_them(fad, tmp_path):
    rows = [_c("a"), _c("b", body="[deleted]"), _c("c", body="[removed]"),
            _c("d", body="[ Removed by Reddit ]"), _c("e", author="[deleted]"), _c("f", body="   ")]
    s, kept = _run(fad, tmp_path, rows)
    assert [r["id"] for r in kept] == ["a"]
    assert s["stats"]["deleted_or_removed"] == 5


def test_window_is_inclusive_utc_days(fad, tmp_path):
    rows = [_c("before", day=-1, hour=23), _c("first", day=0, hour=0),
            _c("last", day=2, hour=23), _c("after", day=3, hour=0)]
    s, kept = _run(fad, tmp_path, rows)
    assert sorted(r["id"] for r in kept) == ["first", "last"]
    assert s["stats"]["outside_window"] == 2


def test_missing_fields_and_bad_json_are_counted_never_invented(fad, tmp_path):
    no_author = _c("x"); del no_author["author"]
    rows = [_c("ok"), no_author, b'{"subreddit": "news", broken json', _c("dup"), _c("dup")]
    s, kept = _run(fad, tmp_path, rows)
    assert sorted(r["id"] for r in kept) == ["dup", "ok"]
    assert s["stats"]["missing_field"] == 1
    assert s["stats"]["bad_json"] == 1
    assert s["stats"]["duplicate_id"] == 1


def test_escaped_subreddit_text_in_a_body_cannot_spoof_the_filter(fad, tmp_path):
    """The fast path finds the subreddit with a regex before parsing JSON."""
    spoof = _c("s", sub="gaming", body='he wrote "subreddit": "news" in his post')
    s, kept = _run(fad, tmp_path, [spoof])
    assert kept == []


def test_cap_is_per_subreddit_per_day_and_eligible_counts_are_pre_cap(fad, tmp_path):
    rows = [_c(f"n{i}", "news", day=0, hour=i % 24) for i in range(10)] + \
           [_c(f"m{i}", "news", day=1) for i in range(2)]
    s, kept = _run(fad, tmp_path, rows, cap=3)
    per = s["per_subreddit_day"]
    assert per["news|2026-08-01"] == {"eligible": 10, "kept": 3, "hit_cap": True}
    assert per["news|2026-08-02"] == {"eligible": 2, "kept": 2, "hit_cap": False}
    assert len(kept) == 5 and s["days_hitting_cap"] == 1


def test_sampling_is_deterministic_for_a_seed(fad, tmp_path):
    rows = [_c(f"n{i}", hour=i % 24) for i in range(40)]
    _, a = _run(fad, tmp_path / "a", rows, cap=5, seed=7)
    _, b = _run(fad, tmp_path / "b", rows, cap=5, seed=7)
    _, c = _run(fad, tmp_path / "c", rows, cap=5, seed=8)
    ids = lambda k: [r["id"] for r in k]
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_output_has_no_extra_columns(fad, tmp_path):
    _, kept = _run(fad, tmp_path, [_c("a", extra_field="should not leak")])
    assert list(kept[0]) == list(fad.FIELDS)


# ═════════════════════════════════════════════════════════════════════════════
# build_corpus.py
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def corpus(tmp_path, monkeypatch):
    import build_corpus
    monkeypatch.setattr(build_corpus, "DATA", tmp_path)

    def run(rows, **kw):
        pd.DataFrame(rows).to_csv(tmp_path / "reddit_live_sample.csv", index=False)
        build_corpus.main(**{"cap": 1000, "seed": 42, "min_days": 3, **kw})
        out = pd.read_csv(tmp_path / "apollo_comments_real.csv")
        summary = json.loads((tmp_path / "corpus_summary.json").read_text())
        return out, summary
    return run


def _row(cid, sub="r/news", day=0, author="bob", body="hello there", hour=12):
    return {"id": cid, "subreddit": sub, "author": author, "body": body,
            "score": 1, "created_utc": _ts(day, hour)}


def _days(n, sub="r/news", per_day=1, start=0):
    return [_row(f"{sub}-{d}-{k}", sub, d, hour=k % 24) for d in range(start, start + n) for k in range(per_day)]


def test_normalises_prefix_and_keeps_schema(corpus):
    out, _ = corpus(_days(3))
    assert set(out["subreddit"]) == {"news"}
    assert list(out.columns[:6]) == ["id", "subreddit", "author", "body", "score", "created_utc"]
    assert out["toxicity_score"].isna().all(), "the builder must not invent scores"


def test_null_subreddit_does_not_become_a_community_called_nan(corpus):
    rows = _days(3) + [{**_row("x"), "subreddit": None}]
    out, s = corpus(rows)
    assert "nan" not in set(out["subreddit"].astype(str))
    assert s["drop_counts"]["missing_fields"] == 1


def test_bad_timestamps_are_counted(corpus):
    rows = _days(3) + [{**_row("bad"), "created_utc": "not-a-time"}]
    _, s = corpus(rows)
    assert s["drop_counts"]["bad_timestamp"] == 1
    assert s["raw_rows"] - sum(s["drop_counts"].values()) == s["rows_in_window"]


def test_picks_the_longest_contiguous_run(corpus):
    rows = _days(3, start=0) + _days(5, start=10)        # gap between day 2 and day 10
    out, s = corpus(rows)
    assert s["window_days"] == 5
    assert s["window_start"] == "2026-08-11" and s["window_end"] == "2026-08-15"
    assert len(out) == 5


def test_refuses_rather_than_falling_back_to_a_gappy_window(corpus):
    with pytest.raises(SystemExit, match="Refusing"):
        corpus(_days(2, start=0) + _days(2, start=5), min_days=3)


def test_cap_and_seed(corpus):
    rows = _days(3, per_day=20)
    out, s = corpus(rows, cap=4)
    assert out.groupby(pd.to_datetime(out["created_utc"], unit="s").dt.date).size().eq(4).all()
    assert all(v["eligible"] == 20 and v["kept"] == 4 and v["hit_cap"] for v in s["per_subreddit_day"].values())
    again, _ = corpus(rows, cap=4)
    assert out["id"].tolist() == again["id"].tolist()


def test_reports_subreddits_with_gaps_inside_the_window(corpus):
    rows = _days(4, "r/news") + [_row("a0", "r/aww", 0), _row("a3", "r/aww", 3)]
    _, s = corpus(rows)
    assert s["subreddits_complete"] == ["news"]
    assert s["subreddits_with_gaps"] == ["aww"]
