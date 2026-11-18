"""
Tests for the bot / boilerplate filter and the insufficient-history flag.
Run from project root: python -m pytest tests/test_data_filters.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from main import drop_bots_and_boilerplate, flag_insufficient_history

CANNED = "Please read the rules before posting. " * 4   # > 100 characters


def _row(i, author="alice", body=None, sub="a", t=1761177600):
    return {"id": f"id{i}", "subreddit": sub, "author": author, "body": body or f"unique comment number {i}", "created_utc": t}


def test_bot_accounts_are_removed():
    df = pd.DataFrame([_row(1, "AutoModerator"), _row(2, "weather-bot"), _row(3, "bot_helper"), _row(4, "alice")])
    out = drop_bots_and_boilerplate(df)
    assert out["author"].tolist() == ["alice"]


def test_names_that_merely_contain_bot_are_kept():
    df = pd.DataFrame([_row(1, "Abbott"), _row(2, "robotics_fan"), _row(3, "bottle")])
    assert len(drop_bots_and_boilerplate(df)) == 3


def test_repeated_long_text_is_removed_but_short_repeats_are_kept():
    rows = [_row(i, f"user{i}", CANNED) for i in range(6)] + [_row(10 + i, f"u{i}", "lol") for i in range(8)]
    out = drop_bots_and_boilerplate(pd.DataFrame(rows))
    assert (out["body"] == "lol").sum() == 8
    assert (out["body"] == CANNED).sum() == 0


def test_long_text_repeated_fewer_than_five_times_is_kept():
    df = pd.DataFrame([_row(i, f"user{i}", CANNED) for i in range(4)])
    assert len(drop_bots_and_boilerplate(df)) == 4


def test_flag_marks_subreddits_with_few_days():
    day = 86400
    comments = pd.DataFrame(
        [_row(i, sub="long", t=1761177600 + i * day) for i in range(10)]
        + [_row(100 + i, sub="short", t=1761177600 + i * day) for i in range(2)]
    )
    report = pd.DataFrame({"subreddit": ["r/long", "r/short", "r/unknown"], "churn_rate": [0.5, 0.0, 0.0]})
    out = flag_insufficient_history(report, comments, min_days=7).set_index("subreddit")
    assert out.loc["r/long", "observed_days"] == 10 and not out.loc["r/long", "insufficient_history"]
    assert out.loc["r/short", "observed_days"] == 2 and out.loc["r/short", "insufficient_history"]
    assert out.loc["r/unknown", "insufficient_history"]
    assert out["churn_rate"].tolist() == [0.5, 0.0, 0.0]   # values are left untouched
