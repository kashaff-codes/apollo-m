"""
Tests for the calendar-day handling in the TFT forecaster (no training needed).
Run from project root: python -m pytest tests/test_tft_calendar.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from models.tft_forecaster import TFTConfig, TFTForecaster


def _daily(sub: str, days: list[int], start="2026-10-23") -> pd.DataFrame:
    dates = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(days, unit="D")
    return pd.DataFrame({
        "subreddit": sub, "date": dates,
        "avg_toxicity": 0.1, "toxic_rate": 0.1, "comment_count": 10,
    })


def test_time_idx_keeps_calendar_gaps():
    """Missing days must stay gaps in time_idx (not be squeezed to consecutive numbers)."""
    f = TFTForecaster(TFTConfig())
    out = f.prepare_dataframe(_daily("a", [0, 1, 2, 6, 7, 8]))
    assert out["time_idx"].tolist() == [0, 1, 2, 6, 7, 8]


def test_time_idx_shares_one_origin_across_subreddits():
    f = TFTForecaster(TFTConfig())
    df = pd.concat([_daily("a", list(range(0, 24))), _daily("b", list(range(3, 24)))])
    out = f.prepare_dataframe(df)
    assert out.loc[out["subreddit"] == "r/b", "time_idx"].min() == 3


def test_eligibility_uses_calendar_span_and_gap_ratio():
    f = TFTForecaster(TFTConfig())   # needs 14 + 5 = 19 calendar days
    df = pd.concat([
        _daily("full", list(range(24))),                          # ok
        _daily("gappy_ok", [d for d in range(24) if d not in (4, 5, 10, 14, 16, 17, 19)]),   # 17/24 observed
        _daily("too_short", list(range(13))),                     # span 13 < 19
        _daily("too_sparse", [0, 5, 10, 15, 20, 23]),             # span 24 but 6 observed (< 60%)
    ])
    got = set(f.eligible_subreddits(df))
    assert got == {"r/full", "r/gappy_ok"}, got


def test_future_rows_continue_after_last_observed_day():
    f = TFTForecaster(TFTConfig())
    df = f.prepare_dataframe(_daily("a", list(range(0, 20))))
    f.origin_date = df["date"].dt.normalize().min()
    out = f._with_future_rows(df, anchor_idx=19)
    assert out["time_idx"].tail(5).tolist() == [20, 21, 22, 23, 24]
    assert out["date"].iloc[-1] == pd.Timestamp("2026-10-23", tz="UTC") + pd.Timedelta(days=24)
    assert not out[["avg_toxicity", "toxic_rate", "comment_count"]].isna().any().any()
