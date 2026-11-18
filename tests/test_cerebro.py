"""
CEREBRO misinformation module (modules/cerebro_detector.py).

Previously `from modules.cerebro_detector import CEREBRODetector; print("syntax OK")`
— an import check that asserted nothing. These tests pin the keyword path, which
is what runs in any checkout without models/cerebro_classifier.pkl (git-ignored)
and without network access for the 1.6 GB zero-shot model.
"""
from __future__ import annotations

import pandas as pd
import pytest

import modules.cerebro_detector as cd


@pytest.fixture
def detector(monkeypatch):
    """Force the offline keyword path: no pickled classifier, zero-shot load fails."""
    class _NoFile:                      # replaces the module's Path only, not pathlib's
        def __init__(self, *a): pass
        def exists(self): return False
    monkeypatch.setattr(cd, "Path", _NoFile)

    def no_network(*a, **k):
        raise OSError("offline in tests")
    monkeypatch.setattr(cd, "pipeline", no_network)
    d = cd.CEREBRODetector()
    assert d.classifier is None and not d.use_supervised
    return d


RESULT_KEYS = {"misinformation_score", "source_credibility",
               "coordinated_behaviour", "misinfo_category", "model_used"}


def test_result_shape(detector):
    out = detector.analyze("The council approved the new budget on Tuesday.")
    assert RESULT_KEYS <= set(out)
    assert out["model_used"] == "keyword_fallback"
    assert 0 <= out["misinformation_score"] <= 1 and 0 <= out["source_credibility"] <= 1
    assert out["misinfo_category"] == "NEUTRAL"


def test_sourced_text_reads_as_credible(detector):
    """Credibility is scored from its own phrase list, independently of misinfo."""
    out = detector.analyze("According to a peer reviewed study published in March, "
                           "the data shows the evidence is consistent.")
    assert out["misinfo_category"] == "CREDIBLE"
    assert out["source_credibility"] > 0.5 and out["misinformation_score"] == 0


def test_coordination_signals_are_detected(detector):
    spam = "Share this before it gets deleted — mainstream media won't report it. Wake up!"
    assert detector.analyze(spam)["coordinated_behaviour"] is True
    assert detector.analyze("Lovely weather in Karachi today.")["coordinated_behaviour"] is False


def test_conspiracy_text_scores_above_plain_news(detector):
    plain = detector.analyze("The central bank held interest rates steady this quarter.")
    wild = detector.analyze("The plandemic was a false flag by the deep state, do your own research.")
    assert wild["misinformation_score"] > plain["misinformation_score"]


def test_batch_and_community_index(detector):
    df = pd.DataFrame({
        "subreddit": ["r/a", "r/a", "r/b", "r/b"],
        "body": ["deep state hoax, wake up", "false flag, repost this",
                 "rates held steady", "museum opens at nine"],
    })
    scored = detector.analyze_batch(df)
    assert len(scored) == len(df)
    idx = detector.compute_community_misinfo_index(scored)
    assert set(idx["subreddit"]) == {"r/a", "r/b"}
    score_col = next(c for c in idx.columns if c != "subreddit" and pd.api.types.is_numeric_dtype(idx[c]))
    by = idx.set_index("subreddit")[score_col]
    assert by["r/a"] > by["r/b"], f"{score_col}: conspiracy-heavy community did not score higher"
