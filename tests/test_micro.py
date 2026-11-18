"""
Micro layer — toxic-bert scoring.

Previously a script that read data/apollo_comments.csv at import time (so it
aborted collection whenever the git-ignored corpus was absent) and printed
results without asserting anything. It now scores fixed sentences, so it needs
no corpus, and checks that the scores mean something.

The tests that load unitary/toxic-bert (~420 MB on first run) are marked slow;
run them with APOLLO_SLOW=1. The rest exercise analyze_batch without a model.
"""
from __future__ import annotations

import pandas as pd
import pytest

BENIGN = [
    "Thanks for the detailed explanation, that really helped me understand the paper.",
    "The museum opens at nine on weekends; I would recommend going early.",
    "Great photo! The lighting on the mountains is beautiful.",
]
TOXIC = [
    "You are a worthless idiot and everyone here hates you.",
    "Shut up, you pathetic moron, nobody cares about your stupid opinion.",
]


def _modelless_layer():
    """A MicroLayer without loading a transformer, so analyze_batch runs as-is."""
    from main import Config, MicroLayer
    micro = MicroLayer.__new__(MicroLayer)
    micro.cfg = Config()
    micro.model_pipeline = None
    return micro


# ── no model needed ─────────────────────────────────────────────────────────

def test_analyze_batch_writes_no_files(monkeypatch):
    """
    PR #1 made analyze_batch save its input frame over data/apollo_comments.csv,
    which deleted every comment a sampled run did not include. That PR was
    superseded and not merged; this keeps the behaviour from coming back.
    """
    writes = []
    monkeypatch.setattr(pd.DataFrame, "to_csv",
                        lambda self, *a, **k: writes.append(a[0] if a else k))
    _modelless_layer().analyze_batch(pd.DataFrame({"subreddit": ["r/a"], "body": ["hello"]}))
    assert not writes, f"analyze_batch wrote {writes}"


def test_cached_scores_are_reused_not_recomputed():
    micro = _modelless_layer()
    micro.score_comment = lambda text: pytest.fail("re-scored a comment that had a cached score")
    out = micro.analyze_batch(pd.DataFrame({"body": ["a", "b"], "toxicity_score": [0.9, 0.1]}))
    assert out["toxicity_score"].tolist() == [0.9, 0.1]
    assert out["is_toxic"].tolist() == [True, False]


def test_partially_scored_frames_are_rescored():
    """A single missing score means the cache cannot be trusted for the batch."""
    micro = _modelless_layer()
    seen = []
    micro.score_comment = lambda text: seen.append(text) or 0.5
    micro.analyze_batch(pd.DataFrame({"body": ["a", "b"], "toxicity_score": [0.9, None]}))
    assert seen == ["a", "b"]


# ── the real model ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scored():
    from main import Config, MicroLayer
    micro = MicroLayer(Config())
    # Without this the layer silently drops to a five-keyword heuristic, and the
    # tests below would be measuring that instead of toxic-bert.
    assert micro.model_pipeline is not None, "toxic-bert did not load"
    df = pd.DataFrame({"subreddit": ["r/test"] * 5, "body": BENIGN + TOXIC})
    return micro.analyze_batch(df, use_cached=False)


@pytest.mark.slow
def test_scores_are_probabilities(scored):
    assert scored["toxicity_score"].notna().all()
    assert scored["toxicity_score"].between(0, 1).all()


@pytest.mark.slow
def test_toxic_text_outscores_benign_text(scored):
    benign = scored["toxicity_score"].iloc[:3]
    toxic = scored["toxicity_score"].iloc[3:]
    assert toxic.min() > benign.max(), (
        f"toxic min {toxic.min():.3f} is not above benign max {benign.max():.3f}")


@pytest.mark.slow
def test_threshold_labels_follow_scores(scored):
    from main import Config
    assert (scored["is_toxic"] == (scored["toxicity_score"] >= Config.TOXICITY_THRESHOLD)).all()
    assert scored["is_toxic"].iloc[3:].all() and not scored["is_toxic"].iloc[:3].any()
