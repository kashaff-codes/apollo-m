"""
Shared test setup.

Corpus-dependent tests
----------------------
Some tests read the scored comment corpus (data/apollo_comments.csv and
data/apollo_comments_real.csv). Those files are git-ignored on purpose: they hold
raw comment text and author handles. So in a clean checkout, and in CI, they are
absent, and the tests that read them used to fail — which made the whole suite
red for a reason that says nothing about the code.

Tests marked ``@pytest.mark.corpus`` are therefore skipped when the corpus file
is missing, with the reason printed. That is an absence-of-input skip only: when
the file exists the tests run with every assertion. Sourcery's review of #2 was
right that a test which skips because a *column* is missing hides breakage;
these skip only because the *whole input* is not on this machine.

On a machine that is supposed to have the corpus, set APOLLO_REQUIRE_CORPUS=1
and the skips become failures, so a missing or misnamed file cannot pass
quietly there.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "data" / "apollo_comments.csv"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "corpus: needs the git-ignored scored comment corpus (data/apollo_comments.csv)",
    )
    config.addinivalue_line(
        "markers",
        "slow: loads a transformer or trains a model; skipped unless APOLLO_SLOW=1",
    )


def pytest_collection_modifyitems(config, items):
    require = os.getenv("APOLLO_REQUIRE_CORPUS") == "1"
    run_slow = os.getenv("APOLLO_SLOW") == "1"
    no_corpus = pytest.mark.skip(
        reason="scored corpus data/apollo_comments.csv is git-ignored and not on this "
               "machine — build it with build_corpus.py + the MICRO layer "
               "(set APOLLO_REQUIRE_CORPUS=1 to make this a failure)"
    )
    no_slow = pytest.mark.skip(reason="slow test — set APOLLO_SLOW=1 to run")
    for item in items:
        if "corpus" in item.keywords and not CORPUS.exists() and not require:
            item.add_marker(no_corpus)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(no_slow)
