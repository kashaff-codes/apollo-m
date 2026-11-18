"""
The paper is reproducible from committed files only if those files never change.

tests/test_validation.py::test_benchmark_files_match_git guarded two of them
(data/apollo_daily.csv, data/ground_truth.json) by diffing against origin/main.
That missed the others — merging #2 replaced outputs/meso_report.csv,
outputs/clusters_report.csv and outputs/forecast_results.csv, all paper inputs,
and nothing failed — and a diff against origin/main passes trivially once the
change is pushed.

docs/paper/inputs.sha256 pins every input by content instead. A deliberate
change means regenerating that manifest in the same commit, which is visible in
review; an accidental one fails here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "paper" / "inputs.sha256"


def _entries():
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(maxsplit=1)
            yield path, digest


@pytest.mark.parametrize("path,digest", list(_entries()))
def test_paper_input_unchanged(path, digest):
    f = ROOT / path
    assert f.exists(), f"{path} is missing — the paper cannot be reproduced"
    actual = hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert actual == digest, (
        f"{path} differs from the version the paper was built from. If that is "
        f"intended, regenerate docs/paper/inputs.sha256 in the same commit.")


def test_manifest_covers_every_paper_script_input():
    """Every file the reproduction scripts read must be pinned."""
    pinned = {p for p, _ in _entries()}
    for needed in ["outputs/meso_report.csv", "outputs/forecast_results.csv",
                   "outputs/real/meso_report.csv", "outputs/real/forecast_results.csv",
                   "data/apollo_daily.csv", "data/ground_truth.json"]:
        assert needed in pinned, f"{needed} is a paper input but is not pinned"
