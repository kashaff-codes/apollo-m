"""
Community Health Index — the formula in main.py, and the alert bands every surface shows.

This file used to be a script that read outputs/community_health_report.csv at
import time and re-typed the CHI formula with August's 70/30/5/5 weights. The
file is git-ignored, so the script aborted collection of the whole suite in any
clean checkout, and the weights it checked were no longer the ones in main.py.
These tests exercise the real implementation instead.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from apollo_bands import alert_band

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def chi():
    from main import Config, MesoLayer
    return MesoLayer(Config()).community_health_index


# ── formula ─────────────────────────────────────────────────────────────────

def test_chi_extremes(chi):
    """Weights sum to 100 and inputs are 0-1, so the scale is exactly 0..100."""
    assert chi(0, 0, 0, 0) == 100.0
    assert chi(1, 1, 1, 1) == 0.0


def test_chi_star_100_overflow_regression(chi):
    """
    `100 - penalty*100` pushed every realistic community below zero, so CHI was
    clamped to 0 for all of them. It was fixed in August and came back in a
    refactor once already. Realistic inputs must land well inside the range.
    """
    value = chi(0.12, 0.2, 0.3, 0.25)
    assert 40 < value < 100, f"realistic inputs gave CHI={value}"


@pytest.mark.parametrize("component", range(4))
def test_chi_falls_as_any_component_rises(chi, component):
    base = [0.2, 0.2, 0.2, 0.2]
    worse = list(base)
    worse[component] = 0.6
    assert chi(*worse) < chi(*base)


def test_chi_weights_match_the_report(chi):
    """CHI = 100 - (35T + 30P + 20C + 15E), report eq. 3.1."""
    for weight, args in [(35, (1, 0, 0, 0)), (30, (0, 1, 0, 0)),
                         (20, (0, 0, 1, 0)), (15, (0, 0, 0, 1))]:
        assert chi(*args) == pytest.approx(100 - weight)


def test_chi_is_clamped(chi):
    assert chi(5, 5, 5, 5) == 0.0
    assert chi(-5, -5, -5, -5) == 100.0


# ── alert bands: one rule, six copies ───────────────────────────────────────

def _lift(path: str, names: tuple[str, ...]):
    """
    Execute one band function from a source file without importing the module.

    Importing api/main.py or database/db_setup.py would open database
    connections; importing the dashboard would start Streamlit. The functions
    themselves are pure, so they are compiled on their own.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, path, "exec"), ns)
            return ns[node.name]
    raise AssertionError(f"no {names} function in {path}")


def _sql_band():
    sql = (ROOT / "web/supabase/schema.sql").read_text(encoding="utf-8")
    rules = [(float(n), b) for n, b in
             re.findall(r"when community_health_index >= (\d+(?:\.\d+)?) then '(\w+)'", sql)]
    default = re.search(r"else '(\w+)'\s*end as alert", sql).group(1)
    assert rules, "no banding CASE found in schema.sql"

    def band(v: float) -> str:
        for floor, name in rules:
            if v >= floor:
                return name
        return default
    return band


def _imports_canonical(path: str) -> bool:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return any(isinstance(n, ast.ImportFrom) and n.module == "apollo_bands"
               and any(a.name == "alert_band" for a in n.names) for n in ast.walk(tree))


def test_api_uses_the_canonical_bands_rather_than_a_copy():
    assert _imports_canonical("api/main.py")


SURFACES = {
    "database/db_setup.py": lambda: _lift("database/db_setup.py", ("band",)),
    "monitoring/exporter.py": lambda: _lift("monitoring/exporter.py", ("_band",)),
    "compute_real_metrics.py": lambda: _lift("compute_real_metrics.py", ("band",)),
    "web/supabase/schema.sql": _sql_band,
}


@pytest.mark.parametrize("surface", sorted(SURFACES))
def test_every_surface_bands_chi_identically(surface):
    band = SURFACES[surface]()
    grid = [i / 10 for i in range(0, 1001)]          # 0.0 .. 100.0 in 0.1 steps
    wrong = [(v, band(v), alert_band(v)) for v in grid if band(v) != alert_band(v)]
    assert not wrong, (f"{surface} disagrees with apollo_bands.alert_band at "
                       f"{len(wrong)} points, first {wrong[:3]}")


def test_band_edges():
    assert [alert_band(v) for v in (64.99, 65, 74.99, 75, 84.99, 85)] == \
        ["CRITICAL", "HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
