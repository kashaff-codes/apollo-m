"""The web view's findings file must match what the console computes."""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
warnings.filterwarnings("ignore", category=FutureWarning)


def _module():
    spec = importlib.util.spec_from_file_location("export_web_evidence", ROOT / "scripts" / "export_web_evidence.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_file_is_current():
    committed = json.loads((ROOT / "outputs" / "web_evidence.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(_module().build())), \
        "outputs/web_evidence.json is stale — run python scripts/export_web_evidence.py"


def test_it_carries_what_the_page_shows():
    e = _module().build()
    assert len(e["corpora"]) == 4
    assert any(a["rho"] > 0.85 for a in e["agreement"])
    assert {"mae_tft", "mae_last_value", "mae_recent_mean"} <= set(e["backtest"]["real_api_2026_07"])
    assert 0 < e["label_agreement"]["precision"] < 1
