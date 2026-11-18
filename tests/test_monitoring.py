"""
The hosted monitoring stack must agree with the code it monitors.

A Grafana panel whose query names a metric nobody exports, or whose datasource
uid nobody provisions, does not error — it renders "No data" forever. These
tests catch that before a deploy instead of during a demo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HOSTED = ROOT / "monitoring" / "hosted"
PROM_BUILTINS = {"up", "scrape_duration_seconds", "vector"}
PROMQL_WORDS = {"sum", "max", "min", "by", "rate", "histogram_quantile", "or", "le", "status", "route",
                "job", "band", "method", "dataset", "source"}


def _dashboards() -> dict[str, dict]:
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (HOSTED / "dashboards").glob("*.json")}


def _targets():
    for name, d in _dashboards().items():
        for p in d["panels"]:
            for t in p.get("targets", []):
                yield name, p["title"], t


def test_committed_dashboards_match_the_generator(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_dashboards", HOSTED / "build_dashboards.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT", tmp_path)
    mod.main()
    for f in tmp_path.glob("*.json"):
        committed = (HOSTED / "dashboards" / f.name).read_text(encoding="utf-8")
        assert f.read_text(encoding="utf-8") == committed, f"{f.name} is stale — rerun build_dashboards.py"


def test_datasource_uids_are_provisioned():
    prov = (HOSTED / "provisioning" / "datasources" / "datasources.yml").read_text(encoding="utf-8")
    uids = set(re.findall(r"^\s+uid:\s*(\S+)", prov, re.M))
    for name, title, t in _targets():
        assert t["datasource"]["uid"] in uids, f"{name} / {title}: unknown datasource {t['datasource']}"


@pytest.fixture(scope="module")
def exported(monkeypatch_module):
    """Metric names the API actually serves, with the database gone as in production."""
    pytest.importorskip("fastapi")
    import importlib
    from fastapi.testclient import TestClient
    monkeypatch_module.setenv("DATABASE_URL", "postgresql://nobody:nothing@127.0.0.1:1/missing")
    import api.main as api
    api = importlib.reload(api)
    client = TestClient(api.app)
    client.get("/communities")
    client.get("/does-not-exist")
    text = client.get("/metrics").text
    names = set(re.findall(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)", text, re.M))
    return {re.sub(r"_(bucket|sum|count)$", "", n) for n in names} | names


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_every_queried_metric_is_exported(exported):
    for name, title, t in _targets():
        if "expr" not in t:
            continue
        expr = re.sub(r'\{[^}]*\}|"[^"]*"|\[[^\]]*\]', " ", t["expr"])
        for ident in re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", expr):
            if ident in PROM_BUILTINS or ident in PROMQL_WORDS:
                continue
            assert ident in exported, f"{name} / {title}: queries {ident}, which /metrics does not export"


def test_corpus_options_point_at_files_that_exist():
    d = _dashboards()["apollo-corpus.json"]
    corpus_var = next(v for v in d["templating"]["list"] if v["name"] == "corpus")
    options = [part.split(":")[-1].strip() for part in corpus_var["query"].split(",")]
    files = {re.search(r"\$\{corpus\}/(\S+)$", t["url"]).group(1) for _, _, t in _targets() if "url" in t}
    for opt in options:
        for f in files:
            assert (ROOT / "outputs" / opt / f).exists(), f"outputs/{opt}/{f} missing — the panel would be empty"


def test_scrape_targets_are_the_deployed_apis():
    cfg = (HOSTED / "prometheus.yml").read_text(encoding="utf-8")
    app = (ROOT / "dashboard" / "app.py").read_text(encoding="utf-8")
    api_host = re.search(r'APOLLO_API_URL", "https://([^"/]+)', app).group(1)
    assert f'"{api_host}"' in cfg, "Prometheus does not scrape the API the console links to"
    assert "scheme: https" in cfg


def test_container_files_use_lf_and_render_builds_them():
    for f in ("entrypoint.sh", "Caddyfile", "prometheus.yml", "Dockerfile"):
        assert b"\r\n" not in (HOSTED / f).read_bytes(), f"{f} has CRLF line endings"
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "dockerfilePath: ./monitoring/hosted/Dockerfile" in render
    assert (HOSTED / "Dockerfile").exists()


def test_grafana_is_public_but_read_only():
    df = (HOSTED / "Dockerfile").read_text(encoding="utf-8")
    assert "GF_AUTH_ANONYMOUS_ENABLED=true" in df
    assert "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer" in df
    entry = (HOSTED / "entrypoint.sh").read_text(encoding="utf-8")
    assert "GF_SECURITY_ADMIN_PASSWORD=" not in df, "no admin password baked into the image"
    assert "/dev/urandom" in entry, "an unset admin password must become a random one, not Grafana's default"
    assert "--web.enable-admin-api" not in entry and "--web.enable-lifecycle" not in entry
