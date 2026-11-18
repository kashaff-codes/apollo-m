"""
Generate the hosted Grafana dashboards.

    python monitoring/hosted/build_dashboards.py

Writes dashboards/apollo-overview.json (the home dashboard, live from
Prometheus) and dashboards/apollo-corpus.json (corpus history, from the
committed CSVs via the Infinity datasource). Generated rather than
hand-edited: Grafana JSON is verbose, and a typo in a hand-edited datasource
uid fails silently as an empty panel. tests/test_monitoring.py checks the
output against the scrape config and the API's /metrics.

Colours: alert bands use the status palette and their names (never colour
alone); categorical series use the dashboard console's fixed SERIES order.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dashboards"
REPO = "kashaff-codes/apollo-m"

PROM = {"type": "prometheus", "uid": "prometheus"}
CORPUS = {"type": "yesoreyeram-infinity-datasource", "uid": "corpus"}
BAND = {"CRITICAL": "#d03b3b", "HIGH": "#ec835a", "MEDIUM": "#fab219", "LOW": "#0ca30c"}
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
CHI_STEPS = [{"color": BAND["CRITICAL"], "value": None}, {"color": BAND["HIGH"], "value": 65},
             {"color": BAND["MEDIUM"], "value": 75}, {"color": BAND["LOW"], "value": 85}]

_id = 0


def _next() -> int:
    global _id
    _id += 1
    return _id


def prom(expr: str, legend: str = "", instant: bool = False) -> dict:
    return {"datasource": PROM, "expr": expr, "legendFormat": legend, "refId": "A",
            "instant": instant, "range": not instant}


def panel(kind: str, title: str, x: int, y: int, w: int, h: int, targets: list, desc: str = "",
          **extra) -> dict:
    p = {"id": _next(), "type": kind, "title": title, "description": desc,
         "gridPos": {"x": x, "y": y, "w": w, "h": h}, "targets": targets,
         "datasource": targets[0]["datasource"] if targets else PROM}
    p.update(extra)
    return p


def row(title: str, y: int) -> dict:
    return {"id": _next(), "type": "row", "title": title, "collapsed": False,
            "gridPos": {"x": 0, "y": y, "w": 24, "h": 1}, "panels": []}


def stat(title, x, y, w, h, target, unit="none", steps=None, decimals=None, mappings=None, desc="",
         text_mode="value", color_mode="background"):
    return panel("stat", title, x, y, w, h, [target], desc,
                 fieldConfig={"defaults": {"unit": unit, "decimals": decimals, "mappings": mappings or [],
                                           "thresholds": {"mode": "absolute",
                                                          "steps": steps or [{"color": SERIES[0], "value": None}]},
                                           "color": {"mode": "thresholds"}}, "overrides": []},
                 options={"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                          "colorMode": color_mode, "graphMode": "area", "textMode": text_mode,
                          "justifyMode": "center", "orientation": "auto"})


def up_mapping(up_text: str, down_text: str) -> list:
    return [{"type": "value", "options": {
        "1": {"text": up_text, "color": BAND["LOW"], "index": 0},
        "0": {"text": down_text, "color": BAND["HIGH"], "index": 1}}},
        {"type": "special", "options": {"match": "null", "result": {"text": "no data yet", "color": "#898781"}}}]


def timeseries(title, x, y, w, h, targets, unit="none", desc="", stack=False, overrides=None, legend="list"):
    return panel("timeseries", title, x, y, w, h, targets, desc,
                 fieldConfig={"defaults": {"unit": unit, "color": {"mode": "palette-classic"},
                                           "custom": {"lineWidth": 2, "fillOpacity": 12, "showPoints": "never",
                                                      "spanNulls": False, "gradientMode": "opacity",
                                                      "stacking": {"mode": "normal" if stack else "none"}}},
                              "overrides": overrides or []},
                 options={"legend": {"displayMode": legend, "placement": "bottom", "showLegend": True},
                          "tooltip": {"mode": "multi", "sort": "desc"}})


# ═════════════════════════════════════════════════════════════════════════════
# Operations (home) — live, from Prometheus
# ═════════════════════════════════════════════════════════════════════════════

def overview() -> dict:
    p = [row("Services", 0)]
    p += [
        stat("APOLLO-M API", 0, 1, 6, 4, prom('max(up{job="apollo-api"})', instant=True),
             mappings=up_mapping("● UP", "✕ DOWN"), desc="Prometheus reached the deployed API's /metrics."),
        stat("CEREBRO API", 6, 1, 6, 4, prom('max(up{job="cerebro-api"})', instant=True),
             mappings=up_mapping("● UP", "✕ DOWN"), desc="Prometheus reached CEREBRO's /metrics."),
        stat("APOLLO-M database", 12, 1, 6, 4, prom("max(apollo_database_up)", instant=True),
             mappings=up_mapping("● connected", "▲ serving committed outputs"),
             desc="0 means PostgreSQL is unreachable and the API is serving the committed outputs of the "
                  "same corpus — the same rows, labelled as such in every response."),
        stat("Serving corpus", 18, 1, 6, 4, prom("max by (dataset, source) (apollo_info)", "{{dataset}} · {{source}}",
                                                 instant=True), text_mode="name", color_mode="none"),
    ]
    p.append(row("Community health — the corpus the API serves", 5))
    p += [
        stat("Communities analysed", 0, 6, 4, 5, prom("max(apollo_communities_total)", instant=True)),
        stat("Average CHI", 4, 6, 4, 5, prom("max(apollo_avg_chi)", instant=True), decimals=1, steps=CHI_STEPS,
             desc="Community Health Index, 0-100. Bands: ✕ CRITICAL < 65 ≤ ▲ HIGH < 75 ≤ ◆ MEDIUM < 85 ≤ ● LOW"),
        stat("Lowest CHI", 8, 6, 4, 5, prom("min(apollo_min_chi)", instant=True), decimals=1, steps=CHI_STEPS),
        stat("Mean toxicity", 12, 6, 4, 5, prom("max(apollo_avg_toxicity)", instant=True), unit="percentunit",
             decimals=1),
        panel("bargauge", "Communities by alert band", 16, 6, 8, 5,
              [prom("max by (band) (apollo_communities_by_band)", "{{band}}", instant=True)],
              "Same rule as the console, API and web view (apollo_bands.py).",
              fieldConfig={"defaults": {"color": {"mode": "fixed", "fixedColor": SERIES[0]}, "min": 0},
                           "overrides": [{"matcher": {"id": "byName", "options": b},
                                          "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}},
                                                         {"id": "displayName", "value": f"{i} {b}"}]}
                                         for (b, c), i in zip(BAND.items(), ["✕", "▲", "◆", "●"])]},
              options={"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                       "valueMode": "color", "namePlacement": "left",
                       "reduceOptions": {"calcs": ["lastNotNull"], "values": False}}),
    ]
    p.append(row("Traffic", 11))
    p += [
        timeseries("Requests per second, by route", 0, 12, 12, 8,
                   [prom('sum by (route) (rate(apollo_api_requests_total[5m]))', "{{route}}")], unit="reqps"),
        timeseries("Latency, p50 and p95", 12, 12, 6, 8,
                   [prom('histogram_quantile(0.5, sum by (le) (rate(apollo_api_request_seconds_bucket[5m])))', "p50"),
                    {**prom('histogram_quantile(0.95, sum by (le) (rate(apollo_api_request_seconds_bucket[5m])))',
                            "p95"), "refId": "B"}], unit="s"),
        timeseries("Server errors (5xx) per second", 18, 12, 6, 8,
                   [prom('sum(rate(apollo_api_requests_total{status=~"5.."}[5m])) or vector(0)', "5xx")],
                   unit="reqps"),
        timeseries("Scrape duration — includes waking a sleeping free-tier service", 0, 20, 12, 7,
                   [prom("scrape_duration_seconds", "{{job}}")], unit="s"),
        timeseries("Target up", 12, 20, 12, 7, [prom("up", "{{job}}")]),
    ]
    p.append(row("Model quality — measured offline, published by the API", 27))
    p += [
        panel("bargauge", "Forecast error on held-out days (MAE, lower is better)", 0, 28, 12, 7,
              [prom("max by (method) (apollo_forecast_mae)", "{{method}}", instant=True)],
              "TFT against repeating the last value and the recent mean. On real Reddit the recent mean wins — "
              "reported as found.",
              fieldConfig={"defaults": {"unit": "none", "decimals": 4, "min": 0,
                                        "color": {"mode": "fixed", "fixedColor": SERIES[0]}},
                           "overrides": [{"matcher": {"id": "byName", "options": n},
                                          "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": c}}]}
                                         for n, c in [("tft", SERIES[0]), ("last_value", SERIES[1]),
                                                      ("recent_mean", SERIES[2])]]},
              options={"orientation": "horizontal", "displayMode": "basic", "showUnfilled": True,
                       "reduceOptions": {"calcs": ["lastNotNull"], "values": False}}),
        stat("p10-p90 band coverage", 12, 28, 4, 7, prom("max(apollo_forecast_coverage)", instant=True),
             unit="percentunit", decimals=0, desc="Share of held-out actuals inside the 80% band."),
        stat("Classifier F1 (macro)", 16, 28, 4, 7, prom("max(apollo_toxicity_f1_macro)", instant=True), decimals=3),
        stat("Instability ROC-AUC (benchmark)", 20, 28, 4, 7, prom("max(apollo_instability_roc_auc)", instant=True),
             decimals=3, desc="Against planted ground truth — the score looks for the planted shape."),
    ]
    return {
        "uid": "apollo-overview", "title": "APOLLO-M · Operations", "tags": ["apollo-m"],
        "timezone": "utc", "refresh": "30s", "time": {"from": "now-1h", "to": "now"},
        "schemaVersion": 39, "version": 1, "editable": False, "graphTooltip": 1,
        "links": [{"title": "Corpus history", "type": "link", "url": "/d/apollo-corpus", "icon": "dashboard"},
                  {"title": "Prometheus", "type": "link", "url": "/prometheus/", "targetBlank": True},
                  {"title": "APOLLO-M console", "type": "link", "url": "https://apollo-m.streamlit.app",
                   "targetBlank": True}],
        "panels": p, "templating": {"list": []}, "annotations": {"list": []},
    }


# ═════════════════════════════════════════════════════════════════════════════
# Corpus history — from the committed CSVs
# ═════════════════════════════════════════════════════════════════════════════

def csv_query(path: str, columns: list[tuple[str, str]], ref="A") -> dict:
    return {"refId": ref, "datasource": CORPUS, "type": "csv", "source": "url", "format": "table",
            "parser": "backend",
            "url": f"https://raw.githubusercontent.com/{REPO}/${{ref}}/outputs/${{corpus}}/{path}",
            "url_options": {"method": "GET", "data": ""},
            "columns": [{"selector": c, "text": c, "type": t} for c, t in columns]}


def by_community(field: str) -> list:
    # One series per community, named "r/<community>" rather than "<field> <community>".
    return [{"id": "partitionByValues", "options": {"fields": ["subreddit"], "keepFields": False}},
            {"id": "renameByRegex", "options": {"regex": r"^\S+ (.+)$", "renamePattern": "r/$1"}}]


def corpus() -> dict:
    daily_cols = [("date", "timestamp"), ("subreddit", "string"), ("avg_toxicity", "number"),
                  ("toxic_rate", "number"), ("comment_count", "number")]
    meso_cols = [("subreddit", "string"), ("community_health_index", "number"), ("toxicity_rate", "number"),
                 ("polarization", "number"), ("churn_rate", "number"), ("echo_chamber_index", "number"),
                 ("total_comments", "number")]
    tox = timeseries("Daily mean toxicity, every community", 0, 0, 24, 11,
                     [{**csv_query("daily_series.csv", [daily_cols[0], daily_cols[1], daily_cols[2]])}],
                     unit="none", legend="table",
                     desc="One line per community. Gaps are days with no data — never zero-filled. "
                          "Click a name in the legend to isolate it.")
    tox["transformations"] = by_community("avg_toxicity")
    tox["fieldConfig"]["defaults"]["custom"]["fillOpacity"] = 0
    tox["options"]["legend"].update({"placement": "right", "calcs": ["mean", "max"], "sortBy": "Mean",
                                     "sortDesc": True})
    rate = timeseries("Daily share of comments over the toxicity threshold (0.6)", 0, 11, 24, 9,
                      [csv_query("daily_series.csv", [daily_cols[0], daily_cols[1], daily_cols[3]])],
                      unit="percentunit", legend="table")
    rate["transformations"] = by_community("toxic_rate")
    rate["fieldConfig"]["defaults"]["custom"]["fillOpacity"] = 0
    rate["options"]["legend"].update({"placement": "right", "calcs": ["mean"], "sortBy": "Mean", "sortDesc": True})

    bars = panel("bargauge", "Community Health Index, lowest first", 0, 20, 10, 14,
                 [csv_query("meso_report.csv", meso_cols[:2])],
                 "✕ CRITICAL < 65 ≤ ▲ HIGH < 75 ≤ ◆ MEDIUM < 85 ≤ ● LOW",
                 transformations=[{"id": "sortBy", "options": {"sort": [{"field": "community_health_index"}]}},
                                  {"id": "rowsToFields", "options": {"mappings": [
                                      {"fieldName": "subreddit", "handlerKey": "field.name"},
                                      {"fieldName": "community_health_index", "handlerKey": "field.value"}]}}],
                 fieldConfig={"defaults": {"min": 0, "max": 100, "decimals": 1,
                                           "thresholds": {"mode": "absolute", "steps": CHI_STEPS},
                                           "color": {"mode": "thresholds"}}, "overrides": []},
                 options={"orientation": "horizontal", "displayMode": "gradient", "showUnfilled": True,
                          "valueMode": "color", "namePlacement": "left",
                          "reduceOptions": {"calcs": ["lastNotNull"], "values": False}})
    table = panel("table", "Every community — the four inputs to CHI", 10, 20, 14, 14,
                  [csv_query("meso_report.csv", meso_cols)],
                  "CHI = 100 − (35·toxicity + 30·polarisation + 20·churn + 15·echo).",
                  transformations=[{"id": "sortBy", "options": {"sort": [{"field": "community_health_index"}]}}],
                  fieldConfig={"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                               "overrides": [
                                   {"matcher": {"id": "byName", "options": "community_health_index"},
                                    "properties": [{"id": "displayName", "value": "CHI"},
                                                   {"id": "decimals", "value": 1},
                                                   {"id": "thresholds", "value": {"mode": "absolute",
                                                                                  "steps": CHI_STEPS}},
                                                   {"id": "custom.cellOptions",
                                                    "value": {"type": "color-background", "mode": "gradient"}}]},
                                   *[{"matcher": {"id": "byName", "options": c},
                                      "properties": [{"id": "decimals", "value": 3},
                                                     {"id": "custom.cellOptions",
                                                      "value": {"type": "gauge", "mode": "basic"}},
                                                     {"id": "min", "value": 0}, {"id": "max", "value": 1},
                                                     {"id": "color", "value": {"mode": "fixed",
                                                                               "fixedColor": SERIES[0]}}]}
                                     for c in ("toxicity_rate", "polarization", "churn_rate", "echo_chamber_index")]]},
                  options={"showHeader": True, "cellHeight": "sm"})
    variables = [
        {"name": "corpus", "label": "Corpus", "type": "custom", "query":
            "RC_2026-08 full monthly dump : real_torrent_2026_08,Arctic Shift rebuild Oct–Nov 2026 : real_arctic_2026_10",
         "current": {"text": "RC_2026-08 full monthly dump", "value": "real_torrent_2026_08"},
         "options": [], "hide": 0, "multi": False, "includeAll": False},
        {"name": "ref", "type": "constant", "query": "main", "hide": 2},
    ]
    return {
        "uid": "apollo-corpus", "title": "APOLLO-M · Corpus history", "tags": ["apollo-m"],
        # Opens on the RC_2026-08 window; the link below jumps to the rebuild's window.
        "timezone": "utc", "time": {"from": "2026-07-31T00:00:00.000Z", "to": "2026-09-01T00:00:00.000Z"},
        "schemaVersion": 39, "version": 1, "editable": False, "graphTooltip": 1,
        "links": [{"title": "Operations", "type": "link", "url": "/d/apollo-overview", "icon": "dashboard"},
                  {"title": "RC_2026-08 (August)", "type": "link",
                   "url": "/d/apollo-corpus?var-corpus=real_torrent_2026_08&from=2026-07-31T00:00:00Z&to=2026-09-01T00:00:00Z"},
                  {"title": "Arctic Shift rebuild (Oct–Nov)", "type": "link",
                   "url": "/d/apollo-corpus?var-corpus=real_arctic_2026_10&from=2026-10-22T00:00:00Z&to=2026-11-16T00:00:00Z"}],
        "panels": [tox, rate, bars, table], "templating": {"list": variables}, "annotations": {"list": []},
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, d in [("apollo-overview.json", overview()), ("apollo-corpus.json", corpus())]:
        (OUT / name).write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {name}: {len(d['panels'])} panels")


if __name__ == "__main__":
    main()
