"""
Independent check of filter_arctic_dump.py on the first N lines of the real dump.

The filter uses a regex fast path to find the subreddit without parsing JSON. This script parses EVERY line fully with
json.loads (no fast path) and recomputes, for the same window and the same 39 subreddits:

  * which JSON keys real lines carry, and whether the six fields the filter needs are present and typed as expected,
  * whether the regex's subreddit equals the parsed "subreddit" field on every line (a mismatch would mean the fast path
    silently routes or drops comments),
  * the matched / deleted-or-removed / eligible counts, to compare with the filter's summary,
  * body markers that look like removals but are not in the filter's REMOVED set (reported, never silently changed).

    python scripts/local/torrent_2026_08/verify_filter.py --input RC_2026-08.zst --summary <filter summary.json> \
        --limit-lines 2000000
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from filter_arctic_dump import (DEFAULT_SUBREDDITS, FIELDS, REMOVED, SUB_RE, iter_lines,  # noqa: E402
                                parse_date)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--limit-lines", type=int, required=True)
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--end", default="2026-08-31")
    a = ap.parse_args()

    wanted = {s.lower(): s for s in DEFAULT_SUBREDDITS}
    start_ts, end_ts = parse_date(a.start), parse_date(a.end, True)
    key_sets = Counter()
    types = Counter()
    c = Counter()
    odd_bodies = Counter()
    eligible_by_sub = Counter()
    marker_re = re.compile(r"^\[\s*(removed|deleted)[^\]]*\]$", re.I)

    for n, line in enumerate(iter_lines(a.input), 1):
        if n > a.limit_lines:
            break
        c["lines"] += 1
        obj = json.loads(line)           # full parse of every line: no fast path here
        if n <= 50_000:
            key_sets[tuple(sorted(obj.keys()))] += 1
        sub = obj.get("subreddit")
        m = SUB_RE.search(line)
        rx = m.group(1).decode("utf-8", "replace") if m else None
        if rx != sub:
            c["regex_vs_json_mismatch"] += 1
        canon = wanted.get(str(sub).lower()) if sub is not None else None
        if canon is None:
            continue
        c["matched"] += 1
        for f in FIELDS:
            types[(f, type(obj.get(f)).__name__)] += 1
        if any(f not in obj or obj[f] is None for f in FIELDS):
            c["missing_field"] += 1
            continue
        body = str(obj["body"]).strip()
        if body in REMOVED or str(obj["author"]) == "[deleted]":
            c["deleted_or_removed"] += 1
            continue
        if marker_re.match(body):
            odd_bodies[body] += 1
        ts = int(float(obj["created_utc"]))
        if ts < start_ts or ts > end_ts:
            c["outside_window"] += 1
            continue
        c["eligible"] += 1
        eligible_by_sub[canon] += 1

    s = json.loads(a.summary.read_text(encoding="utf-8"))
    filt_elig = Counter()
    for k, v in s["per_subreddit_day"].items():
        filt_elig[k.split("|")[0]] += v["eligible"]

    print("most common key sets in the first 50,000 lines:")
    for ks, k in key_sets.most_common(3):
        print(f"  {k:,} lines: {list(ks)}")
    print("field types among matched lines:", dict(types))
    print("independent counts:", dict(c))
    print("removal-like bodies NOT in REMOVED (kept by the filter):", dict(odd_bodies.most_common(10)))
    st = s["stats"]
    rows = [
        ("lines_read", st.get("lines_read", 0), c["lines"]),
        ("matched_subreddit", st.get("matched_subreddit", 0), c["matched"]),
        ("missing_field", st.get("missing_field", 0), c["missing_field"]),
        ("deleted_or_removed", st.get("deleted_or_removed", 0), c["deleted_or_removed"]),
        ("outside_window", st.get("outside_window", 0), c["outside_window"]),
        ("eligible (sum over sub-days)", sum(filt_elig.values()), c["eligible"]),
    ]
    print("\nGATE  filter summary vs independent full-JSON recount")
    print(f"  {'count':32s} {'filter':>12s} {'recount':>12s}  ok")
    ok_all = True
    for name, f, r in rows:
        ok = f == r
        ok_all &= ok
        print(f"  {name:32s} {f:>12,} {r:>12,}  {'PASS' if ok else 'FAIL'}")
    sub_ok = all(filt_elig[k] == eligible_by_sub[k] for k in set(filt_elig) | set(eligible_by_sub))
    print(f"  {'per-subreddit eligible equal':32s} {'':>12s} {'':>12s}  {'PASS' if sub_ok else 'FAIL'}")
    mm = c["regex_vs_json_mismatch"] == 0
    print(f"  {'regex subreddit == JSON field':32s} {c['regex_vs_json_mismatch']:>12,} {'':>12s}  "
          f"{'PASS' if mm else 'FAIL'}")
    ok_all &= sub_ok and mm
    print("GATE RESULT:", "PASS" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
