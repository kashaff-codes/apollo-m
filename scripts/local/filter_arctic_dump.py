"""
Filter an Arctic Shift monthly comments dump (RC_YYYY-MM.zst) down to the subreddits APOLLO-M uses.

Why this exists: the monthly dump holds every public Reddit comment of the month (~50 GB compressed). We only need our
communities, with a bounded number of comments per community per day so the output stays small and the daily series is even.

What it does, in one streaming pass (the file is never unpacked to disk):
  1. reads the .zst line by line (each line is one JSON comment),
  2. quickly skips lines whose subreddit is not on our list (no JSON parse for those, which is what makes it fast enough),
  3. drops deleted/removed/empty comments and comments outside the date window,
  4. keeps a seeded random sample of at most --cap per subreddit per UTC day (reservoir sampling),
  5. writes a CSV with id, subreddit, author, body, score, created_utc, and a JSON summary of what was seen and kept.

The summary records, for every subreddit-day, how many eligible comments existed BEFORE the cap, so nothing about volume is lost
and you can see exactly which days hit the cap.

Usage (Windows, from any folder):
    pip install zstandard
    python filter_arctic_dump.py --input <RC_ZST> ^
        --output <OUTPUT_DIR>\\RC_2026-08_filtered.csv --cap 1000

Options:
    --subreddits-file FILE   one subreddit per line (default: the 39 communities in DEFAULT_SUBREDDITS below)
    --start 2026-08-01 --end 2026-08-31   inclusive UTC dates (default: no limit)
    --cap N                  max comments per subreddit per day (default 1000; 0 = keep everything)
    --seed N                 sampling seed (default 42; keep it fixed and record it)
    --limit-lines N          stop after N lines, for a quick test
    --hash                   also record the SHA-256 of the input file in the summary (slow on 50 GB)

Notes:
  * Keep the output out of git (it contains usernames).
  * The dump's exact JSON field names are the standard Reddit ones (id, subreddit, author, body, score, created_utc).
    Lines missing any of them are counted in the summary and skipped; they are never invented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import zstandard
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: run  pip install zstandard")

# The 39 communities present in the scored corpus supplied by Aaqib (outputs/toxicity_report.csv). Change with --subreddits-file.
DEFAULT_SUBREDDITS = """
AmItheAsshole AskReddit Bitcoin China Conservative CryptoCurrency MensRights PublicFreakout RoastMe SuicideWatch Superstonk
TwoXChromosomes antiwork aww conspiracy depression funny gaming geopolitics gonewild leagueoflegends mildlyinfuriating movies nba
news nfl politics relationship_advice science soccer technology teenagers tifu todayilearned ukraine unpopularopinion
wallstreetbets worldnews worldpolitics
""".split()

SUB_RE = re.compile(rb'"subreddit"\s*:\s*"([^"]*)"')
REMOVED = {"[deleted]", "[removed]", "", "[ Removed by Reddit ]"}  # last one: real RC_2026-08 marker, seen in testing
FIELDS = ("id", "subreddit", "author", "body", "score", "created_utc")


def parse_date(s: str | None, end: bool = False) -> int | None:
    if not s:
        return None
    d = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    ts = int(d.timestamp())
    return ts + 86400 - 1 if end else ts


def iter_lines(path: Path):
    """Stream decompressed lines. Dumps use a large window, so allow it."""
    dctx = zstandard.ZstdDecompressor(max_window_size=2**31)
    with open(path, "rb") as fh, dctx.stream_reader(fh) as reader:
        text = io.BufferedReader(reader, buffer_size=1 << 20)
        for line in text:
            yield line


def filter_dump(input_path: Path, output_path: Path, subreddits: list[str], cap: int, seed: int,
                start_ts: int | None, end_ts: int | None, limit_lines: int | None = None,
                do_hash: bool = False, progress_every: int = 5_000_000) -> dict:
    wanted = {s.lower(): s for s in subreddits}
    rng = random.Random(seed)
    reservoirs: dict[tuple[str, str], list] = defaultdict(list)
    eligible: Counter = Counter()          # (sub, day) -> eligible comments before cap
    stats = Counter()
    seen_ids: set[str] = set()
    t0 = time.time()

    for n, line in enumerate(iter_lines(input_path), 1):
        if limit_lines and n > limit_lines:
            break
        stats["lines_read"] += 1
        if progress_every and n % progress_every == 0:
            print(f"  {n:,} lines read, {stats['matched_subreddit']:,} in our subreddits, "
                  f"{time.time() - t0:,.0f}s elapsed", flush=True)
        m = SUB_RE.search(line)
        if not m:
            stats["no_subreddit_field"] += 1
            continue
        canon = wanted.get(m.group(1).decode("utf-8", "replace").lower())
        if canon is None:
            continue
        stats["matched_subreddit"] += 1
        try:
            obj = json.loads(line)
        except ValueError:
            stats["bad_json"] += 1
            continue
        if any(f not in obj or obj[f] is None for f in FIELDS):
            stats["missing_field"] += 1
            continue
        body = str(obj["body"]).strip()
        if body in REMOVED or str(obj["author"]) in ("[deleted]",):
            stats["deleted_or_removed"] += 1
            continue
        try:
            ts = int(float(obj["created_utc"]))
        except (TypeError, ValueError):
            stats["bad_timestamp"] += 1
            continue
        if (start_ts is not None and ts < start_ts) or (end_ts is not None and ts > end_ts):
            stats["outside_window"] += 1
            continue
        cid = str(obj["id"])
        if cid in seen_ids:
            stats["duplicate_id"] += 1
            continue
        seen_ids.add(cid)
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        key = (canon, day)
        eligible[key] += 1
        rec = (cid, canon, str(obj["author"]), body, obj["score"], ts)
        res = reservoirs[key]
        if cap <= 0 or len(res) < cap:
            res.append(rec)
        else:
            j = rng.randrange(eligible[key])
            if j < cap:
                res[j] = rec

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for key in sorted(reservoirs):
            for rec in sorted(reservoirs[key], key=lambda r: r[5]):
                w.writerow(rec)
                kept += 1

    per_sub_day = {f"{s}|{d}": {"eligible": eligible[(s, d)], "kept": len(reservoirs[(s, d)]),
                                "hit_cap": bool(cap > 0 and eligible[(s, d)] > cap)}
                   for (s, d) in sorted(reservoirs)}
    subs_kept = Counter()
    for (s, _d), res in reservoirs.items():
        subs_kept[s] += len(res)
    days = sorted({d for (_s, d) in reservoirs})
    summary = {
        "input": str(input_path), "output": str(output_path), "seed": seed, "cap_per_sub_per_day": cap,
        "window": [start_ts, end_ts], "stats": dict(stats), "comments_kept": kept,
        "date_range_kept": [days[0], days[-1]] if days else None,
        "subreddits_requested": len(wanted), "subreddits_with_data": len(subs_kept),
        "subreddits_without_data": sorted(set(subreddits) - set(subs_kept)),
        "kept_per_subreddit": dict(sorted(subs_kept.items())),
        "days_hitting_cap": sum(v["hit_cap"] for v in per_sub_day.values()),
        "per_subreddit_day": per_sub_day,
        "seconds": round(time.time() - t0, 1),
    }
    if do_hash:
        h = hashlib.sha256()
        with open(input_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        summary["input_sha256"] = h.hexdigest()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--subreddits-file", type=Path)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit-lines", type=int)
    ap.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    subs = DEFAULT_SUBREDDITS
    if a.subreddits_file:
        subs = [l.strip() for l in a.subreddits_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = filter_dump(a.input, a.output, subs, a.cap, a.seed, parse_date(a.start), parse_date(a.end, True),
                          a.limit_lines, a.hash)
    sp = a.output.with_suffix(".summary.json")
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done: {summary['comments_kept']:,} comments kept from {summary['stats'].get('lines_read', 0):,} lines "
          f"in {summary['seconds']}s. {summary['subreddits_with_data']}/{summary['subreddits_requested']} subreddits had data.")
    print(f"Summary: {sp}")


if __name__ == "__main__":
    main()
