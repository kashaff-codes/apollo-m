"""
Full-pass check that every row of the filtered CSV is the real comment it claims to be.

Streams the whole dump once more. For each line, the comment id is located with a regex on the "id" key (independent of
the filter's subreddit regex); lines whose id is one of the 1,087,806 kept ids are parsed with json.loads, and the
parsed subreddit (case-insensitive), author, body (stripped), score and created_utc are compared with the CSV row via a
digest. Every kept id must be found exactly once and must match.

Memory: one 16-byte digest per kept row, not the rows themselves.

    python scripts/local/torrent_2026_08/verify_kept_rows.py --input <path to RC_2026-08.zst>

(When it was run, the dump path came from config.DUMP; it is now an argument so no local path is committed.
The logic is unchanged.)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import config as C  # noqa: E402
from filter_arctic_dump import iter_lines  # noqa: E402
from gates import Gate  # noqa: E402

ID_RE = re.compile(rb'"id"\s*:\s*"([^"]*)"')
csv.field_size_limit(1 << 30)


def digest(sub: str, author: str, body: str, score, ts) -> bytes:
    key = "\x1f".join([sub.lower(), author, body, str(int(score)), str(int(float(ts)))])
    return hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="path to RC_2026-08.zst")
    dump = ap.parse_args().input
    want: dict[str, bytes] = {}
    with open(C.FILTERED_CSV, encoding="utf-8", newline="") as fh:
        r = csv.reader(fh)
        next(r)
        for cid, sub, author, body, score, ts in r:
            want[cid] = digest(sub, author, body, score, ts)
    print(f"{len(want):,} kept ids loaded", flush=True)
    found = mismatched = dup = 0
    seen: set[str] = set()
    t0 = time.time()
    for n, line in enumerate(iter_lines(dump), 1):
        if n % 25_000_000 == 0:
            print(f"  {n:,} lines, {found:,} found, {time.time() - t0:,.0f}s", flush=True)
        cid = d = None
        for m in ID_RE.finditer(line):            # nested objects (e.g. awards) can carry an "id" key too
            c = m.group(1).decode("utf-8", "replace")
            if c in want:
                cid, d = c, want[c]
                break
        if d is None:
            continue
        obj = json.loads(line)
        if str(obj["id"]) != cid:                 # matched a nested "id", not the comment's own
            continue
        if cid in seen:
            dup += 1
            continue
        seen.add(cid)
        found += 1
        if digest(str(obj["subreddit"]), str(obj["author"]), str(obj["body"]).strip(), obj["score"],
                  obj["created_utc"]) != d:
            mismatched += 1
    g = Gate("full pass: kept rows are the real comments")
    g.check("every kept id found in the dump", found == len(want), f"{found:,} of {len(want):,}")
    g.check("subreddit, author, body, score, created_utc identical", mismatched == 0, f"{mismatched:,} mismatched")
    g.check("no kept id appears twice in the dump", dup == 0, f"{dup:,} repeats")
    print(f"pass took {time.time() - t0:,.0f}s over {n:,} lines")
    g.finish()


if __name__ == "__main__":
    main()
