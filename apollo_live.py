"""
Ask Reddit a question now: on-demand queries to Arctic Shift, no dumps.

Arctic Shift (arctic-shift.photon-reddit.com) mirrors Reddit continuously; the
newest comments it serves are typically minutes old. This module talks to its
public API directly, at the moment someone asks, and returns what it finds:

    topic_watch("remote hiring", TOPICS["Remote work & hiring"], hours=1)
        -> every post (or comment) mentioning it in those communities in the
           last hour, with counts per community and per hour.
    latest("news", hours=1)
        -> the newest comments in one community.

Two limits are the service's, and the page says so rather than hiding them:

  * keyword search must name the communities to search (Arctic Shift requires
    a subreddit with a query), so topics come with a preset list that anyone
    can edit. Searching all of Reddit at once needs Reddit's own API, which
    needs a registered app (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET).
  * it rate-limits under load; requests run a few at a time with retries, and
    a community that fails is reported, not silently dropped.

No credentials, nothing stored: results are fetched per request and shown.
Toxicity scores come from the scheduled live ingest (scripts/live_ingest.py),
which loads the transformer; this module only counts and lists.
"""
from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

API = "https://arctic-shift.photon-reddit.com/api"

# Topic presets: the communities where each kind of conversation happens.
TOPICS: dict[str, list[str]] = {
    "Remote work & hiring": ["remotework", "forhire", "RemoteJobs", "WorkOnline", "digitalnomad",
                             "jobs", "cscareerquestions", "freelance", "hiring", "remotejs"],
    "AI & tech": ["technology", "artificial", "MachineLearning", "OpenAI", "singularity", "Futurology"],
    "Markets & money": ["wallstreetbets", "stocks", "investing", "personalfinance", "CryptoCurrency"],
    "News & politics": ["news", "worldnews", "politics", "worldpolitics", "geopolitics"],
    "Gaming": ["gaming", "pcgaming", "Games", "PS5", "NintendoSwitch"],
}


def _get(path: str, params: dict, tries: int = 3) -> tuple[list[dict], str]:
    import requests
    err = ""
    for attempt in range(tries):
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=(10, 40))
            if r.status_code == 200:
                body = r.json()
                if not body.get("error"):
                    return body.get("data") or [], ""
                err = str(body["error"])[:80]
            else:
                err = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001 - flaky upstream; report, don't crash
            err = type(exc).__name__
        time.sleep(1.5 * 2 ** attempt + random.random())
    return [], err


def _row(kind: str, sub: str, d: dict) -> dict:
    ts = int(d.get("created_utc") or 0)
    if kind == "posts":
        text = str(d.get("title") or "")
        link = f"https://redd.it/{d['id']}" if d.get("id") else d.get("url")
    else:
        text = str(d.get("body") or "")
        link = (f"https://www.reddit.com/comments/{str(d.get('link_id', ''))[3:]}/_/{d.get('id')}"
                if d.get("link_id") else None)
    return {"subreddit": f"r/{d.get('subreddit') or sub}", "created_utc": ts,
            "time": datetime.fromtimestamp(ts, timezone.utc), "text": text.strip(),
            "score": d.get("score"), "comments": d.get("num_comments"), "link": link}


def topic_watch(query: str, subreddits: list[str], hours: float = 24, kind: str = "posts",
                per_community: int = 100, workers: int = 4) -> dict:
    """Everything mentioning `query` in `subreddits` in the last `hours`."""
    now = int(time.time())
    after = now - int(hours * 3600)
    fields = ("id,subreddit,created_utc,title,url,score,num_comments" if kind == "posts"
              else "id,subreddit,created_utc,body,link_id,score")
    params_for = lambda s: {"subreddit": s.removeprefix("r/"), "query": query, "after": after,  # noqa: E731
                            "before": now, "limit": per_community, "sort": "desc", "fields": fields}
    rows, failed = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda s: (s, _get(f"{kind}/search", params_for(s))), subreddits))
    for sub, (data, err) in results:
        if err:
            failed[f"r/{sub.removeprefix('r/')}"] = err
        rows += [_row(kind, sub, d) for d in data]
    rows.sort(key=lambda r: -r["created_utc"])
    return {"query": query, "kind": kind, "hours": hours, "asked_at": now,
            "searched": [f"r/{s.removeprefix('r/')}" for s in subreddits],
            "rows": rows, "failed": failed,
            "newest_age_min": round((now - rows[0]["created_utc"]) / 60, 1) if rows else None}


def latest(subreddit: str, hours: float = 1, limit: int = 100) -> dict:
    """The newest comments in one community."""
    now = int(time.time())
    data, err = _get("comments/search", {"subreddit": subreddit.removeprefix("r/"),
                                         "after": now - int(hours * 3600), "before": now,
                                         "limit": limit, "sort": "desc",
                                         "fields": "id,subreddit,created_utc,body,link_id,score"})
    rows = [_row("comments", subreddit, d) for d in data
            if (d.get("body") or "").strip() not in ("", "[deleted]", "[removed]")]
    return {"rows": rows, "error": err, "asked_at": now,
            "newest_age_min": round((now - rows[0]["created_utc"]) / 60, 1) if rows else None}
