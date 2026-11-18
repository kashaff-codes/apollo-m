# Restoring the shared PostgreSQL

## What happened

APOLLO-M's API and CEREBRO's API shared one PostgreSQL on Render's free tier
(`apollo` and `cerebro` schemas in the same instance). Render deletes free
databases when their free period ends, and this one is gone — its hostname
`dpg-d9lsstbm8hqs739dmcj0-a` no longer resolves.

What still works without it, by design (November changes):

| | Without a database |
|---|---|
| APOLLO-M API | Serves the committed outputs of the same corpus; responses say `"source": "csv:…"`; `/metrics` reports `apollo_database_up 0` |
| CEREBRO API | Email, news and network analysis work; news retrieves from the curated seed corpus in memory; the demo account signs in (offline mode); history, metrics and new accounts return 503 with an explanation |
| Streamlit console, public web view, Grafana | Unaffected — they never depended on it (the web view uses Supabase) |

Restoring it brings back CEREBRO's history, dashboards, registration and the
event stream, and Apollo's database-backed responses.

## 1. Create a database that will not expire

Pick one. All three support `pgvector`, which CEREBRO's schema requires.

| Host | Free tier | Notes |
|---|---|---|
| **Supabase** (recommended) | does not expire; pauses after 7 days idle and wakes on the next connection | The team already has a Supabase project for the web view. Use the **session pooler** connection string (port **5432**, IPv4). Not the transaction pooler (6543): asyncpg's prepared statements break under transaction pooling — the subject of Kashaf's supavisor#1051 report. |
| Neon | does not expire; scales to zero | Use the pooled connection string. |
| Render | expires again | Only if you accept repeating this. |

Copy the connection string; it looks like
`postgresql://USER:PASSWORD@HOST:5432/postgres`. It is a secret: paste it only
into host dashboards, never into the repository.

## 2. CEREBRO — set three variables, redeploy

Render → `cerebro-api` → Environment:

| Key | Value |
|---|---|
| `DATABASE_URL` | the connection string |
| `AUTO_APPLY_SCHEMA` | `true` — applies `backend/db/schema_portable.sql` on boot (idempotent) |
| `SEED_DEMO_USER` | `true` — creates `demo@cerebro.app` if no users exist, plus the demo evidence corpus |

Save; Render redeploys. Check:

```bash
curl https://cerebro-api-nmah.onrender.com/ready        # "database": true
```

and sign in at https://cerebro-sandy-beta.vercel.app as `demo@cerebro.app` /
`cerebro-demo-2026` — the display name no longer says "offline mode".

## 3. APOLLO-M — same database, its own schema

Render → `apollo-api` → Environment: set `DATABASE_URL` to the same string and
`DB_SCHEMA` to `apollo` (already in `render.yaml`). Then load the tables from a
machine with this repository:

```bash
DATABASE_URL="postgresql://…" DB_SCHEMA=apollo python database/db_setup.py
```

`db_setup.py` creates the `apollo` schema, applies `database/schema.sql` and loads
the API corpus (`outputs/real/`, the same rows the web view shows). Check:

```bash
curl https://apollo-api-tllm.onrender.com/health        # "data_source": "database"
```

In Grafana, *APOLLO-M database* turns from "▲ serving committed outputs" to
"● connected" within a minute.

## If you used Supabase for both

That project then holds the web view's `public` tables, `apollo` and `cerebro` —
one PostgreSQL behind all three systems, which is the shared-database
integration the technical report describes. The web view's row-level security is
per schema, so nothing it exposes changes.
