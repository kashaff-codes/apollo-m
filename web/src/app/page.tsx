import { Suspense } from "react";
import { createClient } from "@/lib/supabase/server";
import { isConfigured } from "@/lib/supabase/env";
import { getCommunities } from "@/lib/data";
import { getDailySeries, getEvidence, type Evidence } from "@/lib/corpus";
import { addToWatchlist } from "./auth/actions";
import { ALERT_ICON, ALERT_STYLE, type Community } from "@/lib/types";
import { ForecastBars, HealthRanking, Sparkline } from "@/components/charts";

/**
 * The streaming route.
 *
 * The shell — heading, intro, section frames — is sent immediately. Each
 * data-dependent section sits behind its own Suspense boundary, so the server
 * flushes it as soon as its read finishes instead of holding the document for
 * the slowest. The community rows come from Supabase Postgres; the daily series
 * and findings come from the committed pipeline outputs (lib/corpus.ts). All of
 * it is read on the server and rendered as HTML and SVG: no data-fetching or
 * charting JavaScript is shipped to the browser.
 */

function Pill({ alert }: { alert: Community["alert"] }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium ring-1 ${ALERT_STYLE[alert]}`}
    >
      <span aria-hidden>{ALERT_ICON[alert]}</span>
      {alert}
    </span>
  );
}

function Strip({ items }: { items: [string, string][] }) {
  return (
    <p className="inline-flex flex-wrap items-center gap-x-4 gap-y-1 rounded-full bg-card px-4 py-1.5 font-mono text-xs text-ink/60 ring-1 ring-ink/10">
      <span
        aria-hidden
        className="pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-amber"
      />
      {items.map(([k, v]) => (
        <span key={k}>
          <b className="font-medium text-ink">{v}</b> {k}
        </span>
      ))}
    </p>
  );
}

async function Summary() {
  // Same cached read as the sections below: within a render they resolve from
  // one cache entry, so calling it three times costs one query.
  const result = await getCommunities();
  if (!result.ok)
    return <SetupHint message={result.reason} hint={result.hint} />;
  const rows = result.rows;
  const comments = rows.reduce((n, r) => n + (r.total_comments ?? 0), 0);
  const critical = rows.filter((r) => r.alert === "CRITICAL").length;
  const high = rows.filter((r) => r.alert === "HIGH").length;
  const updated =
    rows
      .map((r) => r.updated_at)
      .sort()
      .at(-1)
      ?.slice(0, 10) ?? "—";

  const cards = [
    {
      label: "communities",
      value: rows.length.toLocaleString(),
      sub: "real subreddits",
    },
    {
      label: "comments scored",
      value: comments.toLocaleString(),
      sub: "by unitary/toxic-bert",
    },
    { label: "critical", value: String(critical), sub: "CHI below 65" },
    { label: "high", value: String(high), sub: "CHI 65–75" },
  ];

  return (
    <div className="space-y-5">
      <Strip
        items={[
          [`comments read`, comments.toLocaleString()],
          ["communities", String(rows.length)],
          ["need attention", String(critical + high)],
          ["updated", updated],
        ]}
      />
      <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {cards.map((c, i) => (
          <div
            key={c.label}
            className="rise rounded-2xl bg-card p-5 ring-1 ring-ink/10"
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <dd className="text-3xl font-semibold tracking-tight">{c.value}</dd>
            <dt className="mt-1 font-mono text-[11px] uppercase tracking-wider text-ink/60">
              {c.label}
            </dt>
            <p className="mt-1 text-xs text-ink/60">{c.sub}</p>
          </div>
        ))}
      </dl>
    </div>
  );
}

async function Ranking() {
  const result = await getCommunities();
  if (!result.ok || !result.rows.length) return null;
  return <HealthRanking rows={result.rows} />;
}

/**
 * The signed-in visitor's watched subreddits, or none.
 *
 * Guarded on configuration because constructing the Supabase client throws
 * when the environment is empty, and an unconfigured clone must still render —
 * CI runs with no Supabase secrets on purpose, so this is the path it exercises.
 * Returns [] when signed out without branching on the session: the select
 * policy scopes rows to auth.uid(), so the database answers that question.
 */
async function watchedSubreddits(): Promise<string[]> {
  if (!isConfigured) return [];
  const supabase = await createClient();
  const { data } = await supabase.from("watchlist").select("subreddit");
  return (data ?? []).map((w) => w.subreddit);
}

async function CommunityTable() {
  // Different lifetimes on purpose: community rows and the daily series are the
  // same for everyone and cached; the watchlist is this visitor's own and is
  // never cached, so it keeps the request-scoped client.
  const [result, watched, series] = await Promise.all([
    getCommunities(),
    watchedSubreddits(),
    getDailySeries(),
  ]);

  if (!result.ok)
    return <SetupHint message={result.reason} hint={result.hint} />;
  const data = result.rows;
  if (!data.length)
    return (
      <SetupHint
        message="No rows yet."
        hint="The project and table exist but are empty. Run database/db_setup.py against this project."
      />
    );

  const onList = new Set(watched);

  return (
    <div className="overflow-x-auto rounded-2xl bg-card ring-1 ring-ink/10">
      <table className="w-full text-sm">
        <thead className="border-b border-ink/10 text-left font-mono text-[11px] uppercase tracking-wider text-ink/60">
          <tr>
            <th scope="col" className="px-4 py-3 font-medium">
              Community
            </th>
            <th scope="col" className="px-4 py-3 text-right font-medium">
              Health
            </th>
            <th scope="col" className="px-4 py-3 text-right font-medium">
              Toxicity
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Daily toxicity, 41 days
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-right font-medium"
              title="Trend-aware instability: how fast a community is deteriorating, not how bad it is now"
            >
              Instability
            </th>
            <th scope="col" className="px-4 py-3 font-medium">
              Alert
            </th>
            <th scope="col" className="px-4 py-3" />
          </tr>
        </thead>
        <tbody className="divide-y divide-ink/[0.07]">
          {data.map((c) => (
            <tr key={c.subreddit} className="hover:bg-violet-soft/40">
              <td className="px-4 py-2.5 font-medium">{c.subreddit}</td>
              <td className="px-4 py-2.5 text-right font-mono tabular-nums">
                {Number(c.community_health_index).toFixed(1)}
              </td>
              <td className="px-4 py-2.5 text-right font-mono tabular-nums text-ink/70">
                {(Number(c.toxicity_rate) * 100).toFixed(1)}%
              </td>
              <td className="px-4 py-1">
                <Sparkline points={series?.[c.subreddit]} />
              </td>
              <td className="px-4 py-2.5 text-right font-mono tabular-nums text-ink/70">
                {c.instability_score == null
                  ? "—"
                  : Number(c.instability_score).toFixed(4)}
              </td>
              <td className="px-4 py-2.5">
                <Pill alert={c.alert} />
              </td>
              <td className="px-4 py-2.5 text-right">
                {onList.has(c.subreddit) ? (
                  <span className="font-mono text-xs text-ink/50">
                    watching
                  </span>
                ) : (
                  <form action={addToWatchlist}>
                    <input type="hidden" name="subreddit" value={c.subreddit} />
                    <button className="text-xs font-medium text-violet hover:underline">
                      + watch
                    </button>
                  </form>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const SHORT: Record<string, string> = {
  benchmark: "Benchmark (simulation)",
  real_api_2026_07: "API corpus, Jul–Aug",
  real_torrent_2026_08: "August dump",
  real_arctic_2026_10: "Rebuild, Oct–Nov",
};

function Stat({
  value,
  label,
  note,
}: {
  value: string;
  label: string;
  note: string;
}) {
  return (
    <div className="rise rounded-2xl bg-card p-6 ring-1 ring-ink/10">
      <p className="text-4xl font-semibold tracking-tight">{value}</p>
      <p className="mt-2 font-mono text-[11px] uppercase tracking-wider text-ink/60">
        {label}
      </p>
      <p className="mt-3 text-sm leading-relaxed text-ink/70">{note}</p>
    </div>
  );
}

function FindingsBody({ e }: { e: Evidence }) {
  const real = e.corpora.filter((c) => c.kind === "real");
  // Computed, not written by hand: an earlier draft said the TFT "beats
  // repeating yesterday", which is false on the August dump.
  const lastWins = real.filter(
    (c) =>
      e.backtest[c.key] &&
      e.backtest[c.key].mae_tft < e.backtest[c.key].mae_last_value,
  );
  const meanWins = real.filter(
    (c) =>
      e.backtest[c.key] &&
      e.backtest[c.key].mae_recent_mean < e.backtest[c.key].mae_tft,
  );
  const best = [...e.agreement].sort((a, b) => b.n - a.n)[0];
  const la = e.label_agreement;
  const bars = e.corpora
    .filter((c) => e.backtest[c.key])
    .map((c) => ({
      label: SHORT[c.key] ?? c.label,
      tft: e.backtest[c.key].mae_tft,
      last: e.backtest[c.key].mae_last_value,
      mean: e.backtest[c.key].mae_recent_mean,
    }));

  return (
    <div className="space-y-8">
      <div className="grid gap-4 md:grid-cols-3">
        <Stat
          value={`ρ ${best.rho.toFixed(3)}`}
          label={`rank agreement, ${best.n} communities`}
          note="Measured again months later, by a different collection method, communities rank almost identically. Which communities are toxic is a stable property, not an artefact of one sample."
        />
        <Stat
          value={`${meanWins.length} of ${real.length}`}
          label="real corpora where a 14-day mean wins"
          note={`On days it never trained on, the TFT loses to a simple recent average on ${meanWins.length} of ${real.length} real corpora, and beats merely repeating yesterday on ${lastWins.length} of ${real.length}. On the benchmark, where trends were planted, it wins. We publish both.`}
        />
        <Stat
          value={la.precision.toFixed(2)}
          label={`toxic-bert precision, ${la.n} labelled comments`}
          note={`Most comments the model calls toxic are not, by the labelling guidelines: ${la.false_positives_with_profanity} of its ${la.false_positives} false positives contain profanity. It largely detects swearing.`}
        />
      </div>
      <ForecastBars rows={bars} />
      <div>
        <h3 className="font-serif text-2xl tracking-tight">
          Four corpora, one pipeline
        </h3>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {e.corpora.map((c) => (
            <div
              key={c.key}
              className="rounded-2xl bg-card p-5 ring-1 ring-ink/10"
            >
              <p className="font-mono text-[11px] uppercase tracking-wider text-ink/60">
                {c.kind === "real" ? "Real Reddit" : "Declared simulation"} ·{" "}
                {c.window}
              </p>
              <p className="mt-1 font-serif text-xl">
                {c.label.replace(/^Real · /, "")}
              </p>
              <p className="mt-2 text-sm text-ink/70">{c.summary}</p>
              <p className="mt-3 font-mono text-xs text-ink/60">
                {c.communities} communities · {c.comments.toLocaleString()}{" "}
                comments · by {c.credit}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

async function Findings() {
  const e = await getEvidence();
  if (!e)
    return (
      <p className="rounded-2xl bg-card p-6 text-sm text-ink/70 ring-1 ring-ink/10">
        The findings could not be loaded just now. They are committed in the
        repository as outputs/web_evidence.json.
      </p>
    );
  return <FindingsBody e={e} />;
}

/**
 * The degraded state, with the remedy that matches the failure.
 *
 * `hint` is chosen by describeFailure() from the driver's own error rather than
 * hardcoded here, because the fix for an unreachable host and the fix for a
 * missing table have nothing in common.
 */
function SetupHint({ message, hint }: { message: string; hint?: string }) {
  return (
    <div className="rounded-2xl bg-amber-soft p-5 text-sm text-ink ring-1 ring-amber/50">
      <p className="font-medium">Supabase is not answering yet.</p>
      <p className="mt-1 text-ink/80">{message}</p>
      {hint ? <p className="mt-2 text-ink/80">{hint}</p> : null}
    </div>
  );
}

function Skeleton({ rows }: { rows: number }) {
  return (
    <div className="space-y-2" aria-hidden>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-10 animate-pulse rounded-lg bg-ink/5" />
      ))}
    </div>
  );
}

export default function Home() {
  return (
    <div className="space-y-16">
      <section className="max-w-3xl">
        <h1 className="font-serif text-5xl leading-[1.02] tracking-tight sm:text-6xl">
          Community instability, measured on real comments.
        </h1>
        <p className="mt-6 max-w-2xl text-lg leading-relaxed text-ink/70">
          The same rows the Streamlit dashboard and the REST API serve: real
          subreddits, worst first, scored by the APOLLO-M pipeline and read here
          on the server from Supabase Postgres. Sign in to keep a watchlist;
          those rows are private, enforced by row-level security rather than by
          this app.
        </p>
      </section>

      {/* Each boundary streams independently: the summary lands as soon as its
          aggregate returns, without waiting for the table or the findings. */}
      <Suspense fallback={<Skeleton rows={2} />}>
        <Summary />
      </Suspense>

      <section className="space-y-4">
        <h2 className="font-serif text-3xl tracking-tight">
          Every community, on one scale
        </h2>
        <Suspense fallback={<Skeleton rows={6} />}>
          <Ranking />
        </Suspense>
      </section>

      <section className="space-y-4">
        <h2 className="font-serif text-3xl tracking-tight">
          The numbers behind it
        </h2>
        <Suspense fallback={<Skeleton rows={8} />}>
          <CommunityTable />
        </Suspense>
      </section>

      <section id="findings" className="scroll-mt-20 space-y-4">
        <h2 className="font-serif text-3xl tracking-tight">
          What we found, including what did not work
        </h2>
        <Suspense fallback={<Skeleton rows={4} />}>
          <Findings />
        </Suspense>
      </section>
    </div>
  );
}
