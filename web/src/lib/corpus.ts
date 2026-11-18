import { cacheLife, cacheTag } from "next/cache";

import { COMMUNITIES_TAG } from "@/lib/data";

/**
 * Pipeline outputs that are committed to the repository rather than loaded
 * into Supabase: the daily series behind each sparkline, and the findings
 * (outputs/web_evidence.json, written by scripts/export_web_evidence.py from
 * the same functions the analyst console uses).
 *
 * Read from GitHub and cached for minutes, not hours: a failed read returns
 * null, and a cached null keeps the section empty for the whole cache life.
 * That happened in development (GitHub's CDN lags a push by a minute or so)
 * and it is the same trap that left the live site on an error for an hour in
 * September. These reads hit a CDN and cost almost nothing, so a short life is
 * the right trade. Tagged like the community rows, so /api/revalidate clears
 * both together.
 *
 * Two rules from building this on Cache Components:
 *  - Failures are returned (null), never thrown. A rejection inside a
 *    `use cache` scope is reported as a prerender error even when the caller
 *    catches it, so a throw here would fail the build whenever GitHub blips.
 *  - Every fetch has a timeout. A fetch that hangs during prerender stalls the
 *    build until it times out ("Filling a cache during prerender timed out").
 */

const RAW =
  "https://raw.githubusercontent.com/kashaff-codes/apollo-m/main/outputs";

async function getText(path: string): Promise<string | null> {
  try {
    const res = await fetch(`${RAW}/${path}`, {
      signal: AbortSignal.timeout(8000),
    });
    return res.ok ? await res.text() : null;
  } catch {
    return null;
  }
}

export type DailyPoint = { date: string; toxicity: number };

/**
 * Daily mean toxicity per community, for the corpus the Supabase rows come
 * from (the Arctic Shift API corpus, outputs/real_reddit_daily.csv). Keys are
 * "r/<name>", matching community_latest.subreddit.
 */
export async function getDailySeries(): Promise<Record<
  string,
  DailyPoint[]
> | null> {
  "use cache";
  cacheLife("minutes");
  cacheTag(COMMUNITIES_TAG);

  const csv = await getText("real_reddit_daily.csv");
  if (!csv) return null;
  const [header, ...lines] = csv.trim().split(/\r?\n/);
  const cols = header.split(",");
  const iSub = cols.indexOf("subreddit");
  const iDate = cols.indexOf("date");
  const iTox = cols.indexOf("avg_toxicity");
  if (iSub < 0 || iDate < 0 || iTox < 0) return null;

  const out: Record<string, DailyPoint[]> = {};
  for (const line of lines) {
    const f = line.split(",");
    const value = Number(f[iTox]);
    if (!Number.isFinite(value)) continue; // a missing day stays missing
    const name = f[iSub].startsWith("r/") ? f[iSub] : `r/${f[iSub]}`;
    (out[name] ??= []).push({ date: f[iDate].slice(0, 10), toxicity: value });
  }
  for (const k of Object.keys(out))
    out[k].sort((a, b) => a.date.localeCompare(b.date));
  return out;
}

export type Corpus = {
  key: string;
  label: string;
  kind: "benchmark" | "real";
  window: string;
  communities: number;
  comments: number;
  days: number;
  forecast: "future" | "held_out" | "hindcast";
  credit: string;
  summary: string;
};

export type Backtest = {
  communities: number;
  days: number;
  mae_tft: number;
  mae_last_value: number;
  mae_recent_mean: number;
  coverage: number;
  beats_last: number;
  beats_mean: number;
};

export type Evidence = {
  corpora: Corpus[];
  agreement: { a: string; b: string; rho: number; n: number }[];
  backtest: Record<string, Backtest>;
  label_agreement: {
    n: number;
    precision: number;
    recall: number;
    f1: number;
    precision_ci95: number[];
    false_positives: number;
    false_positives_with_profanity: number;
  };
  benchmark: {
    chi_roc_auc: number;
    toxicity_only_roc_auc: number;
    tft_slope_roc_auc: number;
  };
  forward_test: Record<string, unknown> | null;
};

export async function getEvidence(): Promise<Evidence | null> {
  "use cache";
  cacheLife("minutes");
  cacheTag(COMMUNITIES_TAG);

  const text = await getText("web_evidence.json");
  if (!text) return null;
  try {
    return JSON.parse(text) as Evidence;
  } catch {
    return null;
  }
}
