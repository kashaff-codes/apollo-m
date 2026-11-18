export type Alert = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

/**
 * One row of `public.community_latest`.
 *
 * These fields must match the view's select list exactly. An earlier version
 * declared a `toxicity_trend` column that the view does not return — and,
 * because the type asserted it existed, the compiler was happy while every row
 * rendered `NaN`. A hand-written type that over-promises is worse than no type,
 * so the safest change here is to keep it in step with
 * `web/supabase/schema.sql` whenever that view changes.
 *
 * Nullable where the pipeline may not have produced a value: a community can be
 * scored before the graph model or the instability pass has run for it.
 */
export type Community = {
  subreddit: string;
  community_health_index: number;
  toxicity_rate: number;
  instability_score: number | null;
  gnn_risk: number | null;
  total_comments: number;
  recommended_action: string;
  alert: Alert;
  updated_at: string;
};

export type WatchlistRow = {
  id: string;
  subreddit: string;
  note: string | null;
  created_at: string;
};

/** Tailwind classes per alert band, for the paper ground. Kept beside the type
 *  so a new band cannot be added without the compiler pointing here. Text uses
 *  a darkened step of each status colour so the label clears 4.5:1. */
export const ALERT_STYLE: Record<Alert, string> = {
  LOW: "bg-[#0ca30c]/10 text-[#0b6b0b] ring-[#0ca30c]/35",
  MEDIUM: "bg-[#fab219]/15 text-[#7a5300] ring-[#fab219]/50",
  HIGH: "bg-[#ec835a]/15 text-[#9a3f17] ring-[#ec835a]/45",
  CRITICAL: "bg-[#d03b3b]/10 text-[#a32727] ring-[#d03b3b]/40",
};

/** The status palette shared with the console (apollo-m/apollo_bands.py). */
export const ALERT_COLOR: Record<Alert, string> = {
  CRITICAL: "#d03b3b",
  HIGH: "#ec835a",
  MEDIUM: "#fab219",
  LOW: "#0ca30c",
};

/** A band is never colour alone: each also has a shape. */
export const ALERT_ICON: Record<Alert, string> = {
  CRITICAL: "✕",
  HIGH: "▲",
  MEDIUM: "◆",
  LOW: "●",
};
