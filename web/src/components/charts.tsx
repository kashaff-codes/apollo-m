import {
  ALERT_COLOR,
  ALERT_ICON,
  type Alert,
  type Community,
} from "@/lib/types";
import type { DailyPoint } from "@/lib/corpus";

/**
 * Charts as server-rendered SVG.
 *
 * No chart library and no client JavaScript: each chart is markup computed on
 * the server, so it is in the HTML before hydration, works with JavaScript
 * off, and adds nothing to the bundle. Every chart carries a text description
 * (role="img" + aria-label) and sits next to the table that holds its values,
 * so nothing is reachable only by looking at a shape.
 *
 * Alert bands use the status colour AND their glyph (✕ ▲ ◆ ●) AND their name:
 * red and green are hard to tell apart for many colour-blind readers.
 */

const BANDS: [Alert, number, number][] = [
  ["CRITICAL", 0, 65],
  ["HIGH", 65, 75],
  ["MEDIUM", 75, 85],
  ["LOW", 85, 100],
];

/** A community's daily toxicity, last point marked. */
export function Sparkline({
  points,
  width = 112,
  height = 28,
}: {
  points: DailyPoint[] | undefined;
  width?: number;
  height?: number;
}) {
  if (!points || points.length < 2)
    return <span className="text-ink/30">—</span>;
  const ys = points.map((p) => p.toxicity);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const pad = 3;
  const x = (i: number) => pad + (i / (points.length - 1)) * (width - 2 * pad);
  const y = (v: number) =>
    height - pad - ((v - lo) / (hi - lo || 1)) * (height - 2 * pad);
  const d = points
    .map(
      (p, i) =>
        `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.toxicity).toFixed(1)}`,
    )
    .join("");
  const last = points[points.length - 1];
  const first = points[0];
  const label =
    `Daily mean toxicity, ${first.date} to ${last.date}: ` +
    `${first.toxicity.toFixed(3)} to ${last.toxicity.toFixed(3)}, ` +
    `range ${lo.toFixed(3)}–${hi.toFixed(3)}`;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
    >
      <title>{label}</title>
      <path
        d={d}
        fill="none"
        stroke="#2a78d6"
        strokeWidth={1.75}
        strokeLinejoin="round"
      />
      <circle
        cx={x(points.length - 1)}
        cy={y(last.toxicity)}
        r={2.6}
        fill="#2a78d6"
        stroke="#fbf9f6"
        strokeWidth={1.5}
      />
    </svg>
  );
}

/**
 * Every community's Community Health Index on one axis, with the four alert
 * bands drawn behind. Bars grow in on load (CSS, and not at all with
 * prefers-reduced-motion).
 */
export function HealthRanking({ rows }: { rows: Community[] }) {
  const sorted = [...rows].sort(
    (a, b) => a.community_health_index - b.community_health_index,
  );
  const W = 760;
  const left = 168;
  const right = 56;
  const rowH = 22;
  const top = 34;
  const H = top + sorted.length * rowH + 28;
  const lo = Math.max(
    0,
    Math.floor(
      (Math.min(...sorted.map((r) => r.community_health_index)) - 5) / 5,
    ) * 5,
  );
  const hi = 100;
  const x = (v: number) => left + ((v - lo) / (hi - lo)) * (W - left - right);
  const worst = sorted[0];
  const label =
    `Community Health Index for ${sorted.length} communities, lowest first. ` +
    `Lowest: ${worst.subreddit} at ${worst.community_health_index.toFixed(1)}. ` +
    `Bands: critical below 65, high 65 to 75, medium 75 to 85, low 85 and above.`;

  return (
    <figure className="rise rounded-2xl bg-card p-4 ring-1 ring-ink/10">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={label}
      >
        <title>{label}</title>
        {BANDS.map(([band, a, b]) => {
          const x0 = x(Math.max(a, lo));
          const x1 = x(Math.min(b, hi));
          if (x1 <= x0) return null;
          return (
            <g key={band}>
              <rect
                x={x0}
                y={top - 6}
                width={x1 - x0}
                height={sorted.length * rowH + 6}
                fill={ALERT_COLOR[band]}
                opacity={0.07}
              />
              <text
                x={(x0 + x1) / 2}
                y={top - 12}
                textAnchor="middle"
                fontSize={11}
                fill="#2b1d4f"
                fillOpacity={0.75}
                className="font-mono"
              >
                {/* Glyph in the band colour, name in ink: amber text on
                    paper is under 2:1 and unreadable on its own. */}
                <tspan fill={ALERT_COLOR[band]} fillOpacity={1}>
                  {ALERT_ICON[band]}
                </tspan>{" "}
                {band}
              </text>
            </g>
          );
        })}
        {[lo, 65, 75, 85, 100]
          .filter((t) => t >= lo)
          .map((t) => (
            <g key={t}>
              <line
                x1={x(t)}
                x2={x(t)}
                y1={top - 6}
                y2={top + sorted.length * rowH}
                stroke="#2b1d4f"
                strokeOpacity={0.12}
              />
              <text
                x={x(t)}
                y={H - 8}
                textAnchor="middle"
                fontSize={11}
                fill="#2b1d4f"
                fillOpacity={0.55}
                className="font-mono"
              >
                {t}
              </text>
            </g>
          ))}
        {sorted.map((r, i) => {
          const cy = top + i * rowH + rowH / 2;
          const v = r.community_health_index;
          return (
            <g key={r.subreddit}>
              <text
                x={left - 10}
                y={cy + 4}
                textAnchor="end"
                fontSize={12}
                fill="#2b1d4f"
              >
                {r.subreddit}
              </text>
              <rect
                className="grow"
                style={{ animationDelay: `${i * 18}ms` }}
                x={x(lo)}
                y={cy - 5}
                width={x(v) - x(lo)}
                height={10}
                rx={3}
                fill="#2b1d4f"
                fillOpacity={0.14}
              />
              <circle
                cx={x(v)}
                cy={cy}
                r={6.5}
                fill={ALERT_COLOR[r.alert]}
                stroke="#fbf9f6"
                strokeWidth={2}
              />
              <text
                x={x(v)}
                y={cy + 3.5}
                textAnchor="middle"
                fontSize={8}
                fill="#ffffff"
                fontWeight={700}
              >
                {r.alert === "CRITICAL"
                  ? "✕"
                  : r.alert === "HIGH"
                    ? "▲"
                    : r.alert === "MEDIUM"
                      ? "◆"
                      : "●"}
              </text>
              <text
                x={x(v) + 12}
                y={cy + 4}
                fontSize={11}
                fill="#2b1d4f"
                fillOpacity={0.75}
                className="font-mono"
              >
                {v.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption className="mt-2 text-xs text-ink/60">
        Community Health Index, 0–100: 100 minus weighted penalties for
        toxicity, polarisation, author churn and echo-chamber behaviour. Every
        value is also in the table below.
      </figcaption>
    </figure>
  );
}

/** Mean absolute error of three forecasters, as horizontal bars on one scale. */
export function ForecastBars({
  rows,
}: {
  rows: { label: string; tft: number; last: number; mean: number }[];
}) {
  const W = 760;
  const left = 230;
  const groupH = 64;
  const H = rows.length * groupH + 34;
  const max = Math.max(...rows.flatMap((r) => [r.tft, r.last, r.mean])) * 1.12;
  const x = (v: number) => left + (v / max) * (W - left - 70);
  const methods: [keyof (typeof rows)[number], string, string][] = [
    ["tft", "TFT forecaster", "#2a78d6"],
    ["last", "repeat last value", "#eb6834"],
    ["mean", "14-day mean", "#1baf7a"],
  ];
  const label =
    "Forecast error on held-out days, lower is better. " +
    rows
      .map(
        (r) =>
          `${r.label}: TFT ${r.tft.toFixed(3)}, last value ${r.last.toFixed(3)}, 14-day mean ${r.mean.toFixed(3)}`,
      )
      .join("; ");
  return (
    <figure className="rise rounded-2xl bg-card p-4 ring-1 ring-ink/10">
      <div className="mb-2 flex flex-wrap gap-4 text-xs text-ink/70">
        {methods.map(([, name, color]) => (
          <span key={name} className="inline-flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rounded-sm"
              style={{ background: color }}
            />
            {name}
          </span>
        ))}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={label}
      >
        <title>{label}</title>
        {rows.map((r, g) => (
          <g key={r.label}>
            <text
              x={left - 12}
              y={g * groupH + 30}
              textAnchor="end"
              fontSize={12}
              fill="#2b1d4f"
            >
              {r.label}
            </text>
            {methods.map(([k, name, color], j) => {
              const v = r[k] as number;
              const y = g * groupH + 8 + j * 16;
              return (
                <g key={name}>
                  <rect
                    className="grow"
                    style={{ animationDelay: `${(g * 3 + j) * 40}ms` }}
                    x={left}
                    y={y}
                    width={x(v) - left}
                    height={12}
                    rx={3}
                    fill={color}
                  />
                  <text
                    x={x(v) + 6}
                    y={y + 10}
                    fontSize={11}
                    fill="#2b1d4f"
                    fillOpacity={0.8}
                    className="font-mono"
                  >
                    {v.toFixed(4)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <figcaption className="mt-2 text-xs text-ink/60">
        Mean absolute error on days the model never trained on. Lower is better.
      </figcaption>
    </figure>
  );
}
