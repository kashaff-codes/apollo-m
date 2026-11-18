/**
 * Capture the screenshots the report uses.
 *
 *   cd web && node ../docs/screenshots/capture.mjs
 *
 * Needs the web view on :3100 (`npx next dev --port 3100`) and the console on
 * :8501 (`streamlit run dashboard/app.py`). Writes PNGs next to this file.
 */
import { mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Resolved from web/ (the cwd), where Playwright is installed: an ESM import
// would resolve relative to this file, which has no node_modules beside it.
const { chromium } = createRequire(path.join(process.cwd(), "package.json"))(
  "@playwright/test",
);

const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), "2026-11");
mkdirSync(OUT, { recursive: true });
const WEB = process.env.WEB_URL ?? "http://localhost:3100";
const CONSOLE = process.env.CONSOLE_URL ?? "http://localhost:8501";
const shot = (name) => path.join(OUT, `${name}.png`);
const settle = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await chromium.launch();

// ── web view ────────────────────────────────────────────────────────────────
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  await page.goto(`${WEB}/`, { waitUntil: "networkidle" });
  await page.getByText("What we found").waitFor();
  await settle(1500); // entrance animations
  await page.screenshot({ path: shot("web-home"), fullPage: true });
  await page.goto(`${WEB}/login`, { waitUntil: "networkidle" });
  await settle(800);
  await page.screenshot({ path: shot("web-login"), fullPage: true });
  // Signed out, /watchlist redirects to sign-in; the signed-in view needs an account.
  await page.goto(`${WEB}/watchlist`, { waitUntil: "networkidle" });
  await settle(800);
  await page.screenshot({ path: shot("web-watchlist-signed-out"), fullPage: true });
  await page.close();
}

// ── analyst console ─────────────────────────────────────────────────────────
{
  const page = await browser.newPage({ viewport: { width: 1440, height: 2400 }, deviceScaleFactor: 1 });
  await page.goto(CONSOLE, { waitUntil: "networkidle" });
  await page.getByText("Moderation reacts to the worst day").waitFor({ timeout: 60000 });
  await settle(1500);
  await page.screenshot({ path: shot("console-login"), clip: { x: 0, y: 0, width: 1440, height: 900 } });
  await page.getByRole("button", { name: /Sign in as demo/ }).click();
  await page.getByText("Community health").first().waitFor({ timeout: 60000 });

  const pages = [
    ["Overview", "console-overview", null],
    ["Dynamics", "console-dynamics", null],
    ["Communities", "console-community", null],
    ["Forecast", "console-forecast", null],
    ["Across corpora", "console-across-corpora", null],
    ["Evidence", "console-evidence", null],
    ["Clusters", "console-clusters", null],
    ["🔴 Live", "console-live", null],
  ];
  for (const [label, name] of pages) {
    await page.getByRole("radiogroup", { name: "Navigate" }).getByText(label, { exact: true }).click();
    await settle(6000); // rerun + chart render
    await page.screenshot({ path: shot(name) });
    console.log("captured", name);
  }
  await page.close();
}

await browser.close();
console.log("done ->", OUT);
