import { Suspense } from "react";
import type { Metadata } from "next";
import Link from "next/link";
import { Fraunces, Geist, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { createClient } from "@/lib/supabase/server";
import { signOut } from "./auth/actions";

// The Occuris type system (occuris.pages.dev): a display serif for titles, a
// neutral sans for reading, a mono for data. Self-hosted by next/font, so no
// request goes to Google at runtime and nothing shifts while fonts load.
const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-fraunces",
  display: "swap",
});
const geist = Geist({
  subsets: ["latin"],
  variable: "--font-geist",
  display: "swap",
});
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export const metadata: Metadata = {
  title: "APOLLO-M — Community instability",
  description:
    "Toxicity, community health and five-day forecasts for Reddit communities, measured on real comments.",
};

/**
 * The only part of the chrome that depends on who is asking.
 *
 * Reading the session touches cookies, which makes this scope request-bound.
 * Isolating it is what lets everything around it — the whole nav frame, the
 * page shell, the footer — be prerendered once and reused, while this streams
 * in per visitor.
 *
 * Still a Server Component: the session is resolved on the server, so the
 * header arrives correct rather than flickering signed-out to signed-in on
 * hydration.
 */
async function SessionControls() {
  // A missing or wrong Supabase config should degrade to the signed-out header,
  // not a 500 on every route. The page body reports the real problem.
  let user = null;
  try {
    const supabase = await createClient();
    ({
      data: { user },
    } = await supabase.auth.getUser());
  } catch {
    user = null;
  }

  return user ? (
    <>
      <span className="hidden font-mono text-xs text-ink/60 sm:inline">
        {user.email}
      </span>
      <form action={signOut}>
        <button className="rounded-lg px-3 py-1.5 ring-1 ring-ink/20 hover:bg-ink/5">
          Sign out
        </button>
      </form>
    </>
  ) : (
    <Link
      href="/login"
      className="rounded-lg bg-violet px-4 py-2 font-medium text-white hover:bg-violet-deep"
    >
      Sign in
    </Link>
  );
}

function Nav() {
  return (
    <header className="border-b border-ink/10 bg-paper/90 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl items-center gap-7 px-6 py-4 text-[15px]">
        <Link href="/" className="flex items-center gap-2 font-medium">
          <span
            aria-hidden
            className="inline-block h-4 w-4 rounded-full border-2 border-ink"
          />
          APOLLO-M
        </Link>
        <Link href="/" className="text-ink/70 hover:text-ink">
          Communities
        </Link>
        <Link href="/#findings" className="text-ink/70 hover:text-ink">
          Findings
        </Link>
        <Link href="/watchlist" className="text-ink/70 hover:text-ink">
          Watchlist
        </Link>
        <div className="ml-auto flex items-center gap-3 text-sm">
          {/* Reserves the button's height so the nav does not shift when the
              session resolves. */}
          <Suspense
            fallback={
              <span
                className="h-[36px] w-20 rounded-lg bg-ink/5"
                aria-hidden="true"
              />
            }
          >
            <SessionControls />
          </Suspense>
        </div>
      </nav>
    </header>
  );
}

const LINKS: [string, string][] = [
  ["Analyst console", "https://apollo-m.streamlit.app"],
  ["Monitoring (Grafana)", "https://apollo-m-monitoring.onrender.com"],
  ["REST API", "https://apollo-api-tllm.onrender.com/docs"],
  ["CEREBRO", "https://cerebro-sandy-beta.vercel.app"],
  ["Source", "https://github.com/kashaff-codes/apollo-m"],
];

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${geist.variable} ${mono.variable}`}
    >
      {/* Background and text colour come from globals.css, which owns the
          palette — setting them here as well gave two sources of truth. */}
      <body className="min-h-screen antialiased">
        <Nav />
        <main className="mx-auto max-w-6xl px-6 py-12">{children}</main>
        <footer className="mx-auto max-w-6xl border-t border-ink/10 px-6 py-10 text-sm text-ink/70">
          <p className="max-w-2xl">
            One of three surfaces on the same APOLLO-M data. This one is Next.js
            App Router with React Server Components, reading Supabase Postgres
            through row-level security. Final Year Project, Bahria University
            Karachi.
          </p>
          <p className="mt-4 flex flex-wrap gap-x-6 gap-y-2">
            {LINKS.map(([label, href]) => (
              <a
                key={href}
                className="text-violet hover:underline"
                href={href}
                target="_blank"
                rel="noreferrer"
              >
                {label} ↗
              </a>
            ))}
          </p>
        </footer>
      </body>
    </html>
  );
}
