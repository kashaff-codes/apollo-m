import { NextResponse, type NextRequest } from "next/server";
import type { EmailOtpType } from "@supabase/supabase-js";
import { createClient } from "@/lib/supabase/server";

/**
 * Where links from Supabase land: the email confirmation link, a resent
 * confirmation, and the return leg of "Continue with Google".
 *
 * Before this route existed a confirmation link had nowhere to go, so a new
 * account stayed unconfirmed and sign-in kept refusing it.
 *
 * Two shapes arrive here:
 *   ?code=…                 PKCE (Google, and email links sent from the server)
 *   ?token_hash=…&type=…    the token-hash form, if the email template uses it
 * Either way the session cookie is set here, on the server, and the visitor is
 * sent on to `next`, which must be a path on this site (never another origin).
 */
function safeNext(raw: string | null): string {
  return raw && raw.startsWith("/") && !raw.startsWith("//")
    ? raw
    : "/watchlist";
}

export async function GET(request: NextRequest) {
  const url = request.nextUrl;
  const next = safeNext(url.searchParams.get("next"));
  const fail = (message: string) => {
    const to = new URL("/login", url.origin);
    to.searchParams.set("error", message);
    return NextResponse.redirect(to);
  };

  const problem = url.searchParams.get("error_description");
  if (problem) return fail(problem);

  const supabase = await createClient();
  const code = url.searchParams.get("code");
  const tokenHash = url.searchParams.get("token_hash");
  const type = url.searchParams.get("type") as EmailOtpType | null;

  if (code) {
    const { error } = await supabase.auth.exchangeCodeForSession(code);
    if (error)
      return fail(
        "That link has expired or was opened in a different browser. Sign in, or ask for a new link.",
      );
  } else if (tokenHash && type) {
    const { error } = await supabase.auth.verifyOtp({
      type,
      token_hash: tokenHash,
    });
    if (error)
      return fail(
        "That link has expired or was already used. Ask for a new one.",
      );
  } else {
    return fail("That link is incomplete. Ask for a new one.");
  }

  return NextResponse.redirect(new URL(next, url.origin));
}
