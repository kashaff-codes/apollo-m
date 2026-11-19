"use server";

import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { headers } from "next/headers";
import { createClient } from "@/lib/supabase/server";

/**
 * Auth as Server Actions rather than client-side calls.
 *
 * The credentials are posted straight from a <form> to the server, so sign-in
 * works before any JavaScript has loaded, and the session cookie is set by the
 * server rather than written from client code.
 */

export type AuthState = {
  error?: string;
  message?: string;
  /** The account exists but its email is not confirmed: offer to resend the link. */
  unconfirmed?: string;
};

/** This site's origin, from the request: where Supabase should send people back to. */
async function siteOrigin(): Promise<string> {
  const h = await headers();
  const origin = h.get("origin");
  if (origin) return origin;
  const host =
    h.get("x-forwarded-host") ?? h.get("host") ?? "apollo-m.vercel.app";
  const proto =
    h.get("x-forwarded-proto") ??
    (host.startsWith("localhost") ? "http" : "https");
  return `${proto}://${host}`;
}

function safeNext(raw: FormDataEntryValue | null): string {
  const s = String(raw ?? "");
  return s.startsWith("/") && !s.startsWith("//") ? s : "/watchlist";
}

async function confirmUrl(next: string): Promise<string> {
  return `${await siteOrigin()}/auth/confirm?next=${encodeURIComponent(next)}`;
}

function readCredentials(formData: FormData) {
  return {
    email: String(formData.get("email") ?? "").trim(),
    password: String(formData.get("password") ?? ""),
  };
}

export async function signIn(
  _prev: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const { email, password } = readCredentials(formData);
  if (!email || !password)
    return { error: "Email and password are both required." };

  const supabase = await createClient();
  const { error } = await supabase.auth.signInWithPassword({ email, password });

  // Supabase deliberately returns one message — "Invalid login credentials" —
  // for both an unknown email and a wrong password, so the form cannot be used
  // to test whether an account exists. That property is worth keeping, but the
  // message on its own sends someone to check a password that was never wrong:
  // on a fresh project the real answer is almost always that they have not
  // signed up yet.
  //
  // So it is rewritten to name both possibilities without distinguishing
  // between them, which leaks nothing and points at the next action. Other
  // errors ("Email not confirmed", rate limits) already say what to do and are
  // passed through untouched.
  if (error) {
    if (/invalid login credentials/i.test(error.message)) {
      return {
        error:
          "That email and password do not match an account. If you have not " +
          "signed up yet, create one below.",
      };
    }
    if (/email not confirmed/i.test(error.message)) {
      return {
        error:
          "This account's email address is not confirmed yet. Open the link we " +
          "emailed you, or send a new one.",
        unconfirmed: email,
      };
    }
    return { error: error.message };
  }

  revalidatePath("/", "layout");
  redirect(safeNext(formData.get("next")));
}

export async function signUp(
  _prev: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const { email, password } = readCredentials(formData);
  if (password.length < 8)
    return { error: "Password must be at least 8 characters." };

  const supabase = await createClient();
  // Without emailRedirectTo the link in the email falls back to the project's
  // Site URL (the Streamlit console), which cannot finish a web sign-up.
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo: await confirmUrl(safeNext(formData.get("next"))),
    },
  });
  if (error) return { error: error.message };

  // Supabase answers a sign-up for an address that already has a confirmed
  // account with a user that has no identities (so the form cannot be used to
  // discover who is registered) and sends nothing. Say so instead of promising
  // an email that will never come.
  if (data.user && (data.user.identities ?? []).length === 0) {
    return {
      error:
        "That email address already has an account. Sign in, or use Continue with Google.",
    };
  }

  // With "Confirm email" on (the Supabase default) there is no session yet and
  // the user has to click the link first. With it off they are signed in
  // immediately. Handle both rather than assuming the project's setting.
  if (data.session) {
    revalidatePath("/", "layout");
    redirect("/watchlist");
  }
  return {
    message:
      `We sent a confirmation link to ${email}. Open it on this device to finish ` +
      "signing up (check Spam or Promotions if it is not in your inbox).",
    unconfirmed: email,
  };
}

/** Send the confirmation email again (the first may be in spam, or expired). */
export async function resendConfirmation(
  _prev: AuthState,
  formData: FormData,
): Promise<AuthState> {
  const email = String(formData.get("email") ?? "").trim();
  if (!email) return { error: "Enter your email address first." };
  const supabase = await createClient();
  const { error } = await supabase.auth.resend({
    type: "signup",
    email,
    options: {
      emailRedirectTo: await confirmUrl(safeNext(formData.get("next"))),
    },
  });
  if (error) {
    if (/rate|seconds|too many/i.test(error.message))
      return {
        error:
          "A link was sent a moment ago. Wait a minute before asking for another.",
        unconfirmed: email,
      };
    return { error: error.message, unconfirmed: email };
  }
  return {
    message: `A new confirmation link is on its way to ${email}.`,
    unconfirmed: email,
  };
}

/** Sign in with Google through Supabase: no password, no confirmation email. */
export async function signInWithGoogle(formData: FormData) {
  const supabase = await createClient();
  const { data, error } = await supabase.auth.signInWithOAuth({
    provider: "google",
    options: { redirectTo: await confirmUrl(safeNext(formData.get("next"))) },
  });
  if (error || !data.url)
    redirect(
      `/login?error=${encodeURIComponent(error?.message ?? "Google sign-in is unavailable.")}`,
    );
  redirect(data.url);
}

export async function signOut() {
  const supabase = await createClient();
  await supabase.auth.signOut();
  revalidatePath("/", "layout");
  redirect("/");
}

/** Add a community to the signed-in user's watchlist.
 *  user_id is taken from the verified session, never from the form — and the
 *  insert policy in schema.sql refuses the row if the two disagree. */
export async function addToWatchlist(formData: FormData) {
  const supabase = await createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) redirect("/login?next=/watchlist");

  const subreddit = String(formData.get("subreddit") ?? "");
  if (!subreddit) return;

  await supabase
    .from("watchlist")
    .upsert(
      { user_id: user.id, subreddit },
      { onConflict: "user_id,subreddit" },
    );

  revalidatePath("/watchlist");
  revalidatePath("/");
}

export async function removeFromWatchlist(formData: FormData) {
  const supabase = await createClient();
  const id = String(formData.get("id") ?? "");
  if (!id) return;

  // No user_id filter here on purpose: the delete policy scopes it to
  // auth.uid(), so another user's id simply matches no rows. The database is
  // the authority, not this query.
  await supabase.from("watchlist").delete().eq("id", id);

  revalidatePath("/watchlist");
  revalidatePath("/");
}
