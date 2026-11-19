"use client";

import { useActionState } from "react";
import { useFormStatus } from "react-dom";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import {
  resendConfirmation,
  signIn,
  signInWithGoogle,
  signUp,
  type AuthState,
} from "../auth/actions";

/**
 * The only Client Component with real interactivity in the app.
 *
 * It stays a <form> posting to a Server Action, so it works without JavaScript;
 * useActionState and useFormStatus only add the pending state and the inline
 * error on top of that.
 */

function Submit({ label }: { label: string }) {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      disabled={pending}
      className="w-full rounded-md bg-violet px-4 py-2 font-medium text-white transition hover:bg-violet-deep disabled:opacity-50"
    >
      {pending ? "Working…" : label}
    </button>
  );
}

function ResendButton() {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      disabled={pending}
      className="text-sm text-violet underline-offset-4 hover:underline disabled:opacity-50"
    >
      {pending ? "Sending…" : "Send the confirmation email again"}
    </button>
  );
}

function LoginForm() {
  const [mode, setMode] = useState<"in" | "up">("in");
  const params = useSearchParams();
  const next = params.get("next") ?? "/watchlist";
  const linkError = params.get("error");
  const action = mode === "in" ? signIn : signUp;
  const [state, formAction] = useActionState<AuthState, FormData>(action, {});
  const [resent, resendAction] = useActionState<AuthState, FormData>(
    resendConfirmation,
    {},
  );
  const unconfirmed = resent.unconfirmed ?? state.unconfirmed;

  return (
    <div className="mx-auto max-w-sm">
      <h1 className="font-serif text-4xl tracking-tight">
        {mode === "in" ? "Sign in" : "Create an account"}
      </h1>
      <p className="mt-1 text-sm text-ink/60">
        Supabase Auth: Google, or email and password.
      </p>

      {linkError && !state.error && (
        <p role="alert" className="mt-4 text-sm text-[#a32727]">
          {linkError}
        </p>
      )}

      <form action={signInWithGoogle} className="mt-6">
        <input type="hidden" name="next" value={next} />
        <button
          type="submit"
          className="flex w-full items-center justify-center gap-2 rounded-md bg-card px-4 py-2 font-medium ring-1 ring-ink/20 transition hover:ring-violet"
        >
          <svg width="16" height="16" viewBox="0 0 48 48" aria-hidden="true">
            <path
              fill="#FFC107"
              d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.3-.1-2.4-.4-3.5z"
            />
            <path
              fill="#FF3D00"
              d="M6.3 14.7l6.6 4.8C14.7 15.1 19 12 24 12c3.1 0 5.8 1.2 7.9 3.1l5.7-5.7C34 6.1 29.3 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"
            />
            <path
              fill="#4CAF50"
              d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z"
            />
            <path
              fill="#1976D2"
              d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C37 39.2 44 34 44 24c0-1.3-.1-2.4-.4-3.5z"
            />
          </svg>
          Continue with Google
        </button>
      </form>

      <div className="mt-6 flex items-center gap-3 text-xs text-ink/50">
        <span className="h-px flex-1 bg-ink/15" />
        or with email
        <span className="h-px flex-1 bg-ink/15" />
      </div>

      <form action={formAction} className="mt-4 space-y-4">
        <input type="hidden" name="next" value={next} />
        <div>
          <label htmlFor="email" className="block text-xs text-ink/70">
            Email
          </label>
          <input
            id="email"
            name="email"
            type="email"
            required
            autoComplete="email"
            className="mt-1 w-full rounded-md bg-card px-3 py-2 ring-1 ring-ink/20 outline-none focus:ring-violet"
          />
        </div>
        <div>
          <label htmlFor="password" className="block text-xs text-ink/70">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            required
            minLength={8}
            autoComplete={mode === "in" ? "current-password" : "new-password"}
            className="mt-1 w-full rounded-md bg-card px-3 py-2 ring-1 ring-ink/20 outline-none focus:ring-violet"
          />
        </div>

        {state.error && (
          <p role="alert" className="text-sm text-[#a32727]">
            {state.error}
          </p>
        )}
        {state.message && (
          <p role="status" className="text-sm text-violet">
            {state.message}
          </p>
        )}

        <Submit label={mode === "in" ? "Sign in" : "Sign up"} />
      </form>

      {unconfirmed && (
        <form action={resendAction} className="mt-3 space-y-2">
          <input type="hidden" name="email" value={unconfirmed} />
          <input type="hidden" name="next" value={next} />
          <ResendButton />
          {resent.message && (
            <p role="status" className="text-sm text-violet">
              {resent.message}
            </p>
          )}
          {resent.error && (
            <p role="alert" className="text-sm text-[#a32727]">
              {resent.error}
            </p>
          )}
        </form>
      )}

      <button
        onClick={() => setMode(mode === "in" ? "up" : "in")}
        className="mt-4 text-sm text-ink/60 hover:text-ink"
      >
        {mode === "in"
          ? "No account? Create one"
          : "Already have an account? Sign in"}
      </button>
    </div>
  );
}

export default function LoginPage() {
  // useSearchParams needs a Suspense boundary so the rest of the route can be
  // prerendered rather than opting the whole page into client-side rendering.
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}
