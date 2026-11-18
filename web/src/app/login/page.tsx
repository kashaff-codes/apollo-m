"use client";

import { useActionState } from "react";
import { useFormStatus } from "react-dom";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import { signIn, signUp, type AuthState } from "../auth/actions";

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

function LoginForm() {
  const [mode, setMode] = useState<"in" | "up">("in");
  const next = useSearchParams().get("next") ?? "/watchlist";
  const action = mode === "in" ? signIn : signUp;
  const [state, formAction] = useActionState<AuthState, FormData>(action, {});

  return (
    <div className="mx-auto max-w-sm">
      <h1 className="font-serif text-4xl tracking-tight">
        {mode === "in" ? "Sign in" : "Create an account"}
      </h1>
      <p className="mt-1 text-sm text-ink/60">
        Supabase Auth, email and password.
      </p>

      <form action={formAction} className="mt-6 space-y-4">
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
