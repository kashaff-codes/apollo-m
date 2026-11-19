"""
Accounts for the analyst console, on Supabase Auth.

The console and the public web view share one identity provider. Supabase Auth
holds the users (passwords hashed by it, never seen here), issues the session
as a signed JWT (about an hour) plus a refresh token, runs Google sign-in, and
the database enforces who may read what (row-level security, accounts.sql).
This module is the console's client for it:

  sign_in(email, password)          -> Session      password grant
  sign_up(email, password, name)    -> Session | "confirm your email"
  google_url(app_url) / google_exchange(code, nonce)
                                    -> Session      OAuth with PKCE
  refresh(refresh_token)            -> Session      "remember me", token rotation
  sign_out(session)                                  revokes the refresh token
  profile(session) / log(session, action, detail)    PostgREST, as the user

Nothing here holds a service-role key: every database call carries the user's
own JWT, so the database, not this code, decides what they may see and write.

Brute-force protection on top of Supabase's own rate limits: five failed
sign-ins for an email within fifteen minutes lock that email out of this
console for fifteen minutes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

TIMEOUT = 15
LOCK_AFTER = 5
LOCK_WINDOW_S = 15 * 60
MIN_PASSWORD = 10


class AuthError(Exception):
    """A message that is safe and useful to show the person signing in."""


@dataclass
class Session:
    access_token: str
    refresh_token: str
    expires_at: float
    user_id: str
    email: str
    name: str = ""
    provider: str = "email"
    role: str = "analyst"
    meta: dict = field(default_factory=dict)

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at - 60


# The project the public web view already uses. A *publishable* key is meant to
# be public (the web view ships it to every browser); it grants nothing beyond
# what row-level security allows an anonymous or signed-in caller. Environment
# variables override both, and APOLLO_ACCOUNTS=off disables accounts entirely.
DEFAULT_URL = "https://wvbheyfoonoobvtkeusm.supabase.co"
DEFAULT_KEY = "sb_publishable_FS6p8WMDYixn6H5iPIq0Pg_H4KEg1uB"


def config() -> tuple[str, str] | None:
    if os.getenv("APOLLO_ACCOUNTS", "").lower() == "off":
        return None
    url = (os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL") or DEFAULT_URL).rstrip("/")
    key = (os.getenv("SUPABASE_PUBLISHABLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
           or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY") or DEFAULT_KEY)
    return (url, key) if url and key else None


_PROVIDERS: dict = {"at": 0.0, "google": False}


def google_enabled() -> bool:
    """Whether Supabase has Google sign-in switched on (public settings endpoint,
    checked at most every five minutes). The button is shown only when it is, so
    no one is sent into a flow that ends in "provider is not enabled"."""
    import requests
    cfg = config()
    if not cfg:
        return False
    if time.time() - _PROVIDERS["at"] < 300:
        return bool(_PROVIDERS["google"])
    try:
        r = requests.get(f"{cfg[0]}/auth/v1/settings", headers=_headers(cfg[1]), timeout=TIMEOUT)
        _PROVIDERS["google"] = bool((r.json().get("external") or {}).get("google")) if r.status_code == 200 else False
    except Exception:  # noqa: BLE001
        _PROVIDERS["google"] = False
    _PROVIDERS["at"] = time.time()
    return bool(_PROVIDERS["google"])


def _headers(key: str, token: str | None = None) -> dict:
    h = {"apikey": key, "Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _session(body: dict, provider: str = "email") -> Session:
    user = body.get("user") or {}
    meta = user.get("user_metadata") or {}
    return Session(
        access_token=body["access_token"], refresh_token=body["refresh_token"],
        expires_at=time.time() + float(body.get("expires_in") or 3600),
        user_id=user.get("id", ""), email=user.get("email", ""),
        name=meta.get("full_name") or meta.get("name") or "",
        provider=(user.get("app_metadata") or {}).get("provider", provider), meta=meta)


def _message(r) -> str:
    try:
        b = r.json()
        return str(b.get("msg") or b.get("error_description") or b.get("message") or b.get("error") or "")
    except Exception:  # noqa: BLE001
        return r.text[:200]


# ── brute-force lockout ──────────────────────────────────────────────────────

_FAILS: dict[str, list[float]] = {}
_LOCK = threading.Lock()


def _locked(email: str) -> int:
    """Seconds until this email may try again (0 = not locked)."""
    now = time.time()
    with _LOCK:
        recent = [t for t in _FAILS.get(email, []) if now - t < LOCK_WINDOW_S]
        _FAILS[email] = recent
        if len(recent) >= LOCK_AFTER:
            return int(LOCK_WINDOW_S - (now - recent[-LOCK_AFTER]))
    return 0


def _fail(email: str) -> None:
    with _LOCK:
        _FAILS.setdefault(email, []).append(time.time())


def _clear(email: str) -> None:
    with _LOCK:
        _FAILS.pop(email, None)


def password_problem(password: str) -> str | None:
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters."
    if password.isalpha() or password.isdigit():
        return "Mix letters with numbers or symbols."
    return None


# ── email and password ───────────────────────────────────────────────────────

def sign_in(email: str, password: str) -> Session:
    import requests
    cfg = config()
    if not cfg:
        raise AuthError("Accounts are not configured on this deployment.")
    email = email.strip().lower()
    if wait := _locked(email):
        raise AuthError(f"Too many attempts. Try again in {max(1, wait // 60)} minutes.")
    url, key = cfg
    r = requests.post(f"{url}/auth/v1/token?grant_type=password", headers=_headers(key),
                      json={"email": email, "password": password}, timeout=TIMEOUT)
    if r.status_code == 200:
        _clear(email)
        return _session(r.json())
    _fail(email)
    msg = _message(r).lower()
    if "not confirmed" in msg:
        raise AuthError("Confirm your email first — the link is in your inbox.")
    # One message for unknown email and wrong password: the form must not reveal
    # which addresses have accounts.
    raise AuthError("That email and password do not match an account.")


def sign_up(email: str, password: str, name: str) -> Session | str:
    import requests
    cfg = config()
    if not cfg:
        raise AuthError("Accounts are not configured on this deployment.")
    if problem := password_problem(password):
        raise AuthError(problem)
    if not name.strip():
        raise AuthError("Tell us your name.")
    url, key = cfg
    r = requests.post(f"{url}/auth/v1/signup", headers=_headers(key),
                      json={"email": email.strip().lower(), "password": password,
                            "data": {"full_name": name.strip()}}, timeout=TIMEOUT)
    if r.status_code not in (200, 201):
        msg = _message(r)
        if "registered" in msg.lower() or "exists" in msg.lower():
            raise AuthError("An account with that email exists. Sign in instead.")
        raise AuthError(msg or "Could not create the account.")
    body = r.json()
    if body.get("access_token"):
        return _session(body)
    return "Check your inbox: confirm your email, then sign in."


# ── Google, OAuth 2.0 with PKCE ──────────────────────────────────────────────

_PKCE: dict[str, tuple[str, float]] = {}


def google_url(app_url: str) -> str:
    """Where to send the browser. The verifier stays on the server, keyed by a nonce
    carried in the redirect; it is single-use and expires after ten minutes."""
    cfg = config()
    if not cfg:
        raise AuthError("Accounts are not configured on this deployment.")
    url, _ = cfg
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    nonce = secrets.token_urlsafe(16)
    with _LOCK:
        now = time.time()
        for k in [k for k, (_, t) in _PKCE.items() if now - t > 600]:
            _PKCE.pop(k, None)
        _PKCE[nonce] = (verifier, now)
    redirect = f"{app_url.rstrip('/')}/?{urlencode({'auth': 'google', 'n': nonce})}"
    return f"{url}/auth/v1/authorize?" + urlencode({
        "provider": "google", "redirect_to": redirect,
        "code_challenge": challenge, "code_challenge_method": "s256"})


def google_exchange(code: str, nonce: str) -> Session:
    import requests
    cfg = config()
    with _LOCK:
        entry = _PKCE.pop(nonce, None)
    if not cfg or not entry or time.time() - entry[1] > 600:
        raise AuthError("That sign-in link expired. Try Google again.")
    url, key = cfg
    r = requests.post(f"{url}/auth/v1/token?grant_type=pkce", headers=_headers(key),
                      json={"auth_code": code, "code_verifier": entry[0]}, timeout=TIMEOUT)
    if r.status_code != 200:
        raise AuthError("Google sign-in did not complete. Try again.")
    return _session(r.json(), provider="google")


# ── keeping and ending a session ─────────────────────────────────────────────

def refresh(refresh_token: str) -> Session:
    """Exchange a refresh token for a new session. Supabase rotates the refresh
    token, so the old one stops working: store the new one."""
    import requests
    cfg = config()
    if not cfg or not refresh_token:
        raise AuthError("No saved session.")
    url, key = cfg
    r = requests.post(f"{url}/auth/v1/token?grant_type=refresh_token", headers=_headers(key),
                      json={"refresh_token": refresh_token}, timeout=TIMEOUT)
    if r.status_code != 200:
        raise AuthError("Your saved session has expired. Sign in again.")
    return _session(r.json())


def sign_out(s: Session) -> None:
    import requests
    cfg = config()
    if not cfg:
        return
    url, key = cfg
    try:  # revokes the refresh token server-side, on every device ("global")
        requests.post(f"{url}/auth/v1/logout?scope=global", headers=_headers(key, s.access_token),
                      timeout=TIMEOUT)
    except Exception:  # noqa: BLE001 - signing out locally must still happen
        pass


def ensure_fresh(s: Session) -> Session:
    return refresh(s.refresh_token) if s.expired else s


# ── the user's own rows, through row-level security ─────────────────────────

def profile(s: Session) -> Session:
    """Role and name from public.profiles, read with the user's own JWT."""
    import requests
    cfg = config()
    if not cfg:
        return s
    url, key = cfg
    try:
        r = requests.get(f"{url}/rest/v1/profiles", headers=_headers(key, s.access_token),
                         params={"id": f"eq.{s.user_id}", "select": "role,full_name"}, timeout=TIMEOUT)
        rows = r.json() if r.status_code == 200 else []
        if rows:
            s.role = rows[0].get("role") or "analyst"
            s.name = rows[0].get("full_name") or s.name
        requests.patch(f"{url}/rest/v1/profiles", headers={**_headers(key, s.access_token),
                                                           "Prefer": "return=minimal"},
                       params={"id": f"eq.{s.user_id}"},
                       json={"last_seen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                       timeout=TIMEOUT)
    except Exception:  # noqa: BLE001 - a missing profile must not block sign-in
        pass
    return s


def log(s: Session | None, action: str, detail: dict | None = None) -> None:
    """Append one row to public.activity as this user. Best effort, never raises."""
    if not s:
        return
    import requests
    cfg = config()
    if not cfg:
        return
    url, key = cfg
    body = {"surface": "console", "action": action[:40],
            "detail": json.loads(json.dumps(detail or {}, default=str)[:1800] or "{}")
            if len(json.dumps(detail or {}, default=str)) < 1800 else {"truncated": True}}
    try:
        requests.post(f"{url}/rest/v1/activity", headers={**_headers(key, s.access_token),
                                                          "Prefer": "return=minimal"},
                      json=body, timeout=5)
    except Exception:  # noqa: BLE001
        pass


def admin_activity(s: Session, since_days: int = 30, limit: int = 2000) -> tuple[list, list]:
    """Everyone's profiles and recent activity. RLS returns all rows only to admins;
    anyone else gets their own, so this is safe to call for any session."""
    import requests
    cfg = config()
    if not cfg:
        return [], []
    url, key = cfg
    since = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - since_days * 86400))
    h = _headers(key, s.access_token)
    try:
        users = requests.get(f"{url}/rest/v1/profiles", headers=h, timeout=TIMEOUT,
                             params={"select": "id,email,full_name,role,created_at,last_seen_at",
                                     "order": "created_at.desc"})
        acts = requests.get(f"{url}/rest/v1/activity", headers=h, timeout=TIMEOUT,
                            params={"select": "user_id,at,surface,action,detail", "at": f"gte.{since}",
                                    "order": "at.desc", "limit": str(limit)})
    except Exception:  # noqa: BLE001 - an outage shows an empty admin page, not a crash
        return [], []
    return (users.json() if users.status_code == 200 else [],
            acts.json() if acts.status_code == 200 else [])
