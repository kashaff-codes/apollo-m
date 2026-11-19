"""
The console's account client (dashboard/auth.py), against a fake Supabase.

What must hold for an enterprise login, and is checked here without network:
  * a wrong password and an unknown email produce the same message;
  * five failures lock that email out, and a success clears the count;
  * the password policy is enforced before any request is made;
  * Google sign-in uses PKCE: the challenge is S256 of the verifier, the
    verifier never leaves the server, and each nonce works exactly once;
  * refresh exchanges and rotates the token; sign-out revokes server-side;
  * every database call carries the user's own JWT, never a service key;
  * activity rows are size-bounded.
"""
from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dashboard"))

import auth as A  # noqa: E402


class _R:
    def __init__(self, status, body):
        self.status_code, self._b = status, body
        self.text = str(body)

    def json(self):
        return self._b


@pytest.fixture
def supa(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    A._FAILS.clear()
    A._PKCE.clear()
    calls = []
    users = {"ana@example.com": "correct-horse-9"}

    def session_body(email="ana@example.com"):
        return {"access_token": "jwt.access", "refresh_token": "rt-2", "expires_in": 3600,
                "user": {"id": "u1", "email": email, "user_metadata": {"full_name": "Ana"},
                         "app_metadata": {"provider": "email"}}}

    def post(url, headers=None, json=None, params=None, timeout=None):
        calls.append(("POST", url, headers, json))
        if "grant_type=password" in url:
            if users.get(json["email"]) == json["password"]:
                return _R(200, session_body(json["email"]))
            return _R(400, {"error": "invalid_grant", "error_description": "Invalid login credentials"})
        if "grant_type=refresh_token" in url:
            return _R(200, session_body()) if json["refresh_token"] == "rt-1" else _R(400, {"msg": "bad"})
        if "grant_type=pkce" in url:
            return _R(200, {**session_body(), "user": {**session_body()["user"],
                                                       "app_metadata": {"provider": "google"}}})
        if url.endswith("/auth/v1/signup"):
            return _R(200, {"id": "u2", "email": json["email"]})       # email confirmation on
        if "/auth/v1/logout" in url:
            return _R(204, {})
        if "/rest/v1/activity" in url:
            return _R(201, {})
        return _R(404, {})

    def get(url, headers=None, params=None, timeout=None):
        calls.append(("GET", url, headers, params))
        return _R(200, [{"role": "admin", "full_name": "Ana A."}])

    def patch(url, headers=None, params=None, json=None, timeout=None):
        calls.append(("PATCH", url, headers, json))
        return _R(204, {})

    import requests
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "patch", patch)
    return calls


def test_sign_in_returns_a_session(supa):
    s = A.sign_in("  Ana@Example.com ", "correct-horse-9")
    assert s.access_token == "jwt.access" and s.email == "ana@example.com" and s.name == "Ana"
    assert not s.expired


def test_wrong_password_and_unknown_email_read_the_same(supa):
    with pytest.raises(A.AuthError) as a:
        A.sign_in("ana@example.com", "wrong-password-1")
    with pytest.raises(A.AuthError) as b:
        A.sign_in("nobody@example.com", "whatever-pass-1")
    assert str(a.value) == str(b.value)


def test_five_failures_lock_the_email_and_success_resets(supa):
    for _ in range(A.LOCK_AFTER):
        with pytest.raises(A.AuthError):
            A.sign_in("ana@example.com", "nope-nope-1")
    with pytest.raises(A.AuthError, match="Too many attempts"):
        A.sign_in("ana@example.com", "correct-horse-9")     # locked even with the right password
    A._FAILS.clear()
    A.sign_in("ana@example.com", "correct-horse-9")
    assert "ana@example.com" not in A._FAILS


@pytest.mark.parametrize("pw", ["short1", "onlyletterslong", "12345678901"])
def test_password_policy_runs_before_any_request(supa, pw):
    with pytest.raises(A.AuthError):
        A.sign_up("new@example.com", pw, "New Person")
    assert not any("/signup" in c[1] for c in supa)


def test_sign_up_with_email_confirmation(supa):
    assert "confirm" in A.sign_up("new@example.com", "a-strong-pass-1", "New Person").lower()


def test_google_uses_pkce_and_each_nonce_once(supa):
    url = A.google_url("https://apollo-m.streamlit.app")
    q = parse_qs(urlparse(url).query)
    assert q["provider"] == ["google"] and q["code_challenge_method"] == ["s256"]
    nonce = parse_qs(urlparse(q["redirect_to"][0]).query)["n"][0]
    verifier = A._PKCE[nonce][0]
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert q["code_challenge"] == [expected]
    assert verifier not in url, "the verifier must never leave the server"
    s = A.google_exchange("code-1", nonce)
    assert s.provider == "google"
    with pytest.raises(A.AuthError):
        A.google_exchange("code-1", nonce)                     # replay refused


def test_refresh_rotates_and_rejects_unknown_tokens(supa):
    assert A.refresh("rt-1").refresh_token == "rt-2"
    with pytest.raises(A.AuthError):
        A.refresh("stolen-or-old")


def test_sign_out_revokes_globally_with_the_users_token(supa):
    s = A.sign_in("ana@example.com", "correct-horse-9")
    A.sign_out(s)
    call = [c for c in supa if "/logout" in c[1]][0]
    assert "scope=global" in call[1] and call[2]["Authorization"] == "Bearer jwt.access"


def test_database_calls_carry_the_users_jwt_not_a_service_key(supa):
    s = A.profile(A.sign_in("ana@example.com", "correct-horse-9"))
    assert s.role == "admin"            # whatever the database says, nothing else
    A.log(s, "topic_search", {"query": "remote hiring"})
    db = [c for c in supa if "/rest/v1/" in c[1]]
    assert db and all(c[2]["Authorization"] == "Bearer jwt.access" for c in db)
    assert all(c[2]["apikey"].startswith("sb_publishable") for c in db)


def test_activity_detail_is_bounded(supa):
    s = A.sign_in("ana@example.com", "correct-horse-9")
    A.log(s, "x" * 99, {"query": "y" * 5000})
    body = [c for c in supa if "/rest/v1/activity" in c[1]][-1][3]
    assert len(body["action"]) == 40 and body["detail"] == {"truncated": True}


def test_not_configured_is_a_clear_message(monkeypatch):
    monkeypatch.setenv("APOLLO_ACCOUNTS", "off")
    for v in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY", "NEXT_PUBLIC_SUPABASE_URL",
              "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "SUPABASE_ANON_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(A.AuthError, match="not configured"):
        A.sign_in("a@b.co", "whatever-1234")
