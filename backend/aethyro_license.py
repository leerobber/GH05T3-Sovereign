"""
Aethyro license gate for GH05T3.

Call `gate()` at the top of every launcher (launch_backend.py, launch_gateway.py,
supervisor.py). If the user's Aethyro trial/subscription isn't active, the
process prints instructions and exits — the system won't run unlicensed.

Activate a device once (stores a refresh token in ~/.aethyro/license.json):
    python backend/aethyro_license.py login you@email.com YOURPASSWORD

How it works:
  • activate-license issues an RS256 token signed by Aethyro's private key.
  • We verify it OFFLINE with the bundled public key (works with no internet,
    up to the token's 7-day grace window).
  • When online we re-check with the server to catch cancellations early.

Deps: requests (required), pyjwt + cryptography (optional — enables offline
verification; without them we fall back to online-only checks).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import requests

try:
    import jwt  # PyJWT (+ cryptography) — optional, enables offline verify
    _HAVE_JWT = True
except Exception:
    _HAVE_JWT = False

SUPABASE_URL = "https://uzmdqbtflcpikjdrggqc.supabase.co"
SUPABASE_ANON = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6"
                 "InV6bWRxYnRmbGNwaWtqZHJnZ3FjIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU2ODQxOTgs"
                 "ImV4cCI6MjA5MTI2MDE5OH0.BwNVBJCbw9SG-ge7PfmoIW8q_33k-ZQlqpDHa2HrvHI")

CACHE = Path.home() / ".aethyro" / "license.json"
PUBKEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAmi4c+Noeb6ZmRNwzJoyz
1Ven4NgBs3F5jP+ZgniEWStQvIzrpCYK5XcldZ83r/L4whQYEQ2ZTqBq2CE6tjVG
GxTHlXDFeQzvzizzGXJRd6jZNtpaOpobTqEPpzDv4C9UblUtYwpiT6yQSdJ1XSIW
TNIAtyUl3dcVFSbylPfr9LScw+U7k/imyisJHt+s6lkNYX9AMJzFH1cOsL7nz5qJ
v8BnHyXM1TffN1Je0xgkdjbrHraJYeGyOPYeoVf7iZQKVZPxdsLoGuiaVs3ACKt2
FGtnglWssjriVIjrLg+ZDwCJ8KCLjHx50Ze6nhKamDEsO7sKcruPW7kY9qivkWAV
EQIDAQAB
-----END PUBLIC KEY-----
"""  # public key — safe to ship; verifies license tokens offline


class LicenseError(Exception):
    pass


def device_fingerprint() -> str:
    raw = f"{platform.node()}|{platform.system()}|{platform.machine()}"
    return "GH05T3-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except Exception:
        return {}


def _save(d: dict):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d))


def login(email: str, password: str) -> dict:
    r = requests.post(f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
                      headers={"apikey": SUPABASE_ANON, "Content-Type": "application/json"},
                      json={"email": email, "password": password}, timeout=20)
    if not r.ok:
        raise LicenseError(f"login failed: {r.json().get('error_description', r.text)}")
    tok = r.json()
    c = _load(); c.update(refresh_token=tok["refresh_token"], access_token=tok["access_token"]); _save(c)
    return tok


def _access_token() -> str:
    c = _load()
    if c.get("refresh_token"):
        r = requests.post(f"{SUPABASE_URL}/auth/v1/token?grant_type=refresh_token",
                          headers={"apikey": SUPABASE_ANON, "Content-Type": "application/json"},
                          json={"refresh_token": c["refresh_token"]}, timeout=20)
        if r.ok:
            tok = r.json(); c.update(access_token=tok["access_token"], refresh_token=tok["refresh_token"]); _save(c)
            return tok["access_token"]
    if c.get("access_token"):
        return c["access_token"]
    raise LicenseError("not logged in — run: python backend/aethyro_license.py login <email> <password>")


def activate() -> dict:
    r = requests.post(f"{SUPABASE_URL}/functions/v1/activate-license",
                      headers={"apikey": SUPABASE_ANON, "Authorization": f"Bearer {_access_token()}",
                               "Content-Type": "application/json"},
                      json={"device_fingerprint": device_fingerprint(), "device_label": platform.node()},
                      timeout=20)
    data = r.json()
    if not data.get("entitled"):
        raise LicenseError(data.get("reason", json.dumps(data)))
    c = _load(); c["license_token"] = data["token"]; _save(c)
    return data


def ensure_licensed(online: bool = True) -> dict:
    """Return {plan, expires} if entitled, else raise LicenseError."""
    c = _load()
    token = c.get("license_token")
    claims = None

    if token and _HAVE_JWT and PUBKEY:
        try:
            claims = jwt.decode(token, PUBKEY, algorithms=["RS256"], audience="aethyro-app", issuer="aethyro")
        except Exception:
            token = None  # expired/invalid -> must re-activate

    # No usable token (or can't verify offline) -> activate online.
    if not token or not claims:
        data = activate()
        return {"plan": data["plan"], "expires": data["expires_at"]}

    # Online re-check (best effort) to catch cancellations early.
    if online:
        try:
            v = requests.post(f"{SUPABASE_URL}/functions/v1/validate-license",
                              headers={"apikey": SUPABASE_ANON, "Content-Type": "application/json"},
                              json={"token": token}, timeout=10).json()
            if not v.get("valid"):
                data = activate()  # raises if no longer entitled
                return {"plan": data["plan"], "expires": data["expires_at"]}
            return {"plan": v["plan"], "expires": v["expires_at"]}
        except requests.RequestException:
            pass  # offline -> trust the still-valid token

    return {"plan": claims["plan"], "expires": claims["exp"]}


def gate() -> dict:
    """Hard gate for launchers. Exits the process if not licensed."""
    if os.environ.get("AETHYRO_SKIP_LICENSE") == "1":
        return {"plan": "ci", "expires": None}
    try:
        ent = ensure_licensed(online=True)
        print(f"[Aethyro] license OK — plan={ent['plan']} expires={ent['expires']}")
        return ent
    except LicenseError as e:
        bar = "=" * 64
        print(f"\n{bar}\n  AETHYRO LICENSE REQUIRED — GH05T3 will not start.\n  Reason: {e}\n"
              f"  Activate this device by logging in once:\n"
              f"    python backend\\aethyro_license.py login <email> <password>\n"
              f"  No account yet? Start a free trial at https://aethyro.com\n{bar}\n")
        sys.exit(1)
    except Exception as e:
        # Never hard-crash the app on a license-system glitch; warn and allow start.
        print(f"[Aethyro] license check skipped (transient error: {e})")
        return {"plan": "unverified", "expires": None}


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "login":
        login(sys.argv[2], sys.argv[3]); print("logged in — activating device…")
        print("activated:", activate())
    else:
        print(gate())
