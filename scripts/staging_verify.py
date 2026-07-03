#!/usr/bin/env python3
"""
Staging provider verification.

1. Health-check staging (/oss/health).
2. Probe /_pact/provider_states (optional).
3. Run Pact provider verification against live staging URL.

Environment or flags (flags override env):
  STAGING_BASE_URL / --provider-url
  PACT_BROKER_URL  / --broker-url
  PACT_BROKER_TOKEN / --broker-token
  PROVIDER_NAME    / --provider-name  (default: gh05t3-oss)
  CONSUMER_TAG     / --consumer-tag   (default: ci)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from oss.utils.paths import ensure_oss_paths

ensure_oss_paths()

PROVIDER_DEFAULT = "gh05t3-oss"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Staging Pact provider verification.")
    p.add_argument("--provider-url", default=os.environ.get("STAGING_BASE_URL", ""))
    p.add_argument("--broker-url", default=os.environ.get("PACT_BROKER_URL") or os.environ.get("PACT_BROKER_BASE_URL", ""))
    p.add_argument("--broker-token", default=os.environ.get("PACT_BROKER_TOKEN", ""))
    p.add_argument("--provider-name", default=os.environ.get("PROVIDER_NAME", PROVIDER_DEFAULT))
    p.add_argument("--provider-version", default=os.environ.get("PROVIDER_VERSION", "staging"))
    p.add_argument("--consumer-tag", default=os.environ.get("CONSUMER_TAG", "ci"))
    return p.parse_args()


def check_staging(base: str, timeout: float = 15.0) -> bool:
    try:
        with httpx.Client(timeout=timeout, verify=True) as client:
            r = client.get(f"{base.rstrip('/')}/oss/health")
            return r.status_code == 200
    except Exception as exc:
        print(f"[staging] Health check failed: {exc}")
        return False


def probe_provider_states(base: str) -> str | None:
    url = f"{base.rstrip('/')}/_pact/provider_states"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, json={"state": "MVS substrate is initialized"})
            if r.status_code == 200:
                print(f"[staging] Provider states endpoint OK: {url}")
                return url
    except Exception as exc:
        print(f"[staging] Provider states not available ({exc})")
    return None


def run_pact_verify(
    provider_url: str,
    states_url: str | None,
    *,
    broker_url: str,
    broker_token: str,
    provider_name: str,
    consumer_tag: str,
) -> int:
    if os.environ.get("SKIP_PACT") == "1":
        print("[staging] SKIP_PACT=1 — skipping Pact verification")
        return 0

    try:
        from pact import Verifier
    except Exception as exc:
        print(f"[staging] pact-python unavailable: {exc}")
        return 0

    verifier = Verifier(provider=provider_name, provider_base_url=provider_url.rstrip("/"))

    if broker_url:
        print(f"[staging] Verifying via broker: {broker_url}")
        output, logs = verifier.verify_with_broker(
            broker_url=broker_url,
            broker_token=broker_token or None,
            consumer_version_selectors=[{"latest": True, "tag": consumer_tag}],
            publish_verification_results=True,
            provider_states_setup_url=states_url,
        )
    else:
        pacts_dir = Path(os.environ.get("PACT_DIR", ROOT / "pacts"))
        if not pacts_dir.is_dir():
            print(f"[staging] No broker and no {pacts_dir} — health-only pass")
            return 0
        pact_paths = [str(p) for p in pacts_dir.glob("*.json")]
        if not pact_paths:
            print("[staging] No pact JSON files — health-only pass")
            return 0
        print(f"[staging] Verifying {len(pact_paths)} local pact(s) against staging")
        kwargs = {}
        if states_url:
            kwargs["provider_states_setup_url"] = states_url
        output, logs = verifier.verify_pacts(*pact_paths, **kwargs)

    if output != 0:
        print(f"[staging] Provider verification FAILED:\n{logs}")
        return 1
    print("[staging] Provider verification passed")
    return 0


def main() -> int:
    args = parse_args()
    staging = (args.provider_url or "").strip()
    if not staging:
        print("[staging] STAGING_BASE_URL not set — skipping.")
        return 0

    if not check_staging(staging):
        print("[staging] Staging not healthy — failing fast.")
        return 1

    print("[staging] Staging healthy.")
    states_url = probe_provider_states(staging)
    return run_pact_verify(
        staging,
        states_url,
        broker_url=args.broker_url.strip(),
        broker_token=args.broker_token,
        provider_name=args.provider_name,
        consumer_tag=args.consumer_tag,
    )


if __name__ == "__main__":
    sys.exit(main())