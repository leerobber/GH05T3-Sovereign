#!/usr/bin/env python3
"""
Can-I-Deploy gate for Pact Broker.

Supports two modes:
  1. Matrix (legacy): --consumer + --provider — verifies latest pact between pair
  2. PactFlow env:  --pacticipant + --to-environment — broker /can-i-deploy API
  3. Record deploy: --pacticipant + --record-deployment ENV

Exit codes:
  0 — safe to deploy / record succeeded / broker not configured (no-op)
  1 — not safe to deploy
  2 — broker not configured (explicit skip)
  3 — broker network error (fail-safe when --strict)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    log.error("httpx is not installed")
    sys.exit(1)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _headers(token: str | None) -> dict[str, str]:
    h = {"Accept": "application/hal+json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _print_verification_summary(data: dict) -> None:
    matrix = data.get("matrix", [])
    if not matrix:
        return
    log.info("  Pact matrix:")
    for row in matrix:
        consumer = row.get("consumer", {}).get("name", "?")
        c_ver = row.get("consumer", {}).get("version", {}).get("number", "?")
        provider = row.get("provider", {}).get("name", "?")
        p_ver = row.get("provider", {}).get("version", {}).get("number", "?")
        result = row.get("verificationResult", {})
        success = result.get("success")
        icon = "✅" if success else ("❌" if success is False else "⏳")
        log.info("  %s  %s@%s  ←→  %s@%s", icon, consumer, c_ver, provider, p_ver)


def can_i_deploy_environment(
    broker_url: str,
    token: str | None,
    pacticipant: str,
    version: str,
    to_environment: str,
    timeout: int = 30,
) -> bool:
    url = f"{broker_url.rstrip('/')}/can-i-deploy"
    params = {
        "pacticipant": pacticipant,
        "version": version,
        "environment": to_environment,
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=_headers(token), params=params)
    log.info("Broker response: HTTP %s", resp.status_code)

    if resp.status_code == 200:
        data = resp.json()
        deployable = data.get("summary", {}).get("deployable") or data.get("deployable")
        if deployable:
            log.info(
                "✅  %s @ %s can be deployed to '%s'.",
                pacticipant, version, to_environment,
            )
            _print_verification_summary(data)
            return True
        log.error(
            "❌  %s @ %s is NOT safe to deploy to '%s'.",
            pacticipant, version, to_environment,
        )
        _print_verification_summary(data)
        return False

    if resp.status_code == 404:
        log.error(
            "❌  No pact found for %s @ %s. Publish and verify first.",
            pacticipant, version,
        )
        return False

    log.error("❌  Unexpected broker response: HTTP %s – %s", resp.status_code, resp.text[:200])
    return False


def record_deployment(
    broker_url: str,
    token: str | None,
    pacticipant: str,
    version: str,
    environment: str,
    timeout: int = 30,
) -> bool:
    url = (
        f"{broker_url.rstrip('/')}/pacticipants/{pacticipant}"
        f"/versions/{version}/deployed-versions/environment/{environment}"
    )
    headers = _headers(token)
    headers["Content-Type"] = "application/json"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json={})
    if resp.status_code in (200, 201):
        log.info("✅  Recorded deployment of %s @ %s to '%s'.", pacticipant, version, environment)
        return True
    log.error("❌  Failed to record deployment: HTTP %s – %s", resp.status_code, resp.text[:200])
    return False


def can_i_deploy_matrix(
    broker_url: str,
    token: str | None,
    consumer: str,
    provider: str,
    version: str,
    tag: str = "ci",
    timeout: int = 30,
) -> bool:
    url = (
        f"{broker_url.rstrip('/')}/matrix/provider/{provider}/consumer/{consumer}"
        f"/latest/{tag}/consumer-version/{version}"
    )
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=_headers(token))
    if resp.status_code == 200:
        data = resp.json()
        if data.get("verified"):
            log.info("✅  %s → %s @ %s verified (tag=%s).", consumer, provider, version, tag)
            return True
        log.error("❌  %s → %s @ %s NOT verified.", consumer, provider, version)
        return False
    log.warning("Matrix check HTTP %s — %s", resp.status_code, resp.text[:120])
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Can-I-Deploy gate for Pact Broker.")
    p.add_argument("--broker-url", default=os.getenv("PACT_BROKER_URL") or os.getenv("PACT_BROKER_BASE_URL", ""))
    p.add_argument("--broker-token", default=os.getenv("PACT_BROKER_TOKEN", ""))
    p.add_argument("--consumer", help="Consumer name (matrix mode)")
    p.add_argument("--provider", help="Provider name (matrix mode)")
    p.add_argument("--pacticipant", help="Single pacticipant (environment mode)")
    p.add_argument("--version", required=True)
    p.add_argument("--tag", default="ci", help="Consumer version tag (matrix mode)")
    p.add_argument("--to-environment", default=None, help="PactFlow environment gate")
    p.add_argument("--record-deployment", default=None, metavar="ENV")
    p.add_argument("--strict", action="store_true", help="Fail on broker unreachable")
    p.add_argument("--timeout", type=int, default=30)
    return p.parse_args()


def main() -> int:
    _load_dotenv()
    args = parse_args()

    broker_url = (args.broker_url or "").strip()
    if not broker_url:
        log.info("ℹ️   PACT_BROKER_URL not set. Skipping can-i-deploy (safe no-op).")
        return 0

    token = (args.broker_token or "").strip() or None

    try:
        if args.record_deployment:
            ok = record_deployment(
                broker_url, token, args.pacticipant or args.provider or args.consumer,
                args.version, args.record_deployment, args.timeout,
            )
            return 0 if ok else 1

        if args.to_environment:
            if not args.pacticipant:
                log.error("--pacticipant is required with --to-environment")
                return 1
            ok = can_i_deploy_environment(
                broker_url, token, args.pacticipant, args.version,
                args.to_environment, args.timeout,
            )
            return 0 if ok else 1

        if args.consumer and args.provider:
            ok = can_i_deploy_matrix(
                broker_url, token, args.consumer, args.provider,
                args.version, args.tag, args.timeout,
            )
            return 0 if ok else 1

        log.error("Provide --consumer + --provider, or --pacticipant + --to-environment")
        return 1

    except httpx.RequestError as exc:
        log.warning("⚠️   Broker unreachable: %s", exc)
        return 3 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())