#!/usr/bin/env python3
"""
Lightweight health check for the Pact Broker.

Exit codes:
  0 — broker healthy (HTTP < 400)
  1 — unhealthy or unreachable
  2 — PACT_BROKER_URL not configured (safe no-op)
"""
from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import httpx
except ImportError:
    log.error("httpx is not installed")
    sys.exit(1)


def check_broker_health(
    broker_url: str,
    token: str | None = None,
    timeout: float = 10.0,
) -> bool:
    base = broker_url.rstrip("/")
    health_url = f"{base}/health" if not base.endswith("/health") else base
    headers: dict[str, str] = {"Accept": "application/hal+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    start = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(health_url, headers=headers)
        latency_ms = int((time.monotonic() - start) * 1000)
        if resp.status_code < 400:
            log.info(
                "✅  Pact Broker is healthy: %s (%s OK, latency: %dms)",
                broker_url, resp.status_code, latency_ms,
            )
            return True
        log.warning(
            "⚠️   Pact Broker returned HTTP %s (%dms)",
            resp.status_code, latency_ms,
        )
        return False
    except httpx.RequestError as exc:
        log.warning("⚠️   Cannot connect to Pact Broker at %s: %s", broker_url, exc)
        return False


def main() -> int:
    url = os.environ.get("PACT_BROKER_URL") or os.environ.get("PACT_BROKER_BASE_URL")
    if not url:
        log.info("ℹ️   PACT_BROKER_URL is not set. Skipping broker health check.")
        return 2

    token = os.environ.get("PACT_BROKER_TOKEN") or None
    timeout = float(os.environ.get("BROKER_TIMEOUT", "10"))
    return 0 if check_broker_health(url, token=token, timeout=timeout) else 1


if __name__ == "__main__":
    sys.exit(main())