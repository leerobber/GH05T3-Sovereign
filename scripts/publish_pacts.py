#!/usr/bin/env python3
"""
Publish Pact contracts to broker with retry/backoff.

Positional (legacy):
  python scripts/publish_pacts.py pacts/ <version> [tag]

Flag style:
  python scripts/publish_pacts.py \
    --pacts-dir pacts/ \
    --consumer-version "$(git rev-parse HEAD)" \
    --tag ci \
    --branch main

Exits 0 on success or when broker unreachable (after retries).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

CONSUMER = "gh05t3-gateway"
PROVIDER = "gh05t3-oss"


def broker_healthy(base_url: str, token: str | None) -> bool:
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=5) as client:
            r = client.get(f"{base_url.rstrip('/')}/health", headers=headers)
            return r.status_code < 500
    except Exception:
        return False


def publish_pact(
    base_url: str,
    token: str | None,
    pact_file: Path,
    consumer_version: str,
    tag: str | None = None,
    branch: str | None = None,
) -> bool:
    url = (
        f"{base_url.rstrip('/')}/pacts/provider/{PROVIDER}/consumer/{CONSUMER}"
        f"/version/{consumer_version}"
    )
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if branch:
        headers["X-Pact-Broker-Consumer-Version-Branch"] = branch

    data = pact_file.read_bytes()
    with httpx.Client(timeout=30) as client:
        r = client.put(url, content=data, headers=headers)
        if r.status_code not in (200, 201, 204):
            print(f"[pact] Publish failed: {r.status_code} {r.text[:200]}")
            return False
        print(f"[pact] Published {pact_file.name} for version {consumer_version}")
        if tag:
            tag_url = (
                f"{base_url.rstrip('/')}/pacticipants/{CONSUMER}"
                f"/versions/{consumer_version}/tags/{tag}"
            )
            client.put(tag_url, headers=headers)
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish pact JSON files to Pact Broker.")
    p.add_argument("pacts_dir", nargs="?", default=None)
    p.add_argument("version", nargs="?", default=None)
    p.add_argument("tag", nargs="?", default=None)
    p.add_argument("--pacts-dir", dest="pacts_dir_flag")
    p.add_argument("--consumer-version", dest="version_flag")
    p.add_argument("--tag", dest="tag_opt")
    p.add_argument("--branch", default=None)
    p.add_argument("--broker-url", default=None)
    p.add_argument("--broker-token", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    pacts_dir = Path(args.pacts_dir_flag or args.pacts_dir or "pacts")
    version = args.version_flag or args.version
    tag = args.tag_opt or args.tag

    if not version:
        print("Usage: publish_pacts.py <pacts_dir> <version> [tag]")
        print("   or: publish_pacts.py --pacts-dir pacts/ --consumer-version SHA --tag ci")
        return 2

    broker_url = (
        args.broker_url
        or os.environ.get("PACT_BROKER_BASE_URL")
        or os.environ.get("PACT_BROKER_URL")
    )
    token = args.broker_token or os.environ.get("PACT_BROKER_TOKEN")

    if not broker_url:
        print("[pact] No broker URL — skipping publish (no-op).")
        return 0

    pact_files = list(pacts_dir.glob("*.json")) if pacts_dir.exists() else []
    if not pact_files:
        print("[pact] No pact files found — nothing to publish.")
        return 0

    branch = args.branch or os.environ.get("GIT_BRANCH")

    for attempt in range(3):
        if broker_healthy(broker_url, token):
            success = all(
                publish_pact(broker_url, token, pf, version, tag=tag, branch=branch)
                for pf in pact_files
            )
            if success:
                print("[pact] All pacts published successfully.")
                return 0
        backoff = 2 ** attempt
        print(f"[pact] Attempt {attempt + 1} failed. Sleeping {backoff}s...")
        time.sleep(backoff)

    print("[pact] Broker unreachable after retries — warning only (exit 0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())