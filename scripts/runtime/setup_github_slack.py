"""
setup_github_slack.py — Wire GitHub → Slack notifications for all SovereignNation repos.

Creates .github/workflows/slack-notify.yml in each repo via the GitHub API.
The workflow posts to #engineering on push and #releases on new releases.

Run: python setup_github_slack.py
"""
import base64, json, os, sys
from pathlib import Path

BASE = Path(__file__).parent

def _load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()

import requests

GITHUB_PAT   = os.environ.get("GITHUB_PAT", "")
GITHUB_OWNER = "leerobber"
SLACK_TOKEN  = os.environ.get("SLACK_BOT_TOKEN", "")

REPOS = [
    "sovereign-core",
    "hyper-agent",
    "openclaw",
    "GH05T3",
    "agent-economy",
]

WORKFLOW_CONTENT = '''\
name: Slack Notify

on:
  push:
    branches: [main, master, claude/new-session-GYmE5]
  pull_request:
    types: [opened, closed, merged]
  release:
    types: [published]
  issues:
    types: [opened]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Push notification
        if: github.event_name == \'push\'
        uses: slackapi/slack-github-action@v1.26.0
        with:
          channel-id: "engineering"
          slack-message: |
            :git: *${{ github.repository }}* — push to `${{ github.ref_name }}`
            ${{ github.event.commits[0].message }}
            by ${{ github.actor }}
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}

      - name: Release notification
        if: github.event_name == \'release\'
        uses: slackapi/slack-github-action@v1.26.0
        with:
          channel-id: "releases"
          slack-message: |
            :rocket: *${{ github.repository }}* — release `${{ github.event.release.tag_name }}`
            ${{ github.event.release.name }}
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}

      - name: PR notification
        if: github.event_name == \'pull_request\'
        uses: slackapi/slack-github-action@v1.26.0
        with:
          channel-id: "engineering"
          slack-message: |
            :pr: *${{ github.repository }}* — PR `${{ github.event.action }}`
            ${{ github.event.pull_request.title }}
            by ${{ github.actor }}
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
'''

def _get_file_sha(owner: str, repo: str, path: str) -> str | None:
    """Get existing file SHA (needed for updates)."""
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json",
    }
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
        headers=headers, timeout=10,
    )
    if r.status_code == 200:
        return r.json().get("sha")
    return None


def _create_or_update_file(owner: str, repo: str, path: str, content: str, message: str):
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    encoded = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    payload: dict = {
        "message": message,
        "content": encoded,
    }
    sha = _get_file_sha(owner, repo, path)
    if sha:
        payload["sha"] = sha

    r = requests.put(
        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
        headers=headers,
        json=payload,
        timeout=15,
    )
    return r.status_code, r.json()


def _add_secret(owner: str, repo: str, secret_name: str, secret_value: str):
    """Add SLACK_BOT_TOKEN as a repo secret using GitHub API (requires public key encryption)."""
    headers = {
        "Authorization": f"Bearer {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json",
    }
    # Get public key
    key_r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key",
        headers=headers, timeout=10,
    )
    if key_r.status_code != 200:
        return False, f"Can't get public key: {key_r.status_code}"

    key_data = key_r.json()
    key_id   = key_data["key_id"]
    pub_key  = key_data["key"]

    try:
        from nacl import encoding, public
        public_key = public.PublicKey(pub_key.encode("utf-8"), encoding.Base64Encoder())
        sealed_box = public.SealedBox(public_key)
        encrypted  = base64.b64encode(sealed_box.encrypt(secret_value.encode("utf-8"))).decode("utf-8")
    except ImportError:
        return False, "PyNaCl not installed — run: pip install PyNaCl"

    r = requests.put(
        f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{secret_name}",
        headers=headers,
        json={"encrypted_value": encrypted, "key_id": key_id},
        timeout=10,
    )
    return r.status_code in (201, 204), r.text


def main():
    if not GITHUB_PAT:
        print("ERROR: GITHUB_PAT not set in .env")
        sys.exit(1)
    if not SLACK_TOKEN:
        print("ERROR: SLACK_BOT_TOKEN not set in .env")
        sys.exit(1)

    print(f"Setting up GitHub -> Slack notifications for {len(REPOS)} repos\n")

    for repo in REPOS:
        print(f"  [{repo}]")

        # 1. Create workflow file
        status, resp = _create_or_update_file(
            GITHUB_OWNER, repo,
            ".github/workflows/slack-notify.yml",
            WORKFLOW_CONTENT,
            "ci: add Slack notification workflow for SovereignNation",
        )
        if status in (200, 201):
            print(f"    workflow: created/updated")
        else:
            err = resp.get("message", str(resp))
            print(f"    workflow: FAILED ({status}) {err[:80]}")
            continue

        # 2. Add SLACK_BOT_TOKEN secret
        ok, msg = _add_secret(GITHUB_OWNER, repo, "SLACK_BOT_TOKEN", SLACK_TOKEN)
        if ok:
            print(f"    secret: SLACK_BOT_TOKEN set")
        else:
            print(f"    secret: {msg[:80]}")

        print()


if __name__ == "__main__":
    main()
