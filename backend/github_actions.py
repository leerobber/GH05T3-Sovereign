"""Real GitHub automation for GH05T3 — create issues, PRs, push files, read repo."""
from __future__ import annotations
import logging
import os
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=True)
LOG = logging.getLogger("ghost.github")

PAT   = os.environ.get("GITHUB_PAT", "")
REPO  = os.environ.get("GITHUB_REPO", "")   # owner/repo
BRANCH = os.environ.get("GITHUB_BRANCH", "main")
_BASE = "https://api.github.com"


def _headers() -> dict:
    return {"Authorization": f"Bearer {PAT}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


async def create_issue(title: str, body: str, labels: list[str] | None = None) -> dict:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{_BASE}/repos/{REPO}/issues",
                         headers=_headers(),
                         json={"title": title, "body": body, "labels": labels or []})
        r.raise_for_status()
        j = r.json()
        return {"ok": True, "number": j["number"], "url": j["html_url"]}


async def list_issues(state: str = "open", limit: int = 10) -> list[dict]:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.get(f"{_BASE}/repos/{REPO}/issues",
                        headers=_headers(), params={"state": state, "per_page": limit})
        r.raise_for_status()
        return [{"number": i["number"], "title": i["title"],
                 "state": i["state"], "url": i["html_url"]} for i in r.json()]


async def close_issue(number: int, comment: str = "") -> dict:
    async with httpx.AsyncClient(timeout=20) as c:
        if comment:
            await c.post(f"{_BASE}/repos/{REPO}/issues/{number}/comments",
                         headers=_headers(), json={"body": comment})
        r = await c.patch(f"{_BASE}/repos/{REPO}/issues/{number}",
                          headers=_headers(), json={"state": "closed"})
        r.raise_for_status()
        return {"ok": True, "number": number}


async def push_file(path: str, content: str, message: str) -> dict:
    """Create or update a file in the repo."""
    import base64
    encoded = base64.b64encode(content.encode()).decode()
    async with httpx.AsyncClient(timeout=30) as c:
        # Get current SHA if file exists
        sha = None
        existing = await c.get(f"{_BASE}/repos/{REPO}/contents/{path}",
                               headers=_headers(), params={"ref": BRANCH})
        if existing.status_code == 200:
            sha = existing.json().get("sha")
        payload = {"message": message, "content": encoded, "branch": BRANCH}
        if sha:
            payload["sha"] = sha
        r = await c.put(f"{_BASE}/repos/{REPO}/contents/{path}",
                        headers=_headers(), json=payload)
        r.raise_for_status()
        return {"ok": True, "path": path, "url": r.json().get("content", {}).get("html_url", "")}


async def repo_stats() -> dict:
    """Return repo metadata."""
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(f"{_BASE}/repos/{REPO}", headers=_headers())
        r.raise_for_status()
        j = r.json()
        return {"name": j["full_name"], "stars": j["stargazers_count"],
                "open_issues": j["open_issues_count"], "default_branch": j["default_branch"],
                "url": j["html_url"]}


async def create_pr(title: str, body: str, head: str, base: str = "main") -> dict:
    async with httpx.AsyncClient(timeout=20) as c:
        r = await c.post(f"{_BASE}/repos/{REPO}/pulls",
                         headers=_headers(),
                         json={"title": title, "body": body, "head": head, "base": base})
        r.raise_for_status()
        j = r.json()
        return {"ok": True, "number": j["number"], "url": j["html_url"]}
