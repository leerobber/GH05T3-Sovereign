"""
GH05T3 SENTINEL → Jira issue auto-creation.

When SENTINEL detects a threat or FORGE produces risky code, this module
creates a Jira issue in the configured project automatically.

Config (.env):
  JIRA_URL          = https://yourteam.atlassian.net
  JIRA_EMAIL        = your@email.com
  JIRA_API_TOKEN    = your Atlassian API token
  JIRA_PROJECT_KEY  = KAN  (or whatever your project key is)

All calls are best-effort (never raise). Silently no-ops when unconfigured.
"""
from __future__ import annotations

import base64
import logging
import os

LOG = logging.getLogger("ghost.jira_sentinel")


def _configured() -> bool:
    return all([
        os.environ.get("JIRA_URL"),
        os.environ.get("JIRA_EMAIL"),
        os.environ.get("JIRA_API_TOKEN"),
        os.environ.get("JIRA_PROJECT_KEY"),
    ])


def _auth_header() -> str:
    email = os.environ.get("JIRA_EMAIL", "")
    token = os.environ.get("JIRA_API_TOKEN", "")
    creds = base64.b64encode(f"{email}:{token}".encode()).decode()
    return f"Basic {creds}"


async def create_threat_issue(threat: str, source: str,
                               severity: str = "Medium") -> str | None:
    """
    Create a Jira issue for a detected threat.
    Returns the created issue key (e.g. 'KAN-42') or None.
    """
    if not _configured():
        return None
    try:
        import httpx
        url     = os.environ.get("JIRA_URL", "").rstrip("/")
        project = os.environ.get("JIRA_PROJECT_KEY", "KAN")

        payload = {
            "fields": {
                "project":     {"key": project},
                "summary":     f"[SENTINEL] Threat detected: {threat[:80]}",
                "description": {
                    "type":    "doc",
                    "version": 1,
                    "content": [{
                        "type":    "paragraph",
                        "content": [{
                            "type": "text",
                            "text": (
                                f"SENTINEL detected a potential injection threat.\n\n"
                                f"Threat pattern: {threat}\n"
                                f"Source agent: {source}\n"
                                f"Severity: {severity}\n\n"
                                f"Review the swarm bus log for full context."
                            ),
                        }],
                    }],
                },
                "issuetype": {"name": "Bug"},
                "priority":  {"name": severity},
            }
        }

        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.post(
                f"{url}/rest/api/3/issue",
                json=payload,
                headers={
                    "Authorization": _auth_header(),
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
            )
            if resp.status_code in (200, 201):
                key = resp.json().get("key", "")
                LOG.info("Jira issue created: %s", key)
                return key
            else:
                LOG.warning("Jira create failed: %d %s", resp.status_code, resp.text[:200])
    except Exception as e:
        LOG.debug("jira_sentinel error: %s", e)
    return None


async def create_code_risk_issue(risks: list[str], forge_preview: str) -> str | None:
    """Create a Jira issue for risky FORGE-generated code."""
    if not _configured():
        return None
    risk_list = ", ".join(risks)
    try:
        import httpx
        url     = os.environ.get("JIRA_URL", "").rstrip("/")
        project = os.environ.get("JIRA_PROJECT_KEY", "KAN")

        payload = {
            "fields": {
                "project":     {"key": project},
                "summary":     f"[SENTINEL] FORGE code risks: {risk_list[:80]}",
                "description": {
                    "type":    "doc",
                    "version": 1,
                    "content": [{
                        "type":    "paragraph",
                        "content": [{
                            "type": "text",
                            "text": (
                                f"SENTINEL flagged FORGE-generated code.\n\n"
                                f"Risks: {risk_list}\n\n"
                                f"Code preview:\n{forge_preview[:500]}"
                            ),
                        }],
                    }],
                },
                "issuetype": {"name": "Bug"},
                "priority":  {"name": "High"},
            }
        }

        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.post(
                f"{url}/rest/api/3/issue",
                json=payload,
                headers={
                    "Authorization": _auth_header(),
                    "Content-Type":  "application/json",
                    "Accept":        "application/json",
                },
            )
            if resp.status_code in (200, 201):
                key = resp.json().get("key", "")
                LOG.info("Jira code-risk issue created: %s", key)
                return key
    except Exception as e:
        LOG.debug("jira_sentinel code-risk error: %s", e)
    return None


def jira_status() -> dict:
    return {
        "configured": _configured(),
        "url":        os.environ.get("JIRA_URL", ""),
        "project":    os.environ.get("JIRA_PROJECT_KEY", ""),
    }
