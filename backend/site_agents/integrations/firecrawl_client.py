"""Firecrawl integration — deep crawling and clean markdown extraction for Aethyro.com.

Falls back to httpx/BeautifulSoup if FIRECRAWL_API_KEY is not set.
Set FIRECRAWL_API_KEY in backend/.env to enable full Firecrawl power.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

LOG = logging.getLogger("site_agents.firecrawl")

FIRECRAWL_BASE = "https://api.firecrawl.dev/v2"
_API_KEY: str | None = None


def _key() -> str | None:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("FIRECRAWL_API_KEY", "").strip() or None
    return _API_KEY


def available() -> bool:
    return bool(_key())


def _headers() -> dict:
    return {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}


def scrape(url: str, formats: list[str] | None = None) -> dict:
    """Scrape a single URL. Returns {url, markdown, html, metadata, links}.
    Falls back to empty dict with 'fallback': True if no API key."""
    if not available():
        return {"fallback": True, "url": url, "markdown": "", "error": "no FIRECRAWL_API_KEY"}

    try:
        import requests
        payload = {
            "url": url,
            "formats": formats or ["markdown", "html", "links"],
            "onlyMainContent": True,
            "includeTags": ["h1", "h2", "h3", "p", "li", "a", "img", "title", "meta"],
            "excludeTags": ["script", "style", "nav", "footer", "cookie-banner"],
            "waitFor": 1000,
        }
        resp = requests.post(
            f"{FIRECRAWL_BASE}/scrape",
            json=payload,
            headers=_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            return {"fallback": False, "url": url, "markdown": "", "error": data.get("error", "unknown")}
        page = data.get("data", {})
        return {
            "fallback": False,
            "url": url,
            "markdown": page.get("markdown", ""),
            "html": page.get("html", ""),
            "links": page.get("links", []),
            "metadata": page.get("metadata", {}),
        }
    except Exception as e:
        LOG.warning("[firecrawl] scrape failed for %s: %s", url, e)
        return {"fallback": True, "url": url, "markdown": "", "error": str(e)}


def crawl(
    url: str,
    max_pages: int = 25,
    include_paths: list[str] | None = None,
    exclude_paths: list[str] | None = None,
) -> list[dict]:
    """Crawl a full site. Returns list of page dicts with markdown + metadata.
    Falls back to empty list if no API key."""
    if not available():
        LOG.info("[firecrawl] no API key — crawl unavailable")
        return []

    try:
        import requests

        # Start crawl job
        payload = {
            "url": url,
            "limit": max_pages,
            "scrapeOptions": {
                "formats": ["markdown", "links"],
                "onlyMainContent": True,
                "excludeTags": ["script", "style", "nav", "footer"],
            },
        }
        if include_paths:
            payload["includePaths"] = include_paths
        if exclude_paths:
            payload["excludePaths"] = exclude_paths

        resp = requests.post(
            f"{FIRECRAWL_BASE}/crawl",
            json=payload,
            headers=_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            LOG.warning("[firecrawl] crawl start failed: %s", data.get("error"))
            return []

        job_id = data.get("id")
        if not job_id:
            return []

        # Poll for results
        for attempt in range(30):
            time.sleep(3)
            status_resp = requests.get(
                f"{FIRECRAWL_BASE}/crawl/{job_id}",
                headers=_headers(),
                timeout=15,
            )
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("status") == "completed":
                pages = status.get("data", [])
                LOG.info("[firecrawl] crawl complete: %d pages", len(pages))
                return pages

            if status.get("status") in ("failed", "cancelled"):
                LOG.warning("[firecrawl] crawl %s: %s", status.get("status"), status.get("error"))
                return []

        LOG.warning("[firecrawl] crawl timed out after 90s")
        return []

    except Exception as e:
        LOG.warning("[firecrawl] crawl failed for %s: %s", url, e)
        return []


def map_site(url: str) -> list[str]:
    """Get all URLs on a site via Firecrawl /map endpoint.
    Returns list of URLs, or empty list if unavailable."""
    if not available():
        return []

    try:
        import requests
        resp = requests.post(
            f"{FIRECRAWL_BASE}/map",
            json={"url": url, "includeSubdomains": False, "limit": 100},
            headers=_headers(),
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("success"):
            return data.get("links", [])
    except Exception as e:
        LOG.warning("[firecrawl] map failed for %s: %s", url, e)
    return []


def markdown_to_page_data(fc_page: dict, base_url: str = "") -> dict:
    """Convert a Firecrawl page dict to our internal page_data format."""
    meta = fc_page.get("metadata", {})
    markdown = fc_page.get("markdown", "")
    word_count = len(markdown.split())

    return {
        "url": meta.get("url") or fc_page.get("url") or base_url,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "og_image": meta.get("ogImage", ""),
        "body_text": markdown[:8000],
        "markdown": markdown,
        "word_count": word_count,
        "links": fc_page.get("links", []),
        "source": "firecrawl",
    }
