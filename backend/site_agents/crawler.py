"""Aethyro.com site crawler — fetches pages, extracts structured content for RAG."""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

TARGET_DOMAIN = "aethyro.com"
BASE_URL = f"https://{TARGET_DOMAIN}"

HEADERS = {
    "User-Agent": "Aethyro-SiteAgent/1.0 (internal improvement bot; contact leer4030@gmail.com)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class PageData:
    url: str
    title: str = ""
    description: str = ""
    h1: list[str] = field(default_factory=list)
    h2: list[str] = field(default_factory=list)
    h3: list[str] = field(default_factory=list)
    body_text: str = ""
    links: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)   # {src, alt, title}
    keywords: list[str] = field(default_factory=list)
    canonical: str = ""
    robots: str = ""
    og_title: str = ""
    og_description: str = ""
    og_image: str = ""
    schema_types: list[str] = field(default_factory=list)
    word_count: int = 0
    status_code: int = 0
    error: str = ""
    crawled_at: float = field(default_factory=time.time)

    @property
    def page_id(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()[:12]

    def to_rag_text(self) -> str:
        """Flat text blob for embedding — includes all semantic content."""
        parts = [
            f"URL: {self.url}",
            f"Title: {self.title}",
            f"Description: {self.description}",
        ]
        if self.h1:
            parts.append("H1: " + " | ".join(self.h1))
        if self.h2:
            parts.append("H2: " + " | ".join(self.h2[:6]))
        if self.h3:
            parts.append("H3: " + " | ".join(self.h3[:10]))
        if self.keywords:
            parts.append("Keywords: " + ", ".join(self.keywords))
        if self.body_text:
            parts.append("Content: " + self.body_text[:3000])
        return "\n".join(parts)

    def seo_summary(self) -> dict:
        issues = []
        if not self.title:
            issues.append("Missing title tag")
        elif len(self.title) < 30:
            issues.append(f"Title too short ({len(self.title)} chars, aim 50-60)")
        elif len(self.title) > 60:
            issues.append(f"Title too long ({len(self.title)} chars, aim 50-60)")

        if not self.description:
            issues.append("Missing meta description")
        elif len(self.description) < 100:
            issues.append(f"Meta description too short ({len(self.description)} chars, aim 150-160)")
        elif len(self.description) > 160:
            issues.append(f"Meta description too long ({len(self.description)} chars)")

        if not self.h1:
            issues.append("No H1 heading found")
        elif len(self.h1) > 1:
            issues.append(f"Multiple H1 tags ({len(self.h1)}) — should be exactly 1")

        imgs_missing_alt = [i for i in self.images if not i.get("alt")]
        if imgs_missing_alt:
            issues.append(f"{len(imgs_missing_alt)} images missing alt text")

        if self.word_count < 300:
            issues.append(f"Low word count ({self.word_count} words, aim 500+)")

        if not self.canonical:
            issues.append("No canonical URL specified")

        return {
            "url": self.url,
            "title_length": len(self.title),
            "description_length": len(self.description),
            "h1_count": len(self.h1),
            "word_count": self.word_count,
            "images_total": len(self.images),
            "images_missing_alt": len(imgs_missing_alt),
            "has_canonical": bool(self.canonical),
            "has_og": bool(self.og_title),
            "schema_types": self.schema_types,
            "issues": issues,
            "score": max(0, 100 - len(issues) * 12),
        }


async def fetch_page_firecrawl(url: str) -> PageData | None:
    """Try Firecrawl for richer content extraction. Returns None if unavailable."""
    try:
        from site_agents.integrations import firecrawl_client as fc
        if not fc.available():
            return None
        result = fc.scrape(url)
        if result.get("fallback") or not result.get("markdown"):
            return None
        meta = result.get("metadata", {})
        page = PageData(url=url, status_code=200)
        page.title = meta.get("title", "")
        page.description = meta.get("description", "")
        page.og_image = meta.get("ogImage", "")
        page.links = [lnk for lnk in result.get("links", []) if TARGET_DOMAIN in lnk]
        markdown = result.get("markdown", "")
        # Extract headings from markdown
        for line in markdown.split("\n"):
            stripped = line.lstrip("#").strip()
            if line.startswith("# ") and stripped:
                page.h1.append(stripped)
            elif line.startswith("## ") and stripped:
                page.h2.append(stripped)
            elif line.startswith("### ") and stripped:
                page.h3.append(stripped)
        page.body_text = markdown[:8000]
        page.word_count = len(markdown.split())
        return page
    except Exception:
        return None


async def fetch_page(url: str, timeout: float = 15.0) -> PageData:
    # Try Firecrawl first for JS-rendered content and cleaner markdown
    fc_page = await fetch_page_firecrawl(url)
    if fc_page is not None:
        return fc_page
    page = PageData(url=url)
    try:
        async with httpx.AsyncClient(
            headers=HEADERS, timeout=timeout, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            page.status_code = resp.status_code
            if resp.status_code != 200:
                page.error = f"HTTP {resp.status_code}"
                return page
            _parse_html(resp.text, page, url)
    except Exception as e:
        page.error = str(e)
    return page


def _parse_html(html: str, page: PageData, base_url: str) -> None:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()

    # Meta tags
    title_tag = soup.find("title")
    page.title = title_tag.get_text(strip=True) if title_tag else ""

    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        content = meta.get("content", "")
        if name in ("description", "og:description"):
            if not page.description:
                page.description = content
        if name == "keywords":
            page.keywords = [k.strip() for k in content.split(",") if k.strip()]
        if name == "robots":
            page.robots = content
        if name == "og:title":
            page.og_title = content
        if name == "og:description":
            page.og_description = content
        if name == "og:image":
            page.og_image = content

    link_tag = soup.find("link", rel=lambda r: r and "canonical" in r)
    if link_tag:
        page.canonical = link_tag.get("href", "")

    # Headings
    page.h1 = [h.get_text(strip=True) for h in soup.find_all("h1")]
    page.h2 = [h.get_text(strip=True) for h in soup.find_all("h2")]
    page.h3 = [h.get_text(strip=True) for h in soup.find_all("h3")]

    # Images
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if src:
            page.images.append({
                "src": urljoin(base_url, src),
                "alt": img.get("alt", ""),
                "title": img.get("title", ""),
            })

    # Links (same domain)
    parsed_base = urlparse(base_url)
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        if parsed.netloc.endswith(TARGET_DOMAIN) and parsed.scheme in ("http", "https"):
            page.links.append(href)
    page.links = list(set(page.links))

    # Schema.org types
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "{}")
            t = data.get("@type")
            if t:
                page.schema_types.append(t if isinstance(t, str) else str(t))
        except Exception:
            pass

    # Body text
    body_text = soup.get_text(separator=" ", strip=True)
    body_text = re.sub(r"\s+", " ", body_text)
    page.body_text = body_text[:8000]
    page.word_count = len(body_text.split())


async def crawl_site(
    start_url: str = BASE_URL,
    max_pages: int = 30,
    delay: float = 0.8,
) -> list[PageData]:
    """BFS crawl of aethyro.com. Returns list of PageData."""
    visited: set[str] = set()
    queue: list[str] = [start_url]
    pages: list[PageData] = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        page = await fetch_page(url)
        pages.append(page)

        if not page.error:
            for link in page.links:
                if link not in visited and link not in queue:
                    queue.append(link)

        if delay > 0:
            time.sleep(delay)

    return pages
