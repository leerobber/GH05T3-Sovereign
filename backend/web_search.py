"""Real web search for GH05T3 — DuckDuckGo, no API key required."""
from __future__ import annotations
import logging
from duckduckgo_search import DDGS

LOG = logging.getLogger("ghost.search")


def search(query: str, max_results: int = 5) -> list[dict]:
    """Return list of {title, url, snippet} from DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [{"title": r.get("title",""), "url": r.get("href",""), "snippet": r.get("body","")} for r in results]
    except Exception as e:
        LOG.warning("web search failed: %s", e)
        return []


def search_news(query: str, max_results: int = 5) -> list[dict]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        return [{"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("body",""), "date": r.get("date","")} for r in results]
    except Exception as e:
        LOG.warning("news search failed: %s", e)
        return []


def format_for_llm(results: list[dict]) -> str:
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet'][:200]}")
    return "\n\n".join(lines)
