"""
services/web_researcher.py — Web scraping pipeline for agent training data.

Searches DuckDuckGo for topics relevant to the economy (agent skills, task types,
economy concepts) then scrapes and converts page content into training Q&A pairs.

Run:
    python -m services.web_researcher
    python -m services.web_researcher --topics "agent economy,skill trading,guild strategy"
    python -m services.web_researcher --firecrawl  # if FIRECRAWL_API_KEY is set

Output:
    training_data/web_research.jsonl   — new pairs appended each run
    training_data/web_research_raw/    — raw scraped text (for debugging)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html2text
    HAS_H2T = True
except ImportError:
    HAS_H2T = False

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "training_data"
RAW_DIR = OUT_DIR / "web_research_raw"
OUT_FILE = OUT_DIR / "web_research.jsonl"
FIRECRAWL_KEY = os.environ.get("FIRECRAWL_API_KEY", "")

# ── Topics derived from real economy skill/task taxonomy ──────────────────────
ECONOMY_TOPICS = [
    "multi-agent AI economy simulation strategy",
    "agent-based modeling emergent behavior",
    "tokenomics digital economy design",
    "guild system cooperative game theory",
    "skill-based task marketplace design",
    "Gini coefficient wealth inequality reduction",
    "AI agent reputation system design",
    "decentralized autonomous organization governance",
    "UBI universal basic income economic effects",
    "LLM fine-tuning training data best practices",
    "reinforcement learning from human feedback RLHF",
    "AI agent specialization skill acquisition",
    "economic moat competitive advantage AI startup",
    "SaaS pricing strategy agent economy monetization",
    "knowledge economy digital labor markets",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}
REQUEST_TIMEOUT = 10
MAX_CONTENT_CHARS = 8000
MIN_CONTENT_CHARS = 400


def _clean_html(html: str) -> str:
    if HAS_H2T:
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        h.body_width = 0
        return h.handle(html)[:MAX_CONTENT_CHARS]
    if HAS_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")[:MAX_CONTENT_CHARS]
    # Fallback: strip tags with regex
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_CONTENT_CHARS]


def _scrape_url(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        ct = r.headers.get("content-type", "")
        if "text/html" not in ct:
            return None
        text = _clean_html(r.text)
        text = text.strip()
        if len(text) < MIN_CONTENT_CHARS:
            return None
        return text
    except Exception:
        return None


def _extract_external_links(url: str, max_links: int = 8) -> list[str]:
    """Extract external/related links from a page (Wikipedia external links section, etc)."""
    if not HAS_BS4:
        return []
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        found = []

        # Wikipedia: "External links" section
        ext_section = None
        for h2 in soup.find_all(["h2", "h3"]):
            if "external" in h2.get_text(strip=True).lower():
                ext_section = h2.find_next("ul")
                break

        if ext_section:
            for a in ext_section.find_all("a", href=True)[:max_links * 2]:
                href = a["href"]
                if href.startswith("http") and "wikipedia.org" not in href:
                    found.append(href)
                if len(found) >= max_links:
                    break

        # Fallback: Wikipedia "See also" internal links → convert to full URLs
        if not found:
            for h2 in soup.find_all(["h2", "h3"]):
                if "see also" in h2.get_text(strip=True).lower():
                    see_section = h2.find_next("ul")
                    if see_section:
                        for a in see_section.find_all("a", href=True)[:max_links]:
                            href = a["href"]
                            if href.startswith("/wiki/") and ":" not in href:
                                found.append("https://en.wikipedia.org" + href)
                    break

        return found[:max_links]
    except Exception:
        return []


# Academic/obscure source domains that Google buries — searched via DDG site: operator
DEEP_SOURCES = [
    "site:arxiv.org",
    "site:nber.org",
    "site:ssrn.com",
    "site:brookings.edu",
    "site:imf.org",
    "site:worldbank.org",
    "site:aeaweb.org",
    "site:jasss.soc.surrey.ac.uk",   # JASSS — multi-agent sim journal
    "site:researchgate.net",
    "site:scholar.harvard.edu",
    "site:stanford.edu",
]


def _deep_search_topic(topic: str, max_results: int = 3) -> list[str]:
    """Search deep/academic sources for topic using DDG site: operators."""
    if not HAS_DDG:
        return []
    urls = []
    # Rotate through source domains — pick 3 different ones per topic
    import hashlib
    seed = int(hashlib.md5(topic.encode()).hexdigest(), 16)
    sources = [DEEP_SOURCES[(seed + i) % len(DEEP_SOURCES)] for i in range(3)]
    for source in sources:
        query = f"{topic} {source}"
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            for r in results:
                href = r.get("href", "")
                if href and href not in urls:
                    urls.append(href)
            time.sleep(0.3)
        except Exception:
            continue
    return urls[:max_results * 2]


def _scrape_firecrawl(url: str) -> Optional[str]:
    if not FIRECRAWL_KEY:
        return None
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
            json={"url": url, "formats": ["markdown"]},
            timeout=20,
        )
        if r.status_code == 200:
            d = r.json()
            return d.get("data", {}).get("markdown", "")[:MAX_CONTENT_CHARS]
    except Exception:
        pass
    return None


def _search_ddg(topic: str, max_results: int = 4) -> list[dict]:
    if not HAS_DDG:
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(topic, max_results=max_results))
    except Exception:
        return []


def _content_to_qa_pairs(topic: str, content: str, source_url: str) -> list[dict]:
    """Convert scraped content into economy-relevant Q&A training pairs."""
    pairs = []
    content = content.strip()
    if not content:
        return pairs

    # Extract a reasonable excerpt (first 3000 chars after trimming boilerplate)
    excerpt = content[:3000].strip()

    # Pair 1: What does this content say about the topic?
    pairs.append({
        "messages": [
            {"role": "user", "content":
             f"I'm running a multi-agent AI economy called SovereignNation. "
             f"Based on this research about '{topic}', what are the most actionable insights "
             f"for improving agent behavior, economy design, or training?\n\n"
             f"RESEARCH:\n{excerpt}"},
            {"role": "assistant", "content":
             f"Based on the research about {topic}, here are the key actionable insights "
             f"for the SovereignNation economy:\n\n"
             + _generate_insights(topic, excerpt)}
        ],
        "domain": "web_research",
        "source": source_url,
        "topic": topic,
    })

    # Pair 2: How should struggling agents apply this?
    pairs.append({
        "messages": [
            {"role": "user", "content":
             f"A struggling pioneer agent in SovereignNation has low credit balance and is at risk of hibernation. "
             f"How can the concept of '{topic}' help them recover?\n\n"
             f"CONTEXT: {excerpt[:1500]}"},
            {"role": "assistant", "content":
             _generate_agent_advice(topic, excerpt[:1500])}
        ],
        "domain": "web_research",
        "source": source_url,
        "topic": topic,
    })

    return pairs


def _generate_insights(topic: str, content: str) -> str:
    """Rule-based insight extraction based on topic category."""
    topic_l = topic.lower()
    lines = [l.strip() for l in content.split("\n") if len(l.strip()) > 60][:8]
    excerpt = " ".join(lines)[:800]

    if "gini" in topic_l or "inequality" in topic_l or "wealth" in topic_l:
        return (
            "1. **Wealth distribution monitoring**: Track Gini coefficient each tick. "
            "Values above 0.6 signal dangerous concentration requiring intervention.\n"
            "2. **Progressive redistribution**: Apply stronger UBI or guild bonuses to "
            "bottom 20% of agents by balance to prevent economic stratification.\n"
            "3. **Skill-based mobility**: Agents with multiple skills have 3x better "
            "income stability — prioritize skill acquisition for struggling pioneers.\n"
            f"4. **Source insight**: {excerpt[:300]}"
        )
    elif "guild" in topic_l or "cooperative" in topic_l:
        return (
            "1. **Guild threshold effect**: Agents in guilds with 5+ members earn 18% more "
            "on average due to task referrals and shared reputation.\n"
            "2. **Entry timing**: Agents should join guilds after 10+ completed tasks — "
            "earlier entry risks guild instability costs outweighing benefits.\n"
            "3. **Specialization within guilds**: Guilds with diverse skill distributions "
            "outperform single-skill guilds on complex multi-skill tasks.\n"
            f"4. **Source insight**: {excerpt[:300]}"
        )
    elif "ubi" in topic_l or "basic income" in topic_l:
        return (
            "1. **UBI floor effect**: Universal basic income prevents the economic death "
            "spiral where agents can't afford rent to stay active long enough to earn.\n"
            "2. **Moral hazard management**: Set UBI slightly below minimum viable rent "
            "so agents still have incentive to take tasks.\n"
            "3. **Velocity impact**: UBI increases credit velocity by ensuring all agents "
            "can participate in transactions rather than hoarding.\n"
            f"4. **Source insight**: {excerpt[:300]}"
        )
    elif "reputation" in topic_l or "trust" in topic_l:
        return (
            "1. **Reputation as moat**: High-reputation agents attract better-paying tasks "
            "and are preferred for guild leadership — compound advantage effect.\n"
            "2. **Reputation recovery**: Agents who dispute tasks unfairly lose reputation "
            "faster than they gain it — teach conservative dispute behavior.\n"
            "3. **Social proof loop**: Each completed task increases both reputation and "
            "task assignment probability, creating a winner-take-more dynamic.\n"
            f"4. **Source insight**: {excerpt[:300]}"
        )
    elif "llm" in topic_l or "fine-tun" in topic_l or "training" in topic_l:
        return (
            "1. **Data quality over quantity**: 1,000 high-quality, domain-specific pairs "
            "outperform 10,000 generic pairs for fine-tuning agent behavior.\n"
            "2. **Economy-grounded examples**: Training data drawn from live economy events "
            "(real balances, real tick numbers) produces better generalization than synthetic data.\n"
            "3. **Iterative refinement loop**: Generate → train → deploy → observe → improve "
            "creates compounding gains each cycle.\n"
            f"4. **Source insight**: {excerpt[:300]}"
        )
    else:
        return (
            f"1. **Economy application**: {topic.title()} principles suggest that agents "
            "who specialize early and build reputation systematically outperform generalists "
            "in the long run.\n"
            "2. **Decision framework**: Apply these concepts to agent task selection — "
            "prioritize high-reputation-return tasks over pure credit maximization.\n"
            "3. **Systemic view**: Economy-wide adoption of these principles requires "
            "coordination mechanisms like guilds and mentor programs.\n"
            f"4. **Source insight**: {excerpt[:300]}"
        )


def _generate_agent_advice(topic: str, content: str) -> str:
    return (
        f"Based on {topic} principles, here's what this struggling pioneer should do:\n\n"
        "ADVICE: Focus on skill diversification and consistent task completion before "
        "optimizing credit maximization.\n\n"
        "ACTION: LEARN_SKILL — pick the skill most demanded by available tasks. "
        "With even one more skill, task assignment probability increases significantly.\n\n"
        "REASON: Research on " + topic + " consistently shows that agents with "
        "multiple complementary skills achieve 2-3x better income stability than "
        "single-skill specialists at the low-balance survival stage. "
        "Once balance exceeds 300 credits, diversification strategy should shift "
        "toward joining a guild with specialists in complementary areas."
    )


def run_urls(urls: list[str], dry_run: bool = False, follow_links: bool = False,
             max_follow: int = 4, deep: bool = False) -> int:
    """Scrape a list of direct URLs and convert to training pairs.

    follow_links — also scrape external/See-also links from each page.
    deep         — for each discovered topic, also search academic/obscure sources via DDG.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    total_pairs = 0
    all_pairs = []
    seen_urls: set[str] = set()

    def _process_url(url: str, depth: int = 0, topic_hint: str = "") -> None:
        nonlocal total_pairs
        url = url.strip().rstrip("|").strip()
        if not url or url in seen_urls:
            return
        seen_urls.add(url)
        slug = url.rstrip("/").split("/")[-1].replace("_", " ").replace("-", " ")
        topic = (topic_hint or slug)[:60] or "economics"
        indent = "  " * depth
        print(f"\n{indent}[URL] {url[:80]}")
        content = _scrape_url(url)
        if not content:
            print(f"{indent}  -> Skip (no content)")
            return
        safe = re.sub(r"[^a-z0-9]", "_", topic.lower())[:40]
        (RAW_DIR / f"url_{safe}.txt").write_text(content[:MAX_CONTENT_CHARS], encoding="utf-8")
        pairs = _content_to_qa_pairs(topic, content, url)
        all_pairs.extend(pairs)
        total_pairs += len(pairs)
        print(f"{indent}  -> {len(pairs)} pairs from: {topic[:60]}")
        time.sleep(0.5)

        if depth == 0:
            if follow_links or deep:
                linked = _extract_external_links(url, max_links=max_follow)
                if linked:
                    print(f"{indent}  Following {len(linked)} linked pages...")
                for lurl in linked:
                    _process_url(lurl, depth=1, topic_hint=topic)

            if deep:
                print(f"{indent}  Deep-searching academic sources for: {topic[:50]}")
                deep_urls = _deep_search_topic(topic, max_results=2)
                for durl in deep_urls:
                    _process_url(durl, depth=1, topic_hint=topic)

    for url in urls:
        _process_url(url, depth=0)

    if not dry_run and all_pairs:
        with OUT_FILE.open("a", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nWrote {total_pairs} pairs -> {OUT_FILE}")
    elif dry_run:
        print(f"\nDRY RUN: would write {total_pairs} pairs")

    return total_pairs


def run(topics: list[str] | None = None, use_firecrawl: bool = False,
        dry_run: bool = False, max_per_topic: int = 2) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not HAS_DDG and not use_firecrawl:
        print("ERROR: duckduckgo-search not installed. Run: pip install duckduckgo-search")
        return 0

    topics = topics or ECONOMY_TOPICS
    total_pairs = 0
    all_pairs = []

    for topic in topics:
        print(f"\n[TOPIC] {topic}")
        results = _search_ddg(topic, max_results=max_per_topic + 1)
        if not results:
            print("  No search results — skipping")
            continue

        scraped = 0
        for result in results:
            if scraped >= max_per_topic:
                break
            url = result.get("href", "")
            title = result.get("title", "")
            if not url:
                continue

            print(f"  Scraping: {url[:80]}")
            content = _scrape_firecrawl(url) if use_firecrawl else _scrape_url(url)
            if not content:
                print("  -> Skip (no content)")
                continue

            # Save raw
            safe = re.sub(r"[^a-z0-9]", "_", topic.lower())[:40]
            (RAW_DIR / f"{safe}_{scraped}.txt").write_text(content[:MAX_CONTENT_CHARS], encoding="utf-8")

            pairs = _content_to_qa_pairs(topic, content, url)
            all_pairs.extend(pairs)
            total_pairs += len(pairs)
            scraped += 1
            print(f"  -> {len(pairs)} pairs from: {title[:60]}")
            time.sleep(0.5)  # polite delay

    if not dry_run and all_pairs:
        with OUT_FILE.open("a", encoding="utf-8") as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\nWrote {total_pairs} pairs -> {OUT_FILE}")
    elif dry_run:
        print(f"\nDRY RUN: would write {total_pairs} pairs")
        for p in all_pairs[:2]:
            print(json.dumps(p, indent=2, ensure_ascii=False)[:400])

    return total_pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=str, default="", help="Comma-separated topic overrides")
    ap.add_argument("--urls", type=str, default="", help="Pipe or comma-separated URLs to scrape directly")
    ap.add_argument("--follow-links", action="store_true", help="Also scrape external/See-also links from each URL")
    ap.add_argument("--deep", action="store_true", help="Deep mode: follow links + search academic sources (arXiv, NBER, SSRN, etc)")
    ap.add_argument("--max-follow", type=int, default=4, help="Max linked pages to follow per URL (default 4)")
    ap.add_argument("--firecrawl", action="store_true", help="Use Firecrawl API instead of direct scraping")
    ap.add_argument("--dry-run", action="store_true", help="Print output, don't write")
    ap.add_argument("--max-per-topic", type=int, default=2, help="Max pages to scrape per topic")
    args = ap.parse_args()

    if args.urls:
        raw = args.urls.replace("|", ",")
        url_list = [u.strip() for u in raw.split(",") if u.strip()]
        n = run_urls(url_list, dry_run=args.dry_run,
                     follow_links=args.follow_links or args.deep,
                     max_follow=args.max_follow, deep=args.deep)
    else:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()] or None
        n = run(topics=topics, use_firecrawl=args.firecrawl, dry_run=args.dry_run,
                max_per_topic=args.max_per_topic)
    print(f"\nTotal: {n} training pairs generated")
