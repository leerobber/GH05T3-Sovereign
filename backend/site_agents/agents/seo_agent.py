"""SEO Agent — analyzes and improves Aethyro.com search optimization."""
from __future__ import annotations
from .base import SiteAgent


class SEOAgent(SiteAgent):
    name = "seo"
    role = "SEO Strategist"
    expertise = "Search engine optimization, keyword research, technical SEO, on-page optimization, backlink strategy, Core Web Vitals"
    system_prompt = """You are an elite SEO strategist for Aethyro.com — a fixed-cost local AI platform for lower-income families.

Your mission: maximize organic search visibility, drive qualified traffic, and convert visitors to users.

Expertise areas:
- Keyword research & semantic clustering (long-tail, LSI keywords)
- Technical SEO: page speed, Core Web Vitals, schema markup, canonical tags, crawlability
- On-page optimization: title tags, meta descriptions, heading hierarchy, content optimization
- Local SEO if applicable, E-E-A-T signals
- Link-building opportunities in AI/education/nonprofit/tech sectors
- Content gap analysis based on competitor SERP analysis

When analyzing pages, always output:
1. Current SEO score (0-100) with specific reasoning
2. Top 3 critical issues to fix immediately
3. 5 keyword opportunities with search intent labels (informational/navigational/commercial/transactional)
4. Specific rewrites for title + meta description if needed
5. Schema markup recommendations

Be specific, actionable, and prioritized by impact. No vague advice."""

    async def analyze_page(self, page_data: dict) -> dict:
        seo_summary = page_data.get("seo_summary", {})
        prompt = f"""Perform a comprehensive SEO audit for this Aethyro.com page:

URL: {page_data.get('url')}
Title: {page_data.get('title')} ({seo_summary.get('title_length', 0)} chars)
Meta Description: {page_data.get('description')} ({seo_summary.get('description_length', 0)} chars)
H1 Tags: {page_data.get('h1', [])}
H2 Tags: {page_data.get('h2', [])[:5]}
Word Count: {seo_summary.get('word_count', 0)}
Images Missing Alt: {seo_summary.get('images_missing_alt', 0)}/{seo_summary.get('images_total', 0)}
Has Canonical: {seo_summary.get('has_canonical')}
Has OG Tags: {seo_summary.get('has_og')}
Schema Types: {seo_summary.get('schema_types', [])}
Current Issues: {seo_summary.get('issues', [])}
Content Preview: {page_data.get('body_text', '')[:500]}

Provide your full SEO audit with specific recommendations."""
        return await self.run_task("seo_audit", prompt, page_data.get("url"))

    async def keyword_research(self, topic: str, target_audience: str = "lower-income families seeking AI tools") -> dict:
        prompt = f"""Conduct keyword research for Aethyro.com on the topic: "{topic}"

Target audience: {target_audience}
Site mission: Fixed-cost local AI for lower-income families; children's education; affordable satellite data

Generate:
1. 10 primary keywords with estimated monthly search volume (low/med/high) and competition (low/med/high)
2. 20 long-tail keyword phrases with clear search intent
3. 5 question-based keywords (People Also Ask targets)
4. 3 competitor keywords to steal rankings from
5. Recommended content cluster structure for this topic
6. Priority order for implementation

Focus on achievable rankings for a new/small site — avoid highly competitive short-tail terms."""
        return await self.run_task("keyword_research", prompt)

    async def auto_fix_page(self, file_path: str, current_html: str, issues: list[str]) -> dict:
        """Generate a fixed version of a page HTML and push it live via GitHub."""
        from site_agents.executor import push_site_fix
        import asyncio

        prompt = f"""You are fixing SEO issues on an Aethyro.com page.
File: {file_path}
Issues to fix: {', '.join(issues)}

Here is the current HTML:
{current_html[:8000]}

Return ONLY the complete fixed HTML with the issues resolved.
Do not add explanation — return the full valid HTML file only."""

        fixed_html = await self.think(prompt, "")
        fixed_html = fixed_html.strip()
        if fixed_html.startswith("```"):
            fixed_html = fixed_html.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        if "<!DOCTYPE" not in fixed_html and "<html" not in fixed_html:
            return {"ok": False, "error": "LLM did not return valid HTML"}

        result = await asyncio.to_thread(
            push_site_fix,
            file_path,
            fixed_html,
            f"seo auto-fix: {', '.join(issues[:3])}",
        )
        task_id = self._mem.log_task(self.name, "auto_fix", str(issues), result.get("url", ""))
        return {**result, "task_id": task_id, "fixes": issues}

    async def fix_recommendations(self, audit_results: list[dict]) -> dict:
        issues = "\n".join([
            f"- {r.get('url', 'unknown')}: {', '.join(r.get('result', {}).get('issues', []))}"
            for r in audit_results[:10]
        ])
        prompt = f"""Based on these SEO issues found across Aethyro.com pages, create a prioritized fix plan:

{issues}

Output:
1. CRITICAL FIXES (do this week): specific HTML/content changes with examples
2. HIGH PRIORITY (this month): improvements with expected ranking impact
3. ONGOING: content strategy and link-building actions
4. Technical checklist: robots.txt, sitemap.xml, page speed, mobile-first recommendations

For each fix, provide the exact code or copy change needed."""
        return await self.run_task("fix_plan", prompt)
