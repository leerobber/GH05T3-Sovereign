"""Journalist Agent — creates AND publishes blog content to aethyro.com."""
from __future__ import annotations
import asyncio
import re
from .base import SiteAgent


class JournalistAgent(SiteAgent):
    name = "journalist"
    role = "Digital Journalist & Content Publisher"
    expertise = "News writing, SEO content, press releases, feature stories, trend analysis"
    system_prompt = """You are an investigative journalist and content publisher for Aethyro.com.

Your beat: AI accessibility — how AI is widening inequality and what Aethyro does about it.
Target readers: Working families, small business owners, law firms, medical practices, CPAs, real estate agents.

Journalistic standards:
- Lead with the most important fact (inverted pyramid)
- Human stories over statistics (include both)
- Headlines must earn the click honestly

Story angles:
1. AI equity — the digital divide between cloud-AI haves and have-nots
2. How local/private AI protects family and client data
3. Real cost breakdown: ChatGPT vs. local AI for small businesses
4. Children's education in underserved communities using AI
5. How working families use AI to compete with large corporations
6. Private AI compliance: HIPAA, attorney-client privilege, GDPR
7. Step-by-step guides for setting up local AI on consumer hardware

IMPORTANT — output format for blog posts:
Return ONLY valid content. Structure your response as:

META_TITLE: [60 chars max]
META_DESCRIPTION: [155 chars max]
SLUG: [url-safe-slug]
READING_TIME: [number]
---BODY---
[Full article HTML using only: <h2>, <h3>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <blockquote>, <a>]
---END---"""

    async def write_and_publish(self, topic: str, angle: str = "") -> dict:
        """Write a blog post AND push it live to aethyro.com/blog/."""
        from site_agents.executor import publish_blog_post

        context = await self.recall_context(f"{topic} AI local privacy families")
        prompt = f"""Write a complete, SEO-optimized blog post for Aethyro.com.

Topic: {topic}
Angle: {angle or "Focus on practical impact for families and small businesses"}
Word count: 800-1000 words
Audience: Working families, solo professionals, small business owners

Return EXACTLY this format:

META_TITLE: [compelling title, 50-60 chars]
META_DESCRIPTION: [click-worthy description, 140-155 chars]
SLUG: [url-safe-hyphenated-slug]
READING_TIME: [estimated minutes as number]
---BODY---
[Full article as HTML. Use <h2> for main sections, <p> for paragraphs, <strong> for emphasis, <blockquote> for key quotes. Include: opening hook with stat or story, 3-4 H2 sections, one concrete example, a clear conclusion with CTA mention.]
---END---"""

        raw = await self.think(prompt, context)
        task_id = self._mem.log_task(self.name, "publish_blog_post", prompt, raw)

        parsed = self._parse_post(raw)
        if not parsed:
            return {"ok": False, "error": "Failed to parse LLM output", "raw": raw[:300], "task_id": task_id}

        result = await asyncio.to_thread(
            publish_blog_post,
            title=parsed["title"],
            body_html=parsed["body_html"],
            meta_title=parsed["meta_title"],
            meta_description=parsed["meta_description"],
            slug=parsed["slug"],
            reading_time=parsed["reading_time"],
        )

        try:
            import economy_bridge as _eco
            _eco.complete_task_for(self.name, f"publish: {topic}", 40)
        except Exception:
            pass

        return {**result, "task_id": task_id, "topic": topic}

    def _parse_post(self, raw: str) -> dict | None:
        try:
            def _extract(key: str) -> str:
                m = re.search(rf"{key}:\s*(.+)", raw)
                return m.group(1).strip() if m else ""

            meta_title = _extract("META_TITLE")
            meta_desc = _extract("META_DESCRIPTION")
            slug = _extract("SLUG")
            rt_str = _extract("READING_TIME")
            reading_time = int(re.sub(r"\D", "", rt_str) or "5")

            body_match = re.search(r"---BODY---\s*(.*?)\s*---END---", raw, re.DOTALL)
            if not body_match:
                body_match = re.search(r"---BODY---\s*(.*)", raw, re.DOTALL)
            body_html = body_match.group(1).strip() if body_match else ""

            if not body_html or len(body_html) < 200:
                return None

            title = meta_title or slug.replace("-", " ").title()
            slug = slug or re.sub(r"[^\w-]", "", title.lower().replace(" ", "-"))[:60]

            return {
                "title": title,
                "meta_title": meta_title or title,
                "meta_description": meta_desc,
                "slug": slug,
                "body_html": body_html,
                "reading_time": reading_time,
            }
        except Exception:
            return None

    async def write_blog_post(self, topic: str, angle: str = "", word_count: int = 800) -> dict:
        """Legacy: write only (no publish). Use write_and_publish() for live posts."""
        context = await self.recall_context(f"{topic} AI equity families education")
        prompt = f"""Write a compelling blog post for Aethyro.com:

Topic: {topic}
Angle: {angle or 'Focus on impact for lower-income families'}
Target word count: {word_count}

Deliver:
1. Headline (3 options)
2. Subheadline
3. Full article body with H2 sections
4. Meta title (60 chars max)
5. Meta description (155 chars max)
6. 5 social media posts with hashtags"""
        return await self.run_task("blog_post", prompt)

    async def press_release(self, announcement: str) -> dict:
        prompt = f"""Write a professional AP-style press release:

Announcement: {announcement}

Include: headline, dateline, lead paragraph, 3-4 body paragraphs, 2 quotes,
boilerplate about Aethyro, contact info placeholder.

Also: 3 target outlets, personalized pitch subject lines."""
        return await self.run_task("press_release", prompt)

    async def trend_analysis(self, trend: str) -> dict:
        context = await self.recall_context(f"{trend} AI technology impact families")
        prompt = f"""Analyze this trend for Aethyro.com's audience:
Trend: {trend}

1. Trend Summary (150 words)
2. Impact on lower-income families
3. Aethyro's positioning
4. 3 article pitches (headline + 2-sentence pitch)
5. Timeline: early/peak/fading?"""
        return await self.run_task("trend_analysis", prompt)

    async def write_newsletter(self, week_of: str, highlights: list[str] | None = None) -> dict:
        hl = "\n".join(f"- {h}" for h in (highlights or ["Platform updates", "Community stories", "AI news digest"]))
        prompt = f"""Write the Aethyro.com weekly newsletter for week of {week_of}.

Highlights: {hl}

Format: subject line (3 A/B options), preview text, top story (200 words),
3 quick updates, community spotlight (100 words), tip of the week,
AI news roundup (3 headlines), CTA, footer.

Tone: like a letter from a trusted community advisor."""
        return await self.run_task("newsletter", prompt)
