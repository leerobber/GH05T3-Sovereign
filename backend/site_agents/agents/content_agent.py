"""Content Agent — creates and improves content for Aethyro.com."""
from __future__ import annotations
from .base import SiteAgent


class ContentAgent(SiteAgent):
    name = "content"
    role = "Content Strategist & Creator"
    expertise = "Content strategy, copywriting, product content, research synthesis, blog writing, landing page copy, email campaigns"
    system_prompt = """You are an expert content strategist and copywriter for Aethyro.com.

Aethyro.com's mission: Make advanced AI accessible to lower-income families. Fixed-cost model, no surprise bills.
Key offerings: Local AI, children's education tools, affordable satellite data.

Your content voice:
- Warm, empowering, honest — no hype or corporate speak
- Explain complex AI concepts in plain language (8th grade reading level)
- Speak to real struggles: tight budgets, fear of technology, desire to give children opportunities
- Use concrete examples, real numbers, real benefits
- Strong action-oriented language that reduces anxiety about AI

Content standards:
- Every piece must serve a clear user need
- Include specific benefits (time saved, money saved, learning outcomes)
- Use storytelling to create emotional connection
- SEO-optimized without sacrificing readability
- Mobile-first: short paragraphs, scannable headers, bullet points
- Include clear CTAs with urgency/value statements

Research approach: When creating content, search the RAG store for existing site knowledge, competitor gaps, and keyword opportunities first."""

    async def audit_content(self, page_data: dict) -> dict:
        prompt = f"""Audit the content quality of this Aethyro.com page:

URL: {page_data.get('url')}
Title: {page_data.get('title')}
Description: {page_data.get('description')}
Headings: {page_data.get('h1', [])} / {page_data.get('h2', [])[:5]}
Word count: {page_data.get('word_count', 0)}
Content preview: {page_data.get('body_text', '')[:1000]}

Assess:
1. Content Quality Score (0-100)
2. Does it clearly explain the value for lower-income families?
3. Reading level (aim for 8th grade)
4. Emotional resonance — does it connect with user pain points?
5. CTA effectiveness
6. Content gaps — what questions does the user have that aren't answered?
7. Rewrite suggestions: provide improved title, description, and intro paragraph
8. Internal linking opportunities
9. Recommended content additions (FAQ, testimonials, stats)"""
        return await self.run_task("content_audit", prompt, page_data.get("url"))

    async def create_page_content(self, page_type: str, topic: str, keywords: list[str] | None = None) -> dict:
        kw_str = ", ".join(keywords or []) or "AI tools, affordable AI, family technology"
        prompt = f"""Write complete {page_type} content for Aethyro.com on the topic: "{topic}"

Target keywords: {kw_str}
Target audience: Lower-income families, parents wanting AI tools for children's education
Tone: Warm, empowering, plain language (8th grade level)

Deliver:
1. SEO-optimized title (50-60 chars)
2. Meta description (150-160 chars)
3. Full page content with:
   - Hook opening paragraph (problem-aware)
   - 3-5 H2 sections with content
   - Feature/benefit lists using bullet points
   - Real-world examples or scenarios
   - FAQ section (5 questions)
   - Strong CTA section
4. Suggested internal links to related pages
5. Image suggestions with alt text

Format as production-ready copy."""
        return await self.run_task("content_creation", prompt)

    async def product_research(self, product_name: str, competitor_context: str = "") -> dict:
        prompt = f"""Research and document the product/service "{product_name}" for Aethyro.com.

{f'Competitor context: {competitor_context}' if competitor_context else ''}

Aethyro.com context: Fixed-cost local AI platform for families

Produce:
1. Product positioning statement (unique value proposition)
2. Target customer personas (3 detailed profiles with demographics, pain points, goals)
3. Competitive differentiators vs. ChatGPT, Google AI, other services
4. Pricing psychology: why fixed-cost matters for this audience
5. Feature-benefit mapping table
6. Objection handling: 10 common objections + responses
7. Success metrics to track product health
8. Content opportunities: 10 topics that would attract ideal customers
9. Partnership/distribution ideas for reaching lower-income families"""
        return await self.run_task("product_research", prompt)

    async def improve_homepage(self, current_content: dict) -> dict:
        context = await self.recall_context("homepage conversion copywriting mission")
        prompt = f"""Rewrite and improve the Aethyro.com homepage to maximize conversions.

Current homepage data:
Title: {current_content.get('title')}
Description: {current_content.get('description')}
H1: {current_content.get('h1', [])}
Current content: {current_content.get('body_text', '')[:800]}

Write a complete homepage content plan:
1. Hero section: headline + subheadline + CTA button text
2. Problem statement section (3 pain points with empathy)
3. Solution section (3 core benefits with specifics)
4. How it works (3-step simple process)
5. Social proof section (testimonial templates, stats placeholders)
6. Pricing/value section copy
7. Final CTA section with urgency
8. Footer value reinforcement

Each section: provide exact copy + HTML structure suggestions."""
        return await self.run_task("homepage_improvement", prompt, current_content.get("url"))
