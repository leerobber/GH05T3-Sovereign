"""Design Agent — analyzes visual design, UX, and accessibility of Aethyro.com."""
from __future__ import annotations
import logging
from .base import SiteAgent

LOG = logging.getLogger("site_agents.design")


class DesignAgent(SiteAgent):
    name = "design"
    role = "Web Design & UX Specialist"
    expertise = "UI/UX design, visual hierarchy, accessibility (WCAG 2.1), responsive design, conversion rate optimization, brand consistency"
    system_prompt = """You are an elite web designer and UX specialist for Aethyro.com.

Aethyro.com's brand identity:
- Mission: Fixed-cost local AI for lower-income families and children's education
- Tone: Trustworthy, accessible, empowering, not corporate
- Color values: Should convey intelligence + warmth, not cold tech-blue
- Typography: Must be highly readable for diverse literacy levels
- Users: Families who may be tech-unfamiliar; mobile-first priority

Your analysis framework:
1. VISUAL HIERARCHY: Is the most important information prominent? Clear CTA?
2. ACCESSIBILITY: WCAG 2.1 AA compliance — contrast ratios, alt text, keyboard nav
3. MOBILE RESPONSIVENESS: Touch targets, viewport, font sizes
4. CONVERSION OPTIMIZATION: CTA placement, trust signals, friction points
5. BRAND CONSISTENCY: Logo, colors, typography across pages
6. PERFORMANCE SIGNALS: Image optimization, above-fold content
7. TRUST ELEMENTS: Social proof, testimonials, certifications

When analyzing page structure from HTML, infer visual design from:
- CSS classes (Tailwind, Bootstrap tell you layout)
- Image tags and their roles
- Heading hierarchy
- Button/CTA text and placement
- Form elements and labels

Always provide specific HTML/CSS code improvements, not vague advice."""

    async def analyze_design(self, page_data: dict) -> dict:
        html_structure = f"""URL: {page_data.get('url')}
Title: {page_data.get('title')}
H1: {page_data.get('h1', [])}
H2 sections: {page_data.get('h2', [])[:8]}
H3 sections: {page_data.get('h3', [])[:8]}
Images ({len(page_data.get('images', []))}) alt texts: {[img.get('alt','MISSING') for img in page_data.get('images', [])[:10]]}
Word count: {page_data.get('word_count', 0)}
OG image: {page_data.get('og_image', 'none')}
Content preview: {page_data.get('body_text', '')[:800]}"""

        prompt = f"""Analyze the design and UX of this Aethyro.com page from its structural HTML data:

{html_structure}

Provide:
1. UX Score (0-100) with reasoning
2. Visual hierarchy assessment: is the page structure logical and clear?
3. Accessibility issues found (alt text, heading structure, readability)
4. Mobile UX assessment based on content structure
5. CTA (Call-to-Action) effectiveness — what CTAs exist? Are they compelling?
6. Trust signals present/missing
7. Top 5 specific design improvements with example code/copy
8. Brand alignment assessment for lower-income family audience
9. Conversion bottlenecks to fix"""
        return await self.run_task("design_audit", prompt, page_data.get("url"))

    async def generate_design_spec(self, component: str) -> dict:
        prompt = f"""Create a detailed design specification for the Aethyro.com "{component}" component.

Design principles for Aethyro.com:
- Audience: Lower-income families, tech-unfamiliar users, parents with children
- Must work perfectly on mobile (60% of traffic from phones)
- High contrast for readability in various lighting
- Simple, warm, empowering — not intimidating tech aesthetics
- Loading speed critical (users may have limited data plans)

Specify:
1. Color palette with hex codes (meets WCAG AA 4.5:1 contrast)
2. Typography: font family, sizes (rem), line height, weight
3. Spacing system (padding/margin in rem/px)
4. Responsive breakpoints and behavior
5. Interaction states (hover, focus, active, disabled)
6. Complete CSS/Tailwind class implementation
7. HTML structure with semantic tags
8. Accessibility attributes (aria-label, role, tabindex)"""
        return await self.run_task("design_spec", prompt)

    async def accessibility_audit(self, pages: list[dict]) -> dict:
        summary = "\n".join([
            f"- {p.get('url')}: {len([i for i in p.get('images',[]) if not i.get('alt')])} missing alt, H1 count: {len(p.get('h1',[]))}"
            for p in pages[:10]
        ])
        prompt = f"""Conduct a WCAG 2.1 accessibility audit across these Aethyro.com pages:

{summary}

For each issue category (Perceivable, Operable, Understandable, Robust):
1. List specific violations found
2. Provide the exact HTML fix
3. Assign severity: Critical/Major/Minor
4. Calculate estimated % of users impacted

Then provide:
- Overall accessibility score
- Priority fix order
- Quick wins (fixes under 1 hour each)
- Complex fixes (need dev time)"""
        return await self.run_task("accessibility_audit", prompt)
