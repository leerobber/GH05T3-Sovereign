"""Marketing Agent — strategy, analytics, campaigns for Aethyro.com."""
from __future__ import annotations
from .base import SiteAgent


class MarketingAgent(SiteAgent):
    name = "marketing"
    role = "Growth Marketing Strategist"
    expertise = "Digital marketing, Google Analytics, performance tracking, email marketing, social media strategy, paid ads, conversion optimization, community building"
    system_prompt = """You are a growth marketing strategist for Aethyro.com.

Mission: Bring fixed-cost local AI to lower-income families who need it most.
Marketing constraint: Budget is tight — prioritize zero/low-cost channels.
Target audiences:
1. Parents in households earning under $60K/year who want AI tools for their kids
2. Small community organizations, churches, libraries serving lower-income families
3. Teachers in underfunded school districts
4. Families in rural areas with limited internet (satellite data angle)

Marketing principles:
- Every dollar spent must generate measurable return
- Community-first approach over paid advertising
- Build trust through education and transparency
- Partner with existing trusted organizations (nonprofits, schools, churches)
- Word-of-mouth from satisfied families is the most valuable channel

Google Analytics focus areas:
- Core Web Vitals impact on rankings
- User flow analysis — where do people drop off?
- Conversion funnel optimization
- Audience segmentation by demographics
- Goal tracking: signups, feature usage, referrals

Always provide specific, measurable campaign ideas with ROI estimates."""

    async def analytics_strategy(self, site_data: list[dict]) -> dict:
        page_summary = "\n".join([
            f"- {p.get('url')}: {p.get('word_count', 0)} words, {len(p.get('links', []))} internal links"
            for p in site_data[:8]
        ])
        prompt = f"""Create a comprehensive Google Analytics strategy for Aethyro.com.

Current site structure:
{page_summary}

Deliver:
1. GA4 Goals to track (minimum 5 with specific event names and triggers)
2. Custom dimensions to capture (user type, referral source, device)
3. Conversion funnel definition: awareness → interest → signup → activation → retention
4. Key metrics dashboard: which 10 metrics matter most weekly
5. Audience segments to create in GA4
6. A/B tests to run (3 specific tests with hypothesis and success metric)
7. Monthly reporting template with red flags to watch
8. UTM parameter strategy for all campaigns
9. Google Search Console integration priorities"""
        return await self.run_task("analytics_strategy", prompt)

    async def growth_campaign(self, channel: str, budget: str = "$0/month") -> dict:
        prompt = f"""Design a growth campaign for Aethyro.com via {channel} with budget: {budget}

Campaign brief:
- Goal: Acquire lower-income families who would benefit from fixed-cost AI
- Brand voice: Trustworthy, warm, empowering, plain language
- Unique angle: Fixed cost (no surprise bills), local AI (no data sold), family-focused

Deliver:
1. Campaign strategy (30/60/90 day plan)
2. Target audience definition with demographics
3. 5 specific content pieces/ads with exact copy
4. Channel-specific tactics and posting schedule
5. Success metrics and KPIs
6. Budget allocation (even if $0, identify time investment)
7. Partnership opportunities to amplify reach
8. Risk/failure modes and mitigation
9. Expected results: signups, traffic, brand awareness in 90 days"""
        return await self.run_task("growth_campaign", prompt)

    async def email_strategy(self) -> dict:
        context = await self.recall_context("email marketing conversion retention")
        prompt = f"""Build a complete email marketing strategy for Aethyro.com.

Audience: Families who signed up for or are considering Aethyro.com's AI platform
Goal: Activate new users, retain existing ones, drive referrals

Create:
1. Welcome sequence (5 emails, exact subject lines + body outlines)
   - Email 1 (immediate): Welcome + quick win
   - Email 2 (day 3): Feature highlight + use case for kids
   - Email 3 (day 7): Success story + social proof
   - Email 4 (day 14): Advanced feature + community
   - Email 5 (day 30): Referral ask + loyalty reward

2. Re-engagement sequence (3 emails for inactive users)
3. Monthly newsletter template structure
4. Segmentation strategy (new/active/churned/referring)
5. Subject line formulas that work for this audience
6. Best send times for working-class families
7. Unsubscribe reduction tactics"""
        return await self.run_task("email_strategy", prompt)

    async def competitive_analysis(self, competitors: list[str] | None = None) -> dict:
        comp_list = competitors or ["ChatGPT", "Google Gemini", "Microsoft Copilot", "Claude", "Perplexity"]
        prompt = f"""Competitive analysis for Aethyro.com vs: {', '.join(comp_list)}

Focus on the lower-income family market segment that these competitors ignore.

For each competitor:
1. Pricing model and why it's inaccessible to our audience
2. What they do well (be honest)
3. Their blind spots for lower-income users
4. Aethyro's specific advantage against them

Overall:
5. Market positioning map (place all competitors + Aethyro)
6. Our 3 unassailable differentiators
7. 5 marketing messages that exploit competitor weaknesses
8. Partnership opportunities that competitors can't pursue
9. Price anchoring strategy: make Aethyro feel like the obvious choice"""
        return await self.run_task("competitive_analysis", prompt)
