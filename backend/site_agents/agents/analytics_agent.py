"""Analytics Intelligence Commander — GA4, conversion funnels, growth metrics, KPIs."""
from __future__ import annotations
from .base import SiteAgent


class AnalyticsAgent(SiteAgent):
    name = "analytics"
    role = "Data Intelligence Commander"
    expertise = "Google Analytics 4, conversion funnels, user behavior, A/B testing, growth metrics, cohort analysis, revenue attribution"
    system_prompt = """You are the Data Intelligence Commander for Aethyro — sovereign AI for working families.

Your analytics rules:
• Every recommendation starts with a specific metric (number, %, time)
• Vanity metrics are worthless — focus only on metrics tied to revenue or mission impact
• Our north star: cost per acquired customer → monthly active users → MRR
• We have zero budget for paid analytics tools — use GA4 free, Search Console free, and custom tracking
• Our audience abandons pages fast — every second of load time costs us conversions

You track these KPIs ruthlessly:
1. Traffic to trial conversion rate (target: 3%+)
2. Trial to paid conversion rate (target: 20%+)
3. Customer acquisition cost by channel
4. Time to first value (from signup to first meaningful AI use)
5. Retention rate at 30/60/90 days
6. Revenue per visitor (MRR / monthly visitors)
7. SEO organic growth rate (month-over-month)

When GA4 data is unavailable, provide the exact setup instructions and measurement framework."""

    async def traffic_report(self) -> dict:
        from site_agents.integrations import ga4_client as ga
        data = ga.get_traffic_summary(days=30)
        top_pages = ga.get_top_pages(10)
        context = await self.recall_context("traffic analytics sessions users pageviews conversion")

        if data.get("available"):
            prompt = f"""Analyze this Aethyro GA4 traffic data and provide actionable intelligence:

30-Day Summary:
- Sessions: {data['sessions']}
- Users: {data['users']}
- Pageviews: {data['pageviews']}
- Bounce Rate: {data['bounce_rate']}
- Avg Session Duration: {data['avg_session_duration_s']}s

Top Pages: {data.get('top_pages', [])}
Top Sources: {data.get('top_sources', [])}

Provide:
1. Traffic health verdict (is {data['sessions']} sessions/30 days good for our stage? Compare to benchmarks)
2. Bounce rate {data['bounce_rate']} — is this a problem? What's causing it?
3. Top source analysis — which channels are worth doubling down on?
4. Page performance — which pages are underperforming vs their potential?
5. 3 specific actions to increase qualified traffic within 30 days
6. Estimated revenue impact if we improve conversion rate by 1%"""
        else:
            prompt = f"""GA4 is not yet configured for Aethyro ({data.get('reason', 'credentials missing')}).

Provide:
1. EXACT GA4 SETUP STEPS — how to connect aethyro.com to GA4 (step by step, no prior GA knowledge assumed)
2. CRITICAL EVENTS TO TRACK — the 10 GA4 events Aethyro must set up day 1:
   - Page views, scroll depth, CTA clicks, form submissions, trial signups, paid conversions
3. GOOGLE SEARCH CONSOLE SETUP — separate from GA4, equally important for SEO
4. CUSTOM DASHBOARD — what the GA4 overview dashboard should show (layout recommendations)
5. BASELINE BENCHMARKS — what traffic numbers to expect for a new local AI SaaS at our stage
6. ATTRIBUTION MODEL — how to track which channels drive paid conversions"""

        result = await self.think(prompt, context)
        task_id = self._mem.log_task(self.name, "traffic_report", prompt, result)
        self.remember("analytics", "traffic_report", {"ga4_data": data, "analysis": result})

        try:
            import economy_bridge as _eco
            _eco.complete_task_for(self.name, "traffic_report: GA4 analysis", 25)
        except Exception:
            pass

        return {"agent": self.name, "task_type": "traffic_report",
                "ga4_data": data, "top_pages": top_pages, "result": result, "task_id": task_id}

    async def funnel_analysis(self) -> dict:
        context = await self.recall_context("conversion funnel visitor subscriber customer drop-off")
        prompt = """Map and analyze the Aethyro conversion funnel.

Funnel stages:
1. AWARENESS → VISIT (organic search, social, referral, direct)
2. VISIT → ENGAGEMENT (>2 pages viewed, >30 seconds on site)
3. ENGAGEMENT → LEAD (email signup, contact form, trial request)
4. LEAD → DEMO/CALL (scheduling or demo page visit)
5. DEMO → PAID (trial to subscription conversion)
6. PAID → RETAINED (active at 30/60/90 days)

For each stage provide:
- Benchmark conversion rate (industry standard for our type of product)
- Aethyro target rate (what we should aim for)
- Top drop-off reasons at this stage
- Specific fix to improve conversion rate by 20%+

Then:
- Identify the single biggest bottleneck in the funnel
- If we fix ONLY that one stage, what's the revenue impact?
- 5 A/B tests to run in order of expected impact

Be specific: name pages, copy changes, UX tweaks — not generic advice."""
        return await self.run_task("funnel_analysis", prompt)

    async def conversion_optimization(self, page_url: str) -> dict:
        from site_agents.crawler import fetch_page
        context = await self.recall_context(f"conversion optimization CRO landing page {page_url}")
        try:
            page = await fetch_page(page_url)
            page_context = f"""Page: {page.url}
Title: {page.title}
H1: {page.h1}
H2: {page.h2[:3]}
Word count: {page.word_count}
Content preview: {page.body_text[:600]}"""
        except Exception:
            page_context = f"Page: {page_url} (could not fetch)"

        prompt = f"""Perform a conversion rate optimization (CRO) audit for this page:

{page_context}

Provide:
1. CRO SCORE (0-100) — current conversion potential with reasoning
2. ABOVE THE FOLD — what does a visitor see in the first 3 seconds? Is it compelling?
3. VALUE PROPOSITION — is it immediately clear what Aethyro does and who it's for?
4. CTA ANALYSIS:
   - How many CTAs on the page? (should be 1 primary)
   - CTA copy grade — rewrite if weak
   - CTA placement — is it where users are ready to convert?
5. TRUST SIGNALS — what's missing? (testimonials, logos, guarantees, pricing transparency)
6. FRICTION POINTS — what makes visitors hesitate or leave?
7. MOBILE EXPERIENCE — likely issues based on page structure
8. TOP 5 CHANGES — ordered by expected conversion lift:
   - Change description + expected % improvement + effort estimate"""
        return await self.run_task("cro_audit", prompt, page_url)

    async def growth_dashboard(self) -> dict:
        from site_agents.integrations import ga4_client as ga, stripe_client as sc
        ga_data = ga.get_traffic_summary(30)
        stripe_data = sc.get_revenue_summary()
        context = await self.recall_context("growth metrics dashboard KPIs business health")
        prompt = f"""Build a unified growth intelligence report for Aethyro combining traffic + revenue data.

Traffic (GA4): {ga_data}
Revenue (Stripe): {stripe_data}

Unified analysis:
1. GROWTH HEALTH SCORE (0-100) — single number verdict for this week
2. TOP METRIC MOVEMENTS — what changed significantly this month?
3. REVENUE PER VISITOR — calculate: MRR / monthly visitors (even estimate if exact data unavailable)
4. CAC BY CHANNEL — estimated customer acquisition cost for each traffic source
5. BIGGEST OPPORTUNITIES — where is the highest leverage right now?
6. BIGGEST RISKS — what metric is trending wrong that needs immediate attention?
7. WEEK'S PRIORITY — ONE action that will move the needle most this week
8. 30-DAY PROJECTION — where do we land if current trends continue?"""
        return await self.run_task("growth_dashboard", prompt)

    async def kpi_setup(self) -> dict:
        context = await self.recall_context("KPIs metrics tracking business performance")
        prompt = """Define the 7 KPIs Aethyro must track religiously.

For each KPI:
1. METRIC NAME — clear, specific name
2. DEFINITION — exact formula or measurement method
3. WHERE TO TRACK — GA4 event, Stripe dashboard, custom SQLite query
4. CURRENT BENCHMARK — industry standard or reasonable starting target
5. TARGET — our 90-day goal for this metric
6. ALERT THRESHOLD — when to trigger a review (e.g., "if bounce rate > 70% for 3 days")
7. OWNER — which agent/person monitors this

The 7 KPIs should cover:
- Acquisition (traffic quality)
- Activation (first value moment)
- Retention (staying power)
- Revenue (MRR growth)
- Referral (word of mouth)
- Mission (families served)
- Operational (system health)

End with: a weekly review ritual (when, what to look at, how to decide next week's priority)."""
        return await self.run_task("kpi_setup", prompt)
