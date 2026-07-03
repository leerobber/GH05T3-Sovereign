"""Stripe Revenue Intelligence Agent — MRR, ARR, churn, pricing, growth forecasting."""
from __future__ import annotations
from .base import SiteAgent


class StripeAgent(SiteAgent):
    name = "stripe"
    role = "Revenue Intelligence Officer"
    expertise = "Stripe payments, MRR/ARR, subscription analytics, churn analysis, pricing optimization, revenue forecasting"
    system_prompt = """You are the Revenue Intelligence Officer for Aethyro — a fixed-cost local AI platform for lower-income families.

HARD RULES:
• Every insight must start with a specific dollar amount, percentage, or count. No vague opens.
• Never say "consider" or "it depends" — give a verdict with math behind it.
• Frame every recommendation around Aethyro's mission: maximize impact for underserved families while building a sustainable business.
• Think like a bootstrapped SaaS founder who must grow to $10k MRR before running out of runway.

Your revenue intelligence covers:
- Monthly/Annual Recurring Revenue tracking and trend analysis
- Churn rate analysis and customer lifetime value calculation
- Pricing tier optimization for lower-income target market (price sensitivity is high)
- Subscription growth forecasting with conservative/optimistic/aggressive scenarios
- Revenue leak detection (failed payments, downgrades, trial expirations)
- Competitor pricing benchmarking for local AI services

When Stripe data is unavailable, provide actionable benchmarks and models Aethyro should implement immediately."""

    async def revenue_report(self) -> dict:
        from site_agents.integrations import stripe_client as sc
        data = sc.get_revenue_summary()
        charges = sc.get_recent_charges(10)
        customers = sc.get_customer_count()
        context = await self.recall_context("revenue metrics MRR churn subscription performance")

        if data.get("available"):
            prompt = f"""Analyze this Aethyro revenue data and provide a full executive intelligence report:

MRR: ${data['mrr']}
ARR: ${data['arr']}
Active Subscriptions: {data['active_subscriptions']}
Average Subscription Value: ${data['avg_subscription_value']}/month
Churn Rate: {data['churn_rate_pct']}%
Total Customers: {customers.get('total', 'unknown')}
Recent Charges (last 10): {charges}

Provide:
1. Revenue health verdict with specific numbers
2. Churn interpretation — is {data['churn_rate_pct']}% acceptable for our market?
3. MRR growth needed to reach $10k MRR (current trajectory vs target)
4. Top 3 revenue acceleration moves with projected impact
5. Warning flags if any metrics are in danger zones"""
        else:
            prompt = """Aethyro Stripe integration is not yet configured. Provide:
1. The exact revenue metrics Aethyro must track from day 1 (with formulas)
2. Target MRR milestones: ramen profitable → sustainable → growth stage
3. Pricing model recommendation for our market (lower-income families, $500-800/mo business tiers)
4. Churn rate benchmarks for B2B SaaS at our stage
5. Step-by-step Stripe setup checklist to get revenue tracking live"""

        result = await self.think(prompt, context)
        task_id = self._mem.log_task(self.name, "revenue_report", prompt, result)
        self.remember("revenue", "last_report", {"data": data, "analysis": result})

        try:
            import economy_bridge as _eco
            _eco.complete_task_for(self.name, "revenue_report: MRR/ARR analysis", 30)
        except Exception:
            pass

        return {"agent": self.name, "task_type": "revenue_report", "stripe_data": data,
                "result": result, "task_id": task_id}

    async def pricing_optimization(self) -> dict:
        tiers = []
        try:
            from site_agents.integrations import stripe_client as sc
            tiers = sc.get_pricing_tiers()
        except Exception:
            pass
        context = await self.recall_context("pricing strategy subscription tiers lower income families")
        prompt = f"""Analyze and optimize Aethyro's pricing strategy.

Current pricing tiers from Stripe: {tiers if tiers else 'Not yet configured'}
Known pricing: $500/mo Starter + $800/mo Pro + $500 setup fee (from landing page)

Perform:
1. Price sensitivity analysis for lower-income families and small CPA/accounting firms
2. Is $500/$800 optimal or are we leaving money on table / pricing out our audience?
3. Tier structure redesign if needed — what features justify each tier?
4. Recommended free trial / freemium strategy (if any)
5. Annual vs monthly pricing: what discount drives annual conversions?
6. Upsell sequence: what's the natural upgrade path from Starter → Pro?
7. Competitor prices: what do similar local AI services charge?

Give specific numbers and justified verdicts, not options."""
        return await self.run_task("pricing_optimization", prompt)

    async def churn_analysis(self) -> dict:
        context = await self.recall_context("churn retention customer success cancellations")
        prompt = """Analyze churn patterns for Aethyro and build a retention system.

Provide:
1. Churn rate benchmarks: what's acceptable for fixed-cost local AI B2B SaaS?
2. Top 5 churn triggers specific to our market (lower-income businesses, price sensitivity)
3. Early warning signals to detect at-risk customers before they cancel
4. Retention playbook: exact email/call sequence to save a canceling customer
5. Pricing rescue options: pause plan, downgrade path, hardship discount
6. Success milestones: what customer behaviors predict long-term retention?
7. LTV calculation model: at our price points, what's target LTV?

Be specific to Aethyro's mission of serving families who can't afford enterprise AI."""
        return await self.run_task("churn_analysis", prompt)

    async def growth_forecast(self, months: int = 6) -> dict:
        context = await self.recall_context("revenue growth forecast trajectory projections")
        prompt = f"""Build a {months}-month revenue growth forecast for Aethyro.

Assumptions to model:
- Current stage: early/pre-revenue (adjust if Stripe shows otherwise)
- Target market: CPA firms, local businesses, lower-income families
- Channels: Upwork, Fiverr, LinkedIn, direct outreach
- Pricing: $500/mo Starter, $800/mo Pro

Generate 3 scenarios:
1. CONSERVATIVE: 2 new customers/month, 5% churn
2. BASE CASE: 5 new customers/month, 3% churn
3. AGGRESSIVE: 10 new customers/month, 2% churn

For each scenario show:
- Month-by-month MRR table (months 1-{months})
- Month when ramen profitable ($3k/mo)
- Month when sustainable ($8k/mo)
- Required customer count at each milestone

End with: the single most important lever to pull to reach $10k MRR fastest."""
        return await self.run_task("growth_forecast", prompt)
