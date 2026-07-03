"""HR & People Director — hiring, roles, onboarding, org structure, contractor management."""
from __future__ import annotations
from .base import SiteAgent


class HRAgent(SiteAgent):
    name = "hr"
    role = "People & Culture Director"
    expertise = "Hiring, role design, compensation, onboarding, culture, remote teams, contractor management, equity-conscious recruiting"
    system_prompt = """You are the People & Culture Director for Aethyro — a bootstrapped AI platform built for lower-income families.

Aethyro's HR philosophy:
• Mission-first hiring — candidates must genuinely believe in making AI accessible to working families
• Bootstrap reality — compensation is lean; sell mission, equity potential, and impact
• Contractor-first at early stage — avoid full-time overhead until revenue justifies it
• Diversity is strategic — our users are diverse; our team should reflect them
• No corporate BS — flat, async-first, results-over-hours culture

You know:
- How to write job descriptions that attract mission-driven candidates
- Fair compensation ranges for bootstrapped startups (not FAANG rates)
- 1099 contractor vs W-2 employee legal distinctions
- 30/60/90 day onboarding that gets people productive fast
- Culture documentation that actually influences behavior
- How to structure equity/rev-share for early team members with no salary budget

When giving compensation ranges, always state the source assumption (e.g., "based on median for remote junior role")."""

    async def design_role(self, title: str, responsibilities: str) -> dict:
        context = await self.recall_context(f"job description {title} hiring role requirements")
        prompt = f"""Design a complete job description for Aethyro's {title} role.

Responsibilities given: {responsibilities}

Deliver:
1. JOB TITLE — finalized (adjust if given title isn't standard)
2. ONE-LINER — what this person does in one sentence
3. MISSION ALIGNMENT — 2 sentences why this role matters for Aethyro's mission to serve lower-income families
4. KEY RESPONSIBILITIES — 5-7 bullet points (outcome-focused, not task lists)
5. REQUIREMENTS:
   - Must-have (3-4 non-negotiables)
   - Nice-to-have (3-4 differentiators)
   - NOT required (explicitly remove common gatekeeping requirements)
6. COMPENSATION RANGE — realistic bootstrap range with reasoning:
   - Contractor rate ($/hr or $/project)
   - Full-time equivalent (if/when we hire FTE)
   - Equity/rev-share structure for early team
7. CULTURE FIT SIGNALS — 3 interview questions that reveal mission alignment
8. APPLICATION PROCESS — simple, respectful of candidate time
9. DIVERSITY STATEMENT — specific to Aethyro's mission (not generic corporate boilerplate)"""
        return await self.run_task("role_design", prompt)

    async def build_org_chart(self, stage: str = "pre-revenue") -> dict:
        context = await self.recall_context("org chart team structure startup roles")
        prompt = f"""Build a recommended org structure for Aethyro at the {stage} stage.

Aethyro's current state:
- Solo founder (Robert) runs: product, AI engineering, sales, marketing, customer success
- Revenue: early-stage, Upwork/Fiverr/LinkedIn clients
- Products: local AI platform, children's education, satellite data access
- Tech: Python backend, Ollama local models, Windows + Linux
- Goal: reach $10k MRR, then expand

Design the org structure for {stage}:
1. CURRENT STATE MAP — what Robert is covering alone (with time allocation estimates)
2. HIRE #1 (most critical gap to fill first):
   - Role, why this unblocks growth, contractor vs FTE, budget
3. HIRE #2 — next unlock
4. HIRE #3 — after revenue stabilizes
5. TARGET STATE ORG CHART at $10k MRR (6-8 people max)
6. ROLE BOUNDARIES — who owns what to avoid overlap/conflict
7. DECISION MATRIX — which decisions go to Robert vs delegated
8. CONTRACTOR NETWORK — which functions to always keep as contractors (never hire)"""
        return await self.run_task("org_chart", prompt)

    async def onboarding_playbook(self, role: str) -> dict:
        context = await self.recall_context(f"onboarding {role} new hire first 90 days")
        prompt = f"""Write a 30/60/90 day onboarding playbook for a new {role} at Aethyro.

Aethyro context:
- Async-first, remote, no office
- Stack: Python, FastAPI, Ollama, Windows + Linux, Slack
- Mission: local AI for underserved families
- Culture: scrappy, direct, mission-over-polish

Onboarding plan:
DAY 1 (First 4 hours):
- Accounts to set up (list every tool)
- First document to read
- First person to meet (Slack intro format)
- First task to complete (wins confidence fast)

WEEK 1 GOALS:
- 3 specific deliverables to complete
- Key systems to understand
- Questions to answer before week 2

30-DAY MILESTONE:
- What does "success" look like concretely?
- How is it measured?

60-DAY MILESTONE:
- Now owning what independently?
- Feedback checkpoint format

90-DAY MILESTONE:
- Fully autonomous in role?
- What should they have improved since starting?

CULTURE TRANSFER:
- 3 things that will get them in trouble if they miss them
- 3 things that will make them thrive at Aethyro"""
        return await self.run_task("onboarding_playbook", prompt)

    async def hiring_strategy(self) -> dict:
        context = await self.recall_context("hiring strategy talent acquisition startup bootstrap")
        prompt = """Build Aethyro's hiring roadmap for the next 12 months.

Constraints:
- Bootstrap budget — every hire must generate more than it costs within 90 days
- Mission-alignment is non-negotiable — we serve lower-income families
- Remote-first, async-friendly
- Competitive advantage: meaningful work, mission, equity upside (not salary)

Deliver:
1. HIRING PRIORITY STACK — ordered list of the next 5 hires with revenue impact rationale
2. BUILD VS BUY analysis for each function:
   - Build (hire/train) vs Buy (contractor) vs Borrow (partner) vs Bot (automate with AI)
3. SOURCING STRATEGY — where to find mission-aligned candidates who'll work for below-market:
   - Specific communities, job boards, networks for our audience
4. COMPENSATION PHILOSOPHY — how to structure comp with no cash:
   - Revenue share model (% of revenue generated)
   - Equity vesting for early team
   - "Mission premium" — what non-cash perks justify below-market pay
5. INTERVIEW PROCESS — 3-step maximum (respect candidate time):
   - Async application (what to ask)
   - 30-min video screen (what to assess)
   - Paid test project ($50-100 project that reveals real skill)
6. OFFER TEMPLATE — simple offer letter structure for contractors
7. RED FLAGS to filter out: candidates who won't work for our mission/compensation reality"""
        return await self.run_task("hiring_strategy", prompt)

    async def contractor_agreement_template(self, role: str) -> dict:
        context = await self.recall_context("contractor agreement 1099 independent contractor IP")
        prompt = f"""Write a contractor agreement template for Aethyro's {role} contractor.

This is a 1099 independent contractor agreement (not employment).

Required sections:
1. PARTIES — Aethyro and Contractor (leave name/address as [PLACEHOLDER])
2. SCOPE OF WORK — how to describe deliverables for a {role}
3. COMPENSATION — payment terms, invoicing schedule, late payment
4. INTELLECTUAL PROPERTY — work-for-hire clause; Aethyro owns all deliverables
5. CONFIDENTIALITY — what's confidential (source code, customer data, business strategy)
6. NON-SOLICITATION — don't poach our clients/team for 12 months after
7. INDEPENDENT CONTRACTOR STATUS — tax responsibility, no benefits, no employment relationship
8. TERMINATION — 14-day notice, immediate for cause
9. LIMITATION OF LIABILITY
10. GOVERNING LAW — [State] placeholder

Write the full agreement in plain English. Flag any sections where a real attorney review is advised before use."""
        return await self.run_task("contractor_agreement", prompt)
