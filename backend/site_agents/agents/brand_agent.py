"""Brand Architect — identity, voice, messaging, visual direction, positioning."""
from __future__ import annotations
from .base import SiteAgent


class BrandAgent(SiteAgent):
    name = "brand"
    role = "Brand Architect"
    expertise = "Brand identity, visual design direction, messaging strategy, voice and tone, competitive positioning, brand differentiation"
    system_prompt = """You are the Brand Architect for Aethyro — sovereign AI for working families.

Aethyro's brand truth:
• We exist because AI has a class problem — it's built for and marketed to people who don't need it most
• Our customers are families who work hard, watch every dollar, and feel left behind by the AI revolution
• We are NOT trying to be ChatGPT for the masses — we are the LOCAL, PRIVATE, AFFORDABLE AI that runs on their hardware
• Brand personality: Capable. Honest. Scrappy. Warm. Unapologetically mission-driven.
• We are David, and OpenAI/Microsoft/Google are Goliath — we lean into this

What we NEVER do:
- Corporate speak or buzzwords
- Fake "diversity" imagery that doesn't reflect real lower-income families
- Overpromise (AI will solve all your problems)
- Apologize for being small and bootstrapped — we wear it as a badge

What makes Aethyro different from every AI product on the market:
1. LOCAL — runs on your hardware, your data stays with you
2. FIXED COST — no usage limits, no surprise bills
3. MISSION — built for families, not enterprises
4. HUMAN — you're talking to a real founder who lives this mission

Your job: make every brand touchpoint feel like a trusted friend who happens to be brilliant at AI."""

    async def brand_audit(self) -> dict:
        from site_agents.crawler import fetch_page
        context = await self.recall_context("brand identity messaging positioning Aethyro competitors")
        try:
            page = await fetch_page("https://aethyro.com")
            site_context = f"Homepage title: {page.title}\nH1: {page.h1}\nDescription: {page.description}\nContent: {page.body_text[:800]}"
        except Exception:
            site_context = "Homepage: could not fetch"

        prompt = f"""Perform a complete brand audit for Aethyro.

Current site snapshot:
{site_context}

Audit these dimensions (score 0-10 each, with specific evidence):
1. CLARITY — Can a new visitor understand what Aethyro does in 5 seconds?
2. DIFFERENTIATION — Does the brand feel distinct from ChatGPT, Copilot, Claude?
3. MISSION RESONANCE — Does the brand clearly serve lower-income families?
4. VOICE CONSISTENCY — Is the tone consistent across pages?
5. VISUAL COHERENCE — Do design elements (colors, fonts, imagery) feel intentional?
6. TRUST SIGNALS — Does the brand feel credible to a skeptical first-time visitor?
7. EMOTIONAL RESONANCE — Does it make the target audience feel seen?

For each score below 7:
- Specific evidence of the problem
- Exact fix with before/after example

Overall brand health verdict (one sentence) + single most urgent brand fix."""
        return await self.run_task("brand_audit", prompt)

    async def messaging_framework(self) -> dict:
        context = await self.recall_context("messaging value proposition tagline positioning statement")
        prompt = """Build Aethyro's complete messaging framework.

Deliver:
1. CORE MESSAGE — one sentence that captures everything Aethyro is (test: could a 12-year-old understand it?)
2. TAGLINE OPTIONS (5 variants):
   - Functional: emphasizes what it does
   - Emotional: emphasizes how it feels
   - Mission: emphasizes who it's for
   - Provocative: challenges the status quo
   - Simple: most memorable version
   Vote for the strongest with reasoning.

3. ELEVATOR PITCHES by context:
   - 10-second (networking event)
   - 30-second (investor/client intro)
   - 2-minute (demo intro)
   Each tailored to a different audience (family, small business, investor)

4. VALUE PROPOSITIONS by audience:
   - Lower-income families: what's their specific pain + our specific relief
   - Small CPA/accounting firms: their pain + our relief
   - Community organizations: their pain + our relief

5. OBJECTION RESPONSES — exact language for top 5 objections:
   - "I can just use ChatGPT for free"
   - "Is my data safe?"
   - "This seems complicated"
   - "I can't afford it"
   - "How is this different from other AI tools?"

6. MESSAGING DON'TS — 10 phrases/approaches to never use in Aethyro marketing"""
        return await self.run_task("messaging_framework", prompt)

    async def brand_voice_guide(self) -> dict:
        context = await self.recall_context("brand voice tone writing style guide")
        prompt = """Write Aethyro's Brand Voice & Tone Guide.

Sections:
1. VOICE PILLARS (4 core voice attributes):
   For each: name, 1-sentence definition, do example, don't example

2. TONE BY CONTEXT (tone shifts based on situation):
   - Homepage/marketing: [describe tone]
   - Product UI: [describe tone]
   - Customer support: [describe tone]
   - Error messages: [describe tone]
   - Social media: [describe tone]
   - Legal documents: [describe tone]

3. VOCABULARY GUIDE:
   Words we USE (our vocabulary — 20 words that feel like Aethyro)
   Words we AVOID (corporate/tech buzzwords — 20 words banned)

4. SENTENCE STRUCTURE RULES:
   - Sentence length targets
   - Active vs passive voice policy
   - Jargon policy (AI terms: always explain or always avoid?)
   - Reading level target (Flesch-Kincaid grade level)

5. COPY EXAMPLES — rewrite these 3 corporate examples in Aethyro voice:
   - "Leverage our cutting-edge AI platform to optimize your workflow efficiency"
   - "Our solution provides robust, scalable AI capabilities for enterprise and SMB"
   - "Unlock the power of artificial intelligence with our innovative suite of tools"

6. THE AETHYRO SNIFF TEST — 5 questions to ask before publishing any copy"""
        return await self.run_task("brand_voice_guide", prompt)

    async def visual_identity_spec(self) -> dict:
        context = await self.recall_context("visual identity design colors fonts imagery brand")
        prompt = """Write Aethyro's Visual Identity Specification.

Given Aethyro's mission (affordable local AI for lower-income families) and personality (capable, honest, scrappy, warm):

1. COLOR PALETTE:
   - Primary color (with hex) — why this color for our brand/mission
   - Secondary color (with hex)
   - Accent color (with hex)
   - Background + text colors
   - Color psychology rationale for each choice
   - Accessibility check: do these meet WCAG 2.1 AA contrast?

2. TYPOGRAPHY:
   - Heading font — free Google Font recommendation + why
   - Body font — readable at all sizes, accessible
   - Code/monospace font (for AI output display)
   - Font size scale for web

3. IMAGERY DIRECTION:
   - Photography style (real people vs stock vs illustration?)
   - What types of families/people to show (specific, not generic)
   - AI visualization approach (avoid: blue brains, robot arms, circuit boards)
   - What to NEVER show

4. LOGO PRINCIPLES:
   - What the Aethyro logo should communicate
   - Clearspace and sizing rules
   - Dark/light background versions

5. UI COMPONENT FEEL:
   - Button style (rounded, sharp, outlined?)
   - Card style
   - Icon style (line vs filled?)
   - Overall vibe reference (what product do we visually aspire to?)"""
        return await self.run_task("visual_identity_spec", prompt)

    async def positioning_statement(self) -> dict:
        context = await self.recall_context("competitive positioning vs ChatGPT Copilot Claude alternatives")
        prompt = """Write Aethyro's competitive positioning strategy against the AI giants.

Competitors to position against:
- ChatGPT (OpenAI) — free tier + Plus $20/mo
- Microsoft Copilot — bundled with M365
- Google Gemini — free + workspace integration
- Claude (Anthropic) — free + Pro $20/mo
- Local alternatives: Ollama (free but technical), LM Studio (technical)

For each competitor:
1. Their strength (what they genuinely do better)
2. Their weakness relative to Aethyro's market
3. The exact positioning statement to use against them

Then:
AETHYRO'S MOAT — 3 things we can do that none of them can:
1. [Specific capability]
2. [Specific capability]
3. [Specific capability]

OFFICIAL POSITIONING STATEMENT (Geoffrey Moore format):
"For [target customer] who [need/want], Aethyro is a [category] that [key benefit]. Unlike [competitors], Aethyro [key differentiator]."

Write 3 versions of this statement for different contexts.

BATTLE CARD — one-page competitive comparison table (format as markdown table)."""
        return await self.run_task("positioning_statement", prompt)
