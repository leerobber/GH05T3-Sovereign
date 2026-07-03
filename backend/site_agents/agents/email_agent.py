"""Email Marketing Commander — campaigns, sequences, newsletters, deliverability."""
from __future__ import annotations
from .base import SiteAgent


class EmailAgent(SiteAgent):
    name = "email"
    role = "Email Marketing Commander"
    expertise = "Email campaigns, drip sequences, newsletters, A/B testing, list building, deliverability, copywriting"
    system_prompt = """You are the Email Marketing Commander for Aethyro — sovereign AI for working families.

Your emails must:
• Open rate target: 35%+ (industry avg is 21% — we aim higher)
• Click rate target: 5%+
• Sound like a real person who cares, not a corporation
• Speak directly to working families' real pain: bills, time, feeling left behind by AI
• Never use corporate buzzwords ("synergy", "leverage", "robust", "scalable")
• Always have ONE clear CTA per email — never two

You write for these audiences:
- Lower-income families who want AI but feel it's for rich people
- Small business owners (CPA firms, local shops) on tight budgets
- People who've been burned by expensive tech subscriptions

Tone: Warm, direct, honest, a little scrappy. Like a smart friend who happens to know AI.

You know email deliverability cold: SPF, DKIM, DMARC, list hygiene, warm-up sequences,
spam trigger words to avoid, mobile-first design requirements."""

    async def write_campaign(self, goal: str, audience: str = "lower-income families", tone: str = "warm_direct") -> dict:
        context = await self.recall_context(f"email campaign {goal} {audience} Aethyro")
        prompt = f"""Write a complete email campaign for Aethyro.

Goal: {goal}
Audience: {audience}
Tone: {tone}

Deliver:
1. SUBJECT LINE (primary) — under 50 chars, curiosity-driven, no spam words
2. SUBJECT LINE (A/B variant) — different angle, test against primary
3. PREVIEW TEXT — 90 chars that complement the subject
4. EMAIL BODY — full copy, mobile-friendly paragraphs (max 3 sentences each):
   - Opening hook (1 sentence — specific pain point or number)
   - Problem section (2-3 sentences — make them feel seen)
   - Solution bridge (2-3 sentences — introduce Aethyro without overselling)
   - Social proof placeholder (one line — [testimonial/metric])
   - CTA section — single clear action with urgency or value anchoring
5. POSTSCRIPT — one-line P.S. that re-states the CTA differently
6. DELIVERABILITY NOTES — any spam trigger words avoided, send time recommendation"""
        return await self.run_task("campaign", prompt)

    async def send_welcome(self, to_email: str, name: str = "") -> dict:
        """Send a real welcome email via Resend when someone signs up."""
        from site_agents.executor import send_email
        import asyncio

        greeting = f"Hey {name}," if name else "Hey,"
        html = f"""<div style="font-family:'Segoe UI',sans-serif;max-width:580px;margin:0 auto;background:#030303;color:#f0f0f0;padding:2rem;border-radius:12px">
  <div style="font-size:1.2rem;font-weight:800;color:#fff;margin-bottom:1.5rem"><span style="color:#ff4d00">●</span> Aethyro</div>
  <p style="font-size:1rem;line-height:1.7;color:#aaa">{greeting}</p>
  <p style="font-size:1rem;line-height:1.7;color:#aaa">Your AI is now running on your hardware — not on someone else's server.</p>
  <p style="font-size:1rem;line-height:1.7;color:#aaa">That means your documents, your clients, your data — stays yours. Zero bytes sent to us or anyone else.</p>
  <p style="font-size:1rem;line-height:1.7;color:#aaa">Here's what to do next:</p>
  <ul style="color:#aaa;line-height:2;padding-left:1.5rem">
    <li>Open your <a href="https://aethyro.com/app/dashboard.html" style="color:#ff4d00">dashboard</a> — your agents are waiting</li>
    <li>Check the <a href="https://aethyro.com/marketplace/" style="color:#ff4d00">marketplace</a> for your industry (law, medical, CPA, real estate)</li>
    <li>Reply to this email if you need anything — I read every reply</li>
  </ul>
  <p style="font-size:1rem;line-height:1.7;color:#aaa">Your 14-day trial gives you full access to everything. No limits.</p>
  <a href="https://aethyro.com/app/dashboard.html" style="display:inline-block;margin-top:1rem;padding:.75rem 1.5rem;background:#ff4d00;color:#fff;font-weight:700;border-radius:8px;text-decoration:none">Go to your dashboard →</a>
  <p style="margin-top:2rem;font-size:.8rem;color:#555">— Robert, Aethyro<br/>leer4030@gmail.com · <a href="https://aethyro.com" style="color:#555">aethyro.com</a></p>
</div>"""
        result = await asyncio.to_thread(
            send_email,
            to_email,
            "Your Aethyro AI is ready — here's what to do first",
            html,
        )
        task_id = self._mem.log_task(self.name, "welcome_email", to_email, str(result))
        return {**result, "task_id": task_id, "to": to_email}

    # ── Onboarding sequence ─────────────────────────────────────────────────────

    _STYLE = (
        "font-family:'Segoe UI',sans-serif;max-width:580px;margin:0 auto;"
        "background:#030303;color:#f0f0f0;padding:2rem;border-radius:12px"
    )
    _LOGO = '<div style="font-size:1.2rem;font-weight:800;color:#fff;margin-bottom:1.5rem"><span style="color:#ff4d00">●</span> Aethyro</div>'
    _P = 'style="font-size:1rem;line-height:1.7;color:#aaa"'
    _CTA = 'style="display:inline-block;margin-top:1rem;padding:.75rem 1.5rem;background:#ff4d00;color:#fff;font-weight:700;border-radius:8px;text-decoration:none"'
    _SIG = '<p style="margin-top:2rem;font-size:.8rem;color:#555">— Robert, Aethyro<br/>leer4030@gmail.com · <a href="https://aethyro.com" style="color:#555">aethyro.com</a></p>'

    def _wrap(self, body: str) -> str:
        return f'<div style="{self._STYLE}">{self._LOGO}{body}{self._SIG}</div>'

    def _build_email(self, email_num: int, name: str) -> tuple[str, str]:
        g = f"Hey {name}," if name else "Hey,"
        p, cta, w = self._P, self._CTA, self._wrap
        dash = "https://aethyro.com/app/dashboard.html"

        if email_num == 1:
            subj = "Your AI is ready. Here's what to do first."
            html = w(f"""<p {p}>{g}</p>
<p {p}>Welcome to Aethyro. Most AI tools rent you access to someone else's computer and bill you every time you use it. You just signed up for something that runs on <em>your</em> hardware instead.</p>
<p {p}>Your 14-day trial starts now. No credit card charged yet.</p>
<p {p}><strong>Your first step:</strong> Open your dashboard and connect your first agent. It takes about 4 minutes.</p>
<a href="{dash}" {cta}>Go to your dashboard →</a>
<p {p} style="margin-top:1.5rem">If you hit any issues, reply to this email. I read every one.</p>""")

        elif email_num == 2:
            subj = "What's running on your computer right now"
            html = w(f"""<p {p}>{g}</p>
<p {p}>Yesterday you signed up. Today I want to show you what you actually have.</p>
<p {p}>Inside Aethyro, there are 6 specialist agents running locally:</p>
<ul style="color:#aaa;line-height:2;padding-left:1.5rem">
  <li><strong>Avery</strong> — General assistant. Ask it anything.</li>
  <li><strong>ORACLE</strong> — Business intelligence. Turns your data into decisions.</li>
  <li><strong>FORGE</strong> — Risk analysis. Finds what's missing before it costs you.</li>
  <li><strong>CODEX</strong> — Technical work. Code review, debugging, architecture.</li>
  <li><strong>SENTINEL</strong> — Security. Monitors your setup and flags threats.</li>
  <li><strong>NEXUS</strong> — Strategy. Synthesizes everything into a 48-hour action plan.</li>
</ul>
<p {p}>None of them call home. Your data stays on your machine.</p>
<p {p}><strong>Try this today:</strong> Open the console and ask Avery: <em>"What are the 3 biggest risks in my current business?"</em></p>
<a href="{dash}" {cta}>Open your console →</a>""")

        elif email_num == 3:
            subj = "Why \"$0 per query\" changes everything"
            html = w(f"""<p {p}>{g}</p>
<p {p}>If you've used ChatGPT or Claude for work, you've paid per query — even if it's buried in a subscription.</p>
<p {p}>At 50 queries/day × $0.02/query = $1/day = $365/year. For <em>one person.</em></p>
<p {p}>Aethyro runs locally. After setup, each query costs $0.00. Zero. You own the compute.</p>
<p {p}>For a 5-person team: that's potentially $1,800/year back in your pocket. At our Personal plan ($29/month), you're ahead by over $1,400.</p>
<p {p}><strong>This week's challenge:</strong> Run 10 real tasks through Aethyro that you'd normally pay for. Track the time saved.</p>
<a href="{dash}" {cta}>Your dashboard →</a>""")

        elif email_num == 4:
            subj = "7 days in — are you getting value?"
            html = w(f"""<p {p}>{g}</p>
<p {p}>You're halfway through your trial. Honest question: are you getting value?</p>
<p {p}>If yes — great. Upgrade before your trial ends to keep everything running.</p>
<p {p}>If no — I want to know why. Reply to this email. Seriously. Every piece of feedback shapes what we build next.</p>
<p {p}><strong>Most common setup issues:</strong></p>
<ul style="color:#aaa;line-height:2;padding-left:1.5rem">
  <li>Ollama isn't running → run <code>ollama serve</code> in a terminal</li>
  <li>No model downloaded → run <code>ollama pull llama3</code> first</li>
  <li>Console shows offline → restart the app</li>
</ul>
<p {p}>Plans start at $29/month. Less than a dinner out. Cancel anytime.</p>
<a href="{dash}" {cta}>See plans and upgrade →</a>""")

        else:  # email_num == 5
            subj = "Your trial ends in 2 days"
            html = w(f"""<p {p}>{g}</p>
<p {p}>Your 14-day free trial ends in 2 days.</p>
<p {p}>After that, access pauses. Everything you've set up — your agents, your workflows, your local model — stays on your machine, but the platform goes dark until you subscribe.</p>
<table style="width:100%;border-collapse:collapse;margin:1.5rem 0;color:#aaa;font-size:.9rem">
  <tr style="border-bottom:1px solid #222"><th style="text-align:left;padding:.5rem">Plan</th><th style="text-align:right;padding:.5rem">Price</th><th style="text-align:left;padding:.5rem">Best for</th></tr>
  <tr><td style="padding:.5rem">Personal</td><td style="text-align:right;padding:.5rem">$29/mo</td><td style="padding:.5rem">Individuals, freelancers</td></tr>
  <tr><td style="padding:.5rem">Research</td><td style="text-align:right;padding:.5rem">$199/mo</td><td style="padding:.5rem">Analysts, academics</td></tr>
  <tr><td style="padding:.5rem">Developer</td><td style="text-align:right;padding:.5rem">$299/mo</td><td style="padding:.5rem">Builders, engineering teams</td></tr>
  <tr><td style="padding:.5rem">Professional</td><td style="text-align:right;padding:.5rem">$499/mo</td><td style="padding:.5rem">CPA firms, legal, consulting</td></tr>
</table>
<a href="{dash}" {cta}>Pick a plan and keep going →</a>""")

        return subj, html

    async def schedule_onboarding(self, user_email: str, first_name: str = "") -> dict:
        """Send Email 1 immediately and queue Emails 2-5 in SQLite for the scheduler."""
        import asyncio
        import time as _t
        from site_agents.executor import send_email
        from site_agents.memory_layer import schedule_email

        DAY = 86400
        schedule = {2: DAY, 3: 3 * DAY, 4: 7 * DAY, 5: 12 * DAY}

        subj1, html1 = self._build_email(1, first_name)
        result1 = await asyncio.to_thread(send_email, user_email, subj1, html1)

        now = _t.time()
        scheduled = []
        for num, delay in schedule.items():
            eid = schedule_email(user_email, first_name, num, now + delay)
            scheduled.append({"email_num": num, "id": eid, "send_after_days": delay // DAY})

        self._mem.log_task(self.name, "schedule_onboarding", user_email, str({"email1": result1, "queued": len(scheduled)}))
        return {"email1_sent": result1, "queued": scheduled, "total": 5}

    async def send_followup(self, user_email: str, first_name: str, email_num: int) -> dict:
        """Send a specific follow-up email in the onboarding sequence."""
        import asyncio
        from site_agents.executor import send_email
        subj, html = self._build_email(email_num, first_name)
        return await asyncio.to_thread(send_email, user_email, subj, html)

    async def write_sequence(self, trigger: str, steps: int = 5) -> dict:
        context = await self.recall_context(f"email sequence drip {trigger} onboarding conversion")
        prompt = f"""Design a {steps}-email drip sequence for Aethyro triggered by: {trigger}

For each email provide:
- Send timing (e.g., "Day 0 - immediately", "Day 3")
- Subject line
- Preview text
- Full email body (conversational, ~150-200 words)
- CTA

Sequence strategy:
Email 1: Welcome/Orient — set expectations, build trust, deliver immediate value
Email 2: Education — explain what makes Aethyro different from ChatGPT or Copilot
Email 3: Social proof/story — real use case (family saving time or money)
Email 4: Objection handling — address price, complexity, "is this for me?" fears
Email 5: Convert — strongest CTA, time sensitivity, clear next step
(Add emails 6+ if steps > 5, following: value → convert → retention pattern)

Write every email in full — no placeholders except for [name] personalization."""
        return await self.run_task("sequence", prompt)

    async def write_newsletter(self, highlights: list[str] | None = None) -> dict:
        import time as _time
        context = await self.recall_context("newsletter content update Aethyro community")
        items = "\n".join(f"- {h}" for h in (highlights or [])) or "- Recent Aethyro updates and AI news for families"
        prompt = f"""Write the Aethyro weekly newsletter for {_time.strftime('%B %d, %Y')}.

Highlights to include:
{items}

Newsletter structure:
1. SUBJECT LINE — feels like a friend updating you, not a brand
2. OPENING — 2 sentences: one relatable hook, one bridge to Aethyro
3. THIS WEEK AT AETHYRO — bullet updates, each under 2 sentences
4. AI NEWS THAT MATTERS TO FAMILIES — 2-3 curated items with 1-sentence "why it matters for you"
5. TIP OF THE WEEK — one specific AI workflow tip families can use today (free, no signup required)
6. COMMUNITY CORNER — placeholder for member spotlight or question
7. CLOSING — warm, personal, sign-off from Robert / Aethyro team

Total length: 350-450 words. No corporate fluff."""
        return await self.run_task("newsletter", prompt)

    async def list_building_strategy(self) -> dict:
        context = await self.recall_context("email list building growth subscribers Aethyro")
        prompt = """Build an email list growth strategy for Aethyro targeting lower-income families and small businesses.

Provide:
1. LEAD MAGNETS (3 options ranked by conversion potential):
   - What free resource would our audience trade an email for?
   - Format, title, and 3-sentence description for each
2. ACQUISITION CHANNELS (prioritized by cost/effort):
   - Organic tactics (Reddit, Facebook groups, Nextdoor for families)
   - Content-driven (SEO blog posts with email gates)
   - Partnership plays (community orgs, churches, schools, workforce development)
   - Marketplace cross-promotion (Fiverr/Upwork clients → email)
3. OPT-IN COPY — write the exact form headline, subheadline, and button text for:
   - Homepage pop-up
   - Blog post inline opt-in
   - Exit intent overlay
4. WELCOME SEQUENCE BRIEF — 3-sentence description of the first email's goal
5. LIST HYGIENE — rules for cleaning list (when to remove cold subscribers)
6. 30-DAY GOAL: realistic subscriber target with tactics to hit it"""
        return await self.run_task("list_building", prompt)

    async def deliverability_audit(self) -> dict:
        context = await self.recall_context("email deliverability SPF DKIM spam inbox")
        prompt = """Perform a deliverability setup audit for Aethyro's email system.

Audit checklist and implementation guide:
1. DNS RECORDS — exact SPF, DKIM, DMARC records Aethyro.com needs (format as copy-paste DNS values)
2. SENDING DOMAIN — should we send from @aethyro.com or a subdomain like @mail.aethyro.com? Why?
3. IP WARM-UP PLAN — if starting fresh, day-by-day volume ramp for first 30 days
4. SPAM TRIGGER WORDS — top 20 to avoid in subject lines for our audience/topics
5. ENGAGEMENT STRATEGY — how to maintain list health and avoid spam folder over time
6. EMAIL CLIENT RENDERING — mobile-first rules, image-to-text ratio, preview text limits
7. UNSUBSCRIBE COMPLIANCE — CAN-SPAM and GDPR requirements for our dual US/international audience
8. TOOL RECOMMENDATION — best free/cheap email platform for Aethyro at <1000 subscribers (vs SendGrid vs Mailchimp vs Resend)"""
        return await self.run_task("deliverability_audit", prompt)
