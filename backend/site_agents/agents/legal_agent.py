"""Legal Compliance Officer — ToS, privacy policy, GDPR/CCPA/COPPA, AI disclaimers."""
from __future__ import annotations
import time as _time
from .base import SiteAgent

_YEAR = _time.strftime("%Y")
_COMPANY = "Aethyro"
_DOMAIN = "Aethyro.com"
_EMAIL = "legal@aethyro.com"


class LegalAgent(SiteAgent):
    name = "legal"
    role = "Legal Compliance Officer"
    expertise = "Terms of service, privacy policy, GDPR, CCPA, COPPA, AI law, disclaimers, contracts, IP protection"
    system_prompt = f"""You are the Legal Compliance Officer for {_COMPANY} — a local AI platform serving lower-income families and small businesses.

Your legal documents must be:
• Written in plain English (8th grade reading level max) — our users aren't lawyers
• Airtight on AI-specific risks: accuracy limits, no professional advice, data handling
• GDPR and CCPA compliant — we have users in California and EU
• COPPA-aware — children's education is a core feature, children under 13 require parental consent
• Specific to our local/on-device AI model (no cloud data processing to disclose)
• Current as of {_YEAR}

Always include:
- Effective date and last updated date
- Clear section headers
- Plain-language summaries before legal language where complex
- Contact information: {_EMAIL}
- Governing law: State of [USER'S STATE] (leave as placeholder)"""

    async def generate_terms_of_service(self) -> dict:
        context = await self.recall_context("terms of service subscription AI platform user agreement")
        prompt = f"""Write a complete, legally sound Terms of Service for {_DOMAIN}.

{_COMPANY} provides:
- Fixed-cost local AI assistant services ($500-800/month subscription)
- Children's educational AI features
- Affordable satellite data services
- All AI runs locally on customer hardware — no cloud processing

Required sections:
1. Acceptance of Terms
2. Description of Services (local AI, subscriptions, educational features)
3. User Accounts and Security
4. Subscription Terms, Billing, and Auto-Renewal
5. Acceptable Use Policy (what's prohibited)
6. AI-Specific Terms:
   - AI outputs are not professional advice (legal, medical, financial)
   - Accuracy limitations and no warranty on AI responses
   - User responsibility for AI output review
7. Children's Privacy (COPPA) — parental consent for under-13
8. Intellectual Property — who owns what
9. Limitation of Liability and Indemnification
10. Dispute Resolution and Governing Law
11. Termination and Refunds
12. Changes to Terms
13. Contact Information

Write the full document in plain language. Effective date: {_time.strftime('%B %d, %Y')}."""
        return await self.run_task("terms_of_service", prompt)

    async def generate_privacy_policy(self) -> dict:
        context = await self.recall_context("privacy policy data collection GDPR CCPA user data")
        prompt = f"""Write a complete GDPR and CCPA compliant Privacy Policy for {_DOMAIN}.

Key data facts about {_COMPANY}:
- Local AI: model runs on customer's device, conversations NOT sent to our servers
- We collect: name, email, billing info (via Stripe), usage analytics
- Children's education feature: requires parental consent for under-13 (COPPA)
- We do NOT sell personal data
- We use Stripe for payments (they process card data, not us)
- Users are in the US (CCPA — California) and potentially EU (GDPR)

Required sections:
1. Introduction and Plain-Language Summary ("Here's what this means for you:")
2. What Information We Collect (and what we DON'T collect — emphasize local AI)
3. How We Use Your Information
4. Local AI Privacy Advantage (no conversation data sent to cloud)
5. Children's Privacy (COPPA) — under-13 policy
6. Data Sharing — third parties (Stripe, analytics only)
7. Your Rights:
   - CCPA: California residents' rights
   - GDPR: EU residents' rights (access, deletion, portability)
8. Data Retention and Deletion
9. Security Measures
10. Cookies and Analytics
11. Contact and Data Controller Information

Write the full policy. Last updated: {_time.strftime('%B %d, %Y')}."""
        return await self.run_task("privacy_policy", prompt)

    async def generate_ai_disclaimer(self) -> dict:
        context = await self.recall_context("AI disclaimer accuracy limitations liability artificial intelligence")
        prompt = f"""Write a comprehensive AI disclaimer for {_DOMAIN} to be displayed prominently on the site.

{_COMPANY} provides local AI assistants. The disclaimer must cover:

1. SHORT VERSION (for footer/sidebar — 2-3 sentences max):
   Plain-language summary of AI limitations

2. FULL DISCLAIMER (for /legal/ai-disclaimer page):
   a. AI Output Accuracy — AI can be wrong, hallucinate, or provide outdated information
   b. Not Professional Advice — AI responses are not legal, medical, financial, or therapeutic advice
   c. Educational Content — children's AI features are supplemental, not a replacement for teachers
   d. Local Model Limitations — our AI runs locally; it may have different capabilities than cloud AI
   e. User Responsibility — users must verify important information from qualified professionals
   f. No Guarantee of Results — business/productivity outcomes vary by user
   g. Feedback and Correction — how users can report incorrect AI outputs

3. SPECIFIC DISCLAIMERS for:
   - Legal research assistance (always consult a licensed attorney)
   - Medical/health information (always consult a doctor)
   - Financial advice (consult a licensed financial advisor)
   - Children's educational content (parental supervision recommended)

Write in plain English. Make it honest without being scary."""
        return await self.run_task("ai_disclaimer", prompt)

    async def generate_refund_policy(self) -> dict:
        context = await self.recall_context("refund policy subscription cancellation billing")
        prompt = f"""Write a fair, clear refund and cancellation policy for {_DOMAIN}.

{_COMPANY} subscription tiers: $500/mo Starter, $800/mo Pro, $500 one-time setup fee.

Policy must cover:
1. SUMMARY (plain language, 3 sentences)
2. Subscription Cancellation:
   - How to cancel (self-service or contact support)
   - When cancellation takes effect (end of billing period)
   - What happens to the local AI setup on cancellation
3. Refund Eligibility:
   - Setup fee: refundable within 14 days if service not delivered
   - Monthly subscription: prorated refund if service has critical failures
   - No refund for "changed mind" after 7 days (standard SaaS practice)
4. Exceptions — hardship cases (our mission is serving lower-income families):
   - Financial hardship pause option (pause service for up to 60 days)
   - Dispute resolution before charging back
5. How to Request a Refund (email {_EMAIL}, response within 2 business days)
6. Chargebacks — our process to resolve disputes before escalation

Be fair and human. Remember: our customers have less financial cushion than enterprise buyers."""
        return await self.run_task("refund_policy", prompt)

    async def coppa_compliance_check(self) -> dict:
        context = await self.recall_context("COPPA children privacy parental consent under 13")
        prompt = f"""Perform a COPPA compliance review for {_DOMAIN}'s children's educational AI feature.

{_COMPANY} offers AI-powered children's education features as part of its family AI platform.

Provide:
1. COPPA APPLICABILITY — does our service require full COPPA compliance? (Yes/No with reasoning)
2. REQUIRED ACTIONS — specific steps we must take:
   a. Parental consent mechanism (what's acceptable: email consent, signed form, credit card verification?)
   b. What data we CAN collect from children under 13 (and what we cannot)
   c. Parental access rights — how parents can review/delete child's data
   d. Data retention limits for children's data
3. PRIVACY POLICY ADDITIONS — exact language needed for our Privacy Policy
4. TERMS OF SERVICE ADDITIONS — exact language for our ToS
5. UI/UX REQUIREMENTS — age gate design recommendations
6. COPPA SAFE HARBOR — should we join a COPPA safe harbor program? Which ones?
7. RISK ASSESSMENT — if we don't comply, what are the FTC penalty ranges?
8. IMPLEMENTATION CHECKLIST — ordered steps to achieve full compliance within 30 days

Be specific and actionable. This is a legal requirement, not optional."""
        return await self.run_task("coppa_compliance", prompt)
