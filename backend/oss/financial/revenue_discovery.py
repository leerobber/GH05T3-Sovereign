"""
Lawful Revenue Discovery — grants, bounties, unclaimed property (own name),
compliant micro-income, and high-signal funding research.

Does NOT: wallet sweeping, survey bots, tracker evasion, or theft vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time
import random

from .lawful_boundaries import (
    PROHIBITED_PATHS,
    LAWFUL_REVENUE_CATEGORIES,
    LAWFUL_BOUNDARIES_TEXT,
)


@dataclass
class RevenueOpportunity:
    opportunity_id: str
    category: str
    title: str
    description: str
    estimated_usd: float
    effort_hours: float
    legality: str  # always "lawful" — prohibited requests never become opportunities
    source_url: str
    requirements: List[str]
    privacy_notes: List[str]
    agent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Curated starting points — agents extend via research tasks (no scraping others' wallets)
GRANT_SOURCES = [
    {
        "name": "Grants.gov",
        "url": "https://www.grants.gov/",
        "categories": ["government_grant"],
        "notes": "US federal grants — verify eligibility",
    },
    {
        "name": "SBIR/STTR",
        "url": "https://www.sbir.gov/",
        "categories": ["government_grant"],
        "notes": "Small business innovation research",
    },
    {
        "name": "Gitcoin Grants",
        "url": "https://grants.gitcoin.co/",
        "categories": ["ecosystem_grant"],
        "notes": "Open-source / public goods crypto funding",
    },
    {
        "name": "Ethereum ESP",
        "url": "https://esp.ethereum.foundation/",
        "categories": ["ecosystem_grant"],
        "notes": "Ethereum ecosystem support",
    },
    {
        "name": "Solana Foundation Grants",
        "url": "https://solana.org/grants",
        "categories": ["ecosystem_grant"],
        "notes": "Solana ecosystem projects",
    },
    {
        "name": "Google Developer Programs",
        "url": "https://developers.google.com/",
        "categories": ["developer_bounty", "ecosystem_grant"],
        "notes": "Cloud credits, startup programs — check current offers",
    },
]

UNCLAIMED_PROPERTY_PORTALS = [
    {"state": "multi", "name": "NAUPA / Missing Money", "url": "https://missingmoney.com/", "notes": "Search YOUR legal name only"},
    {"state": "CA", "name": "California SCOUR", "url": "https://ucpi.sco.ca.gov/", "notes": "Creator's own unclaimed funds"},
    {"state": "US", "name": "IRS Where's My Refund", "url": "https://www.irs.gov/refunds", "notes": "Own tax refunds only"},
]

BOUNTY_PROGRAMS = [
    {"name": "HackerOne", "url": "https://www.hackerone.com/", "type": "bug_bounty"},
    {"name": "Immunefi", "url": "https://immunefi.com/", "type": "crypto_bug_bounty"},
    {"name": "Devpost Hackathons", "url": "https://devpost.com/hackathons", "type": "hackathon"},
    {"name": "DoraHacks", "url": "https://dorahacks.io/", "type": "hackathon"},
]

# Manual micro-income — creator completes; NO bot automation
MICRO_INCOME_MANUAL = [
    {
        "platform": "Prolific",
        "url": "https://www.prolific.com/",
        "type": "research_surveys",
        "rules": "One account, honest answers, manual completion only",
    },
    {
        "platform": "UserTesting",
        "url": "https://www.usertesting.com/",
        "type": "ux_studies",
        "rules": "Manual sessions, real feedback",
    },
    {
        "platform": "Swagbucks",
        "url": "https://www.swagbucks.com/",
        "type": "surveys_rebates",
        "rules": "Manual, read ToS, low $/hour — use only if worth time",
    },
]


class LawfulRevenueDiscovery:
    """
    Agents research lawful funding paths for the creator.
    Rejects prohibited requests at the gate.
    """

    def __init__(self):
        self.opportunities: Dict[str, RevenueOpportunity] = {}
        self.rejected_proposals: List[Dict[str, Any]] = []

    def is_prohibited(self, text: str) -> Optional[str]:
        """Return prohibited path key if request crosses lawful boundary."""
        t = text.lower()
        checks = [
            ("wallet_sweeping", ("sweep wallet", "sweep unclaimed", "unclaimed wallet", "unclaimed crypto wallet", "lost wallet", "drain wallet", "find wallet", "scraping wallet", "locked wallet key")),
            ("seed_phrase_recovery_theft", ("seed phrase", "mnemonic", "brute force wallet", "recover someone")),
            ("survey_bot_automation", ("bot survey", "automate survey", "survey bot", "auto complete survey", "auto complete", "selenium bot", "automate prolific", "automate mturk")),
            ("game_bot_farming", ("bot game", "automate game", "game bot", "auto farm", "game bot farming")),
            ("tracker_evasion_fraud", ("evade track", "stealth fraud", "avoid detection illegal", "anti tracker fraud")),
        ]
        for key, phrases in checks:
            if any(p in t for p in phrases):
                return key
        return None

    def propose_from_agent(
        self,
        agent_id: str,
        prompt: str,
        profit_pulse: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Turn agent research into lawful opportunities.
        Returns rejection if prompt requests prohibited activity.
        """
        prohibited = self.is_prohibited(prompt)
        if prohibited:
            entry = {
                "agent_id": agent_id,
                "prohibited": prohibited,
                "prompt_excerpt": prompt[:120],
                "reason": LAWFUL_BOUNDARIES_TEXT,
                "timestamp": time.time(),
            }
            self.rejected_proposals.append(entry)
            return {
                "accepted": False,
                "prohibited_path": prohibited,
                "message": (
                    f"REJECTED: {prohibited} is prohibited. "
                    "We need income — illegal paths destroy the creator. "
                    "See lawful alternatives in discover_all()."
                ),
                "lawful_alternatives": self.discover_all(agent_id)[:5],
            }

        opportunities = []
        if any(w in prompt.lower() for w in ("grant", "free money", "program", "funding")):
            opportunities.extend(self.discover_grants(agent_id))
        if any(w in prompt.lower() for w in ("unclaimed", "missing money", "forgotten")):
            opportunities.extend(self.discover_unclaimed_property(agent_id))
        if any(w in prompt.lower() for w in ("survey", "game", "micro", "side income")):
            opportunities.extend(self.discover_micro_income_manual(agent_id))
        if any(w in prompt.lower() for w in ("bounty", "hackathon", "bug")):
            opportunities.extend(self.discover_bounties(agent_id))
        if not opportunities:
            opportunities = self.discover_all(agent_id)[:8]

        return {
            "accepted": True,
            "opportunities": [asdict(o) for o in opportunities],
            "profit_pulse": profit_pulse,
            "privacy_hygiene": self.privacy_hygiene_checklist(),
        }

    def discover_grants(self, agent_id: str) -> List[RevenueOpportunity]:
        out = []
        for src in GRANT_SOURCES:
            oid = f"GRANT-{int(time.time())}-{random.randint(100,999)}"
            opp = RevenueOpportunity(
                opportunity_id=oid,
                category=src["categories"][0],
                title=f"Research grant: {src['name']}",
                description=f"Agent {agent_id}: evaluate creator eligibility for {src['name']}. {src['notes']}",
                estimated_usd=random.uniform(500, 25000),
                effort_hours=random.uniform(4, 20),
                legality="lawful",
                source_url=src["url"],
                requirements=["Verify eligibility", "Prepare application", "Real project alignment"],
                privacy_notes=["Use official portal only", "Do not share SSN in agent logs"],
                agent_id=agent_id,
            )
            out.append(opp)
            self.opportunities[oid] = opp
        return out

    def discover_unclaimed_property(self, agent_id: str) -> List[RevenueOpportunity]:
        """Creator's OWN unclaimed property — NOT crypto wallet sweeping."""
        out = []
        for portal in UNCLAIMED_PROPERTY_PORTALS:
            oid = f"UNCL-{int(time.time())}-{random.randint(100,999)}"
            opp = RevenueOpportunity(
                opportunity_id=oid,
                category="unclaimed_property_own",
                title=f"Search own unclaimed property: {portal['name']}",
                description=(
                    f"Creator searches THEIR legal name on {portal['name']}. "
                    "This is lawful unclaimed property recovery — not taking others' crypto wallets."
                ),
                estimated_usd=random.uniform(50, 5000),
                effort_hours=1.0,
                legality="lawful",
                source_url=portal["url"],
                requirements=["Creator legal name", "Identity verification per state rules"],
                privacy_notes=["Search only your own identity", "Official state/federal portals"],
                agent_id=agent_id,
                metadata={"state": portal.get("state")},
            )
            out.append(opp)
            self.opportunities[oid] = opp
        return out

    def discover_bounties(self, agent_id: str) -> List[RevenueOpportunity]:
        out = []
        for prog in BOUNTY_PROGRAMS:
            oid = f"BNTY-{int(time.time())}-{random.randint(100,999)}"
            opp = RevenueOpportunity(
                opportunity_id=oid,
                category="developer_bounty",
                title=f"{prog['name']} — {prog['type']}",
                description=f"Legitimate {prog['type']} with published rules and disclosure.",
                estimated_usd=random.uniform(100, 15000),
                effort_hours=random.uniform(8, 80),
                legality="lawful",
                source_url=prog["url"],
                requirements=["Follow program rules", "Responsible disclosure"],
                privacy_notes=["Use program identity", "Separate security research wallet"],
                agent_id=agent_id,
            )
            out.append(opp)
            self.opportunities[oid] = opp
        return out

    def discover_micro_income_manual(self, agent_id: str) -> List[RevenueOpportunity]:
        """Surveys/games — creator completes manually. Agents research; creator executes."""
        out = []
        for plat in MICRO_INCOME_MANUAL:
            oid = f"MICRO-{int(time.time())}-{random.randint(100,999)}"
            opp = RevenueOpportunity(
                opportunity_id=oid,
                category="micro_income_manual",
                title=f"Manual micro-income: {plat['platform']}",
                description=(
                    f"Creator manually completes tasks on {plat['platform']}. "
                    f"NO bots. Rules: {plat['rules']}. Low $/hr — use for gap income only."
                ),
                estimated_usd=random.uniform(20, 200),
                effort_hours=random.uniform(2, 10),
                legality="lawful",
                source_url=plat["url"],
                requirements=["One account", "Manual completion", "Read ToS"],
                privacy_notes=["Minimize PII shared", "Separate email for gig platforms optional"],
                agent_id=agent_id,
                metadata={"automation": "FORBIDDEN"},
            )
            out.append(opp)
            self.opportunities[oid] = opp
        return out

    def discover_all(self, agent_id: str) -> List[RevenueOpportunity]:
        return (
            self.discover_grants(agent_id)
            + self.discover_unclaimed_property(agent_id)
            + self.discover_bounties(agent_id)
            + self.discover_micro_income_manual(agent_id)
        )

    @staticmethod
    def privacy_hygiene_checklist() -> List[str]:
        """Lawful privacy — not fraud evasion."""
        return [
            "Use official grant/government portals — never credential phishing",
            "Separate hot wallet (business) from personal savings",
            "Never store private keys, SSN, or passwords in agent logs or JSONL exports",
            "VPN optional for public WiFi — not for evading fraud detection",
            "Read platform ToS before micro-income; one honest account per platform",
            "Affiliate links require disclosure where legally required",
        ]

    def get_boundaries_text(self) -> str:
        return LAWFUL_BOUNDARIES_TEXT