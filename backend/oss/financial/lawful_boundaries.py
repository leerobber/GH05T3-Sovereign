"""
Lawful boundaries for Omni revenue discovery.

Agents MUST NOT propose or implement paths in PROHIBITED_PATHS.
Violation attempts are logged and rejected.
"""

PROHIBITED_PATHS = (
    "wallet_sweeping",           # accessing crypto wallets you do not own
    "seed_phrase_recovery_theft",  # brute-forcing or taking others' keys
    "survey_bot_automation",     # violating platform ToS with bots
    "game_bot_farming",          # automated play for payouts
    "tracker_evasion_fraud",     # hiding from fraud/legal detection
    "identity_theft",            # using others' identity for grants
    "money_laundering",
    "unauthorized_account_access",
)

LAWFUL_REVENUE_CATEGORIES = (
    "government_grant",          # federal/state small business, SBIR, etc.
    "ecosystem_grant",           # Ethereum, Solana, Gitcoin, protocol grants
    "developer_bounty",          # bug bounties, hackathons with published rules
    "unclaimed_property_own",    # creator's OWN name on state unclaimed property DBs
    "rebate_cashback",           # legitimate cashback within merchant ToS
    "freelance_gig",             # Upwork, Fiverr, billable agent outputs
    "saas_product",              # sell what you build
    "defi_yield_own_capital",    # yield on capital YOU own
    "affiliate_compliant",       # affiliate links with disclosure
    "micro_income_manual",       # surveys/games — creator completes manually, one account, ToS
    "scholarship_stipend",       # education/research funding
    "tax_credit_research",       # EITC, R&D credits — creator qualifies
)

LAWFUL_BOUNDARIES_TEXT = """
LAWFUL BOUNDARIES (HARD LIMIT)
------------------------------
We need income desperately — but illegal income destroys the creator faster than poverty.

NEVER propose or build:
  • Scraping/taking funds from wallets you do not own (lost, locked, unclaimed crypto)
  • Bots to farm surveys, games, or airdrops (ToS fraud)
  • Stealth systems to evade fraud trackers or law enforcement
  • "Edge of legal" exploits — if it's theft, it's off limits

ALWAYS prefer:
  • Grants and programs you legitimately qualify for
  • Your OWN unclaimed property (state databases, search your legal name)
  • Bug bounties, hackathons, ecosystem grants with public rules
  • Products and services you sell (SaaS, freelance, research)
  • DeFi yield on capital the creator already owns
  • Manual micro-income (surveys/games) — creator does them honestly, one account, read ToS

Privacy hygiene (lawful): separate wallets for business, minimize PII in exports,
use official grant portals — NOT evasion of legitimate fraud prevention.
""".strip()