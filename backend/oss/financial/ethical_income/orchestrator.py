"""
Ethical Income Orchestrator — unified lawful income assistant system.

Routes creator requests to survey, airdrop, P2E, and staking assistants.
All paths: consent → discover → human confirm → creator executes → track earnings.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import time

from .core import ComplianceGate, RateLimiter, HumanConfirmationGate, EthicalActionResult
from .survey_assistant import SurveyAssistant
from .airdrop_tracker import AirdropTracker
from .p2e_assistant import P2EAssistant
from .staking_optimizer import StakingOptimizer


class EthicalIncomeOrchestrator:
    """
    Scalable ethical income system for the Omni Financial Sector.

    1 agent + 1 consenting user → assistant discovers and ranks opportunities
    N users → N consent records; rate limits prevent platform abuse
    """

    def __init__(
        self,
        user_wallet: Optional[str] = None,
        user_address: Optional[str] = None,
    ):
        self.compliance = ComplianceGate()
        self.rate_limiter = RateLimiter()
        self.confirmation_gate = HumanConfirmationGate()
        self.surveys = SurveyAssistant(self.rate_limiter, self.confirmation_gate)
        self.airdrops = AirdropTracker(user_wallet, self.rate_limiter, self.confirmation_gate)
        self.p2e = P2EAssistant(self.confirmation_gate)
        self.staking = StakingOptimizer(user_address, self.rate_limiter, self.confirmation_gate)
        self._earnings_ledger: List[Dict[str, Any]] = []

    def route(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        """
        Route natural-language request to the correct ethical assistant.
        Rejects prohibited automation at the compliance gate.
        """
        blocked = self.compliance.check(prompt)
        if blocked:
            return {
                "agent_id": agent_id,
                "routed": False,
                "result": blocked.to_dict(),
            }

        p = prompt.lower()
        if any(w in p for w in ("survey", "mturk", "prolific", "microtask", "clickworker", "appen")):
            platform = None
            for key in ("prolific", "mturk", "clickworker", "microworkers", "appen"):
                if key in p:
                    platform = key
                    break
            result = self.surveys.discover_opportunities(platform)
            channel = "survey"

        elif any(w in p for w in ("airdrop", "faucet", "coinmarketcap", "coingecko", "coinbase earn")):
            source = None
            for key in ("coinmarketcap", "coingecko", "binance", "coinbase", "faucetpay"):
                if key in p:
                    source = key
                    break
            result = self.airdrops.fetch_listings(source)
            channel = "airdrop"

        elif any(w in p for w in ("splinterlands", "axie", "p2e", "play to earn", "game", "stepn")):
            if "suggest" in p or "team" in p or "deck" in p:
                platform = "splinterlands"
                if "axie" in p:
                    platform = "axie_infinity"
                result = self.p2e.suggest_team(platform)
            else:
                result = self.p2e.list_games()
            channel = "p2e"

        elif any(w in p for w in ("stake", "staking", "yield", "aave", "compound", "defi", "apy")):
            if "compare" in p or "market" in p:
                asset = "USDC" if "usdc" in p else ("ETH" if "eth" in p else None)
                result = self.staking.compare_markets(asset)
            else:
                result = self.staking.suggest_strategy()
            channel = "staking"

        else:
            result = self.get_dashboard()
            channel = "dashboard"

        return {
            "agent_id": agent_id,
            "routed": True,
            "channel": channel,
            "result": result.to_dict() if isinstance(result, EthicalActionResult) else result,
            "timestamp": time.time(),
        }

    def get_dashboard(self) -> Dict[str, Any]:
        """Unified income dashboard across all ethical channels."""
        survey_sum = self.surveys.earnings_summary()
        airdrop_sum = self.airdrops.earnings_summary()
        p2e_ranked = self.p2e.rank_daily_opportunities()
        staking_preview = self.staking.compare_markets()

        return {
            "mode": "ethical_income_dashboard",
            "compliance": "human_in_loop_only",
            "survey_earnings": survey_sum,
            "airdrop_earnings": airdrop_sum,
            "p2e_opportunities": p2e_ranked[:5],
            "top_yield_markets": staking_preview.opportunities[:5],
            "pending_confirmations": len(self.confirmation_gate._pending),
            "scaling_note": (
                "1 user + consent = $5–20/hr surveys (manual), $10–50/day airdrops (manual), "
                "$5–50/day P2E assist, 2–10% APY on owned capital. "
                "Scale by adding consenting users — never bots or multi-accounts."
            ),
        }

    def onboard_creator(
        self,
        platforms: List[str],
        wallet: Optional[str] = None,
        address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One-shot creator opt-in across platforms."""
        consents = []
        for plat in platforms:
            key = plat.lower()
            if key in ("prolific", "mturk", "clickworker", "microworkers", "appen"):
                consents.append(self.surveys.register_consent(key).to_dict())
            elif key in ("coinmarketcap", "coingecko", "binance", "coinbase", "faucetpay"):
                consents.append(self.airdrops.register_consent(key).to_dict())
            elif key in ("splinterlands", "axie", "axie_infinity", "stepn", "gods_unchained"):
                consents.append(self.p2e.register_consent(key).to_dict())

        if wallet:
            self.airdrops.set_wallet(wallet)
        if address:
            self.staking.set_address(address)

        return {
            "onboarded": True,
            "consents": consents,
            "wallet_set": bool(wallet),
            "staking_address_set": bool(address),
            "next": "Call route(agent_id, prompt) to discover opportunities",
        }

    def record_creator_revenue(self, channel: str, amount_usd: float, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        entry = {
            "channel": channel,
            "amount_usd": round(amount_usd, 2),
            "timestamp": time.time(),
            "metadata": metadata or {},
        }
        self._earnings_ledger.append(entry)
        return entry

    def total_tracked_revenue(self) -> float:
        return round(sum(e["amount_usd"] for e in self._earnings_ledger), 2)