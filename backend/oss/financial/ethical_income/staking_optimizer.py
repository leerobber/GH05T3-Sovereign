"""
Staking & yield optimizer — read-only analysis on user-owned capital.

DOES NOT: store private keys, auto-execute transactions, or wash trade.
Creator signs all on-chain actions manually.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time

from .core import HumanConfirmationGate, EthicalActionResult, RateLimiter


@dataclass
class YieldMarket:
    protocol: str
    asset: str
    deposit_apy: float
    borrow_apy: float
    risk_score: float  # 0=low, 1=high
    chain: str
    url: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StakingStrategy:
    asset: str
    balance: float
    deposit_rate: float
    suggested_action: str  # deposit, hold, rebalance
    expected_annual_usd: float
    risks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Representative APY bands — replace with live oracle/API when creator provides RPC (read-only).
YIELD_MARKETS: List[YieldMarket] = [
    YieldMarket("Aave", "USDC", 0.035, 0.055, 0.25, "ethereum", "https://app.aave.com/", ["Battle-tested lending"]),
    YieldMarket("Aave", "DAI", 0.040, 0.060, 0.25, "ethereum", "https://app.aave.com/"),
    YieldMarket("Aave", "ETH", 0.020, 0.040, 0.30, "ethereum", "https://app.aave.com/"),
    YieldMarket("Compound", "USDC", 0.038, 0.058, 0.28, "ethereum", "https://app.compound.finance/"),
    YieldMarket("Compound", "DAI", 0.042, 0.062, 0.28, "ethereum", "https://app.compound.finance/"),
    YieldMarket("Uniswap", "ETH/USDC LP", 0.12, 0.0, 0.55, "ethereum", "https://app.uniswap.org/", ["Impermanent loss risk"]),
    YieldMarket("PancakeSwap", "BNB/USDT LP", 0.25, 0.0, 0.60, "bsc", "https://pancakeswap.finance/", ["IL + smart contract risk"]),
    YieldMarket("Binance Staking", "ETH", 0.04, 0.0, 0.20, "cex", "https://www.binance.com/en/staking", ["Custodial — platform risk"]),
]


class StakingOptimizer:
    """
    Compare yields and suggest strategies on capital the creator already owns.
    Transactions are intents only — creator signs via wallet.
    """

    DEPOSIT_THRESHOLD_APY = 0.03

    def __init__(
        self,
        user_address: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        confirmation_gate: Optional[HumanConfirmationGate] = None,
    ):
        self.user_address = user_address
        self.rate_limiter = rate_limiter or RateLimiter()
        self.confirmation_gate = confirmation_gate or HumanConfirmationGate()
        self._balances: Dict[str, float] = {}

    def set_address(self, address: str) -> None:
        if not address.startswith("0x"):
            raise ValueError("Provide user-owned address (0x...) — never private keys")
        self.user_address = address

    def set_balances(self, balances: Dict[str, float]) -> None:
        """Creator-provided balances (read-only input — not scraped from others' wallets)."""
        self._balances = {k.upper(): float(v) for k, v in balances.items()}

    def compare_markets(self, asset: Optional[str] = None) -> EthicalActionResult:
        if not self.rate_limiter.allow("aave"):
            return EthicalActionResult(
                accepted=False,
                mode="discover_only",
                platform="defi",
                message="Rate limit — wait before next yield query.",
            )

        markets = YIELD_MARKETS
        if asset:
            markets = [m for m in markets if m.asset.upper().startswith(asset.upper())]

        ranked = sorted(markets, key=lambda m: m.deposit_apy, reverse=True)
        return EthicalActionResult(
            accepted=True,
            mode="discover_only",
            platform="defi",
            message=f"Compared {len(ranked)} yield markets (read-only).",
            opportunities=[m.to_dict() for m in ranked],
            next_steps=[
                "Verify APY on official protocol UI before depositing",
                "Understand smart contract and impermanent loss risks",
                "Use suggest_strategy() then prepare_tx_intent() — you sign txs",
            ],
            requires_human_confirmation=False,
            metadata={"private_keys": "NEVER_STORED"},
        )

    def suggest_strategy(self) -> EthicalActionResult:
        if not self._balances:
            return EthicalActionResult(
                accepted=False,
                mode="suggest_strategy",
                platform="defi",
                message="Set balances via set_balances() with YOUR capital amounts.",
                next_steps=["set_balances({'USDC': 1000, 'ETH': 0.5})"],
            )

        strategies: List[StakingStrategy] = []
        for token, balance in self._balances.items():
            market = next((m for m in YIELD_MARKETS if m.asset.upper().startswith(token)), None)
            if not market:
                strategies.append(StakingStrategy(
                    asset=token,
                    balance=balance,
                    deposit_rate=0.0,
                    suggested_action="hold",
                    expected_annual_usd=0.0,
                    risks=["No tracked market — research manually"],
                ))
                continue

            action = "deposit" if market.deposit_apy > self.DEPOSIT_THRESHOLD_APY else "hold"
            annual = round(balance * market.deposit_apy, 2)
            risks = []
            if market.risk_score > 0.5:
                risks.append("Elevated DeFi risk — IL or contract exposure")
            if "cex" in market.chain:
                risks.append("Custodial platform risk")

            strategies.append(StakingStrategy(
                asset=token,
                balance=balance,
                deposit_rate=market.deposit_apy,
                suggested_action=action,
                expected_annual_usd=annual,
                risks=risks or ["Standard smart contract risk"],
            ))

        total_annual = sum(s.expected_annual_usd for s in strategies)
        return EthicalActionResult(
            accepted=True,
            mode="suggest_strategy",
            platform="defi",
            message=f"Strategy for ${sum(self._balances.values()):,.0f} — est. ${total_annual:,.0f}/yr (not guaranteed).",
            opportunities=[s.to_dict() for s in strategies],
            next_steps=[
                "Review risks per asset",
                "Call prepare_tx_intent() for each deposit you approve",
                "Sign transactions only in your own wallet",
            ],
            metadata={"executor": "creator_manual"},
        )

    def prepare_tx_intent(self, protocol: str, asset: str, action: str, amount: float) -> EthicalActionResult:
        if not self.user_address:
            return EthicalActionResult(
                accepted=False,
                mode="prepare_session",
                platform=protocol,
                message="Set user_address before preparing transaction intents.",
            )

        action_id = f"STAKE-{int(time.time())}-{protocol.lower()}"
        details = {
            "protocol": protocol,
            "asset": asset,
            "action": action,
            "amount": amount,
            "from_address": self.user_address,
            "unsigned": True,
            "private_key_required": False,
        }
        result = self.confirmation_gate.request(action_id, "defi_tx_intent", details)
        result.next_steps = [
            f"Open official {protocol} app",
            f"Prepare {action} {amount} {asset}",
            f"Connect wallet {self.user_address[:6]}...{self.user_address[-4:]}",
            "Review gas, APY, and contract address on-chain",
            f"Sign ONLY after verifying — action_id={action_id}",
        ]
        result.metadata["warning"] = "Agents never sign transactions. Creator executes."
        return result