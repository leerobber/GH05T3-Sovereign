"""
Airdrop & faucet tracker — public listings only, user-owned wallets.

DOES NOT: claim to unowned wallets, multi-account farming, or automated claims.
Creator manually completes KYC/quiz/claim steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time

from .core import RateLimiter, HumanConfirmationGate, EthicalActionResult, UserConsentRecord


@dataclass
class AirdropListing:
    platform: str
    name: str
    reward_estimate_usd: float
    status: str  # active, upcoming, ended
    url: str
    requirements: List[str] = field(default_factory=list)
    claim_steps: List[str] = field(default_factory=list)
    wallet_must_be_owned: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


AIRDROP_SOURCES: Dict[str, Dict[str, Any]] = {
    "coinmarketcap": {
        "name": "CoinMarketCap Airdrops",
        "url": "https://coinmarketcap.com/airdrop/",
        "type": "token_giveaways",
        "reward_range": (1.0, 100.0),
    },
    "coingecko": {
        "name": "CoinGecko Airdrops",
        "url": "https://www.coingecko.com/en/airdrops",
        "type": "token_giveaways",
        "reward_range": (1.0, 50.0),
    },
    "binance_learn": {
        "name": "Binance Learn & Earn",
        "url": "https://www.binance.com/en/learn-and-earn",
        "type": "educational_quiz",
        "reward_range": (1.0, 50.0),
    },
    "coinbase_earn": {
        "name": "Coinbase Earn",
        "url": "https://www.coinbase.com/earn",
        "type": "educational_tasks",
        "reward_range": (1.0, 10.0),
    },
    "faucetpay": {
        "name": "FaucetPay Micro-faucets",
        "url": "https://faucetpay.io/",
        "type": "micro_faucets",
        "reward_range": (0.01, 1.0),
    },
}


class AirdropTracker:
    """
    Track legitimate airdrops/faucets. Rate-limited public discovery.
    Claims always require creator action on their own wallet.
    """

    def __init__(
        self,
        user_wallet: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
        confirmation_gate: Optional[HumanConfirmationGate] = None,
    ):
        self.user_wallet = user_wallet
        self.rate_limiter = rate_limiter or RateLimiter()
        self.confirmation_gate = confirmation_gate or HumanConfirmationGate()
        self.consents: Dict[str, UserConsentRecord] = {}
        self.claim_log: List[Dict[str, Any]] = []

    def set_wallet(self, address: str) -> None:
        """Set creator-owned wallet address (public address only — never private keys)."""
        if not address or not address.startswith("0x"):
            raise ValueError("Provide a valid user-owned wallet address (0x...)")
        self.user_wallet = address

    def register_consent(self, platform: str) -> UserConsentRecord:
        record = UserConsentRecord(
            platform=platform.lower(),
            consented_at=time.time(),
            scopes=["discover", "prepare_session", "track_earnings"],
        )
        self.consents[platform.lower()] = record
        return record

    def fetch_listings(self, source: Optional[str] = None) -> EthicalActionResult:
        """Return curated airdrop listings (public data; optional live fetch later)."""
        sources = [source.lower()] if source else list(AIRDROP_SOURCES.keys())
        listings: List[AirdropListing] = []

        for key in sources:
            if key not in AIRDROP_SOURCES:
                continue
            if not self.rate_limiter.allow(key):
                continue
            meta = AIRDROP_SOURCES[key]
            low, high = meta["reward_range"]
            est = round((low + high) / 2, 2)
            listings.append(
                AirdropListing(
                    platform=meta["name"],
                    name=f"{meta['name']} — active campaigns",
                    reward_estimate_usd=est,
                    status="active",
                    url=meta["url"],
                    requirements=[
                        "User-owned wallet only",
                        "Complete KYC/quiz manually if required",
                        "One account per platform",
                        "Read official rules",
                    ],
                    claim_steps=[
                        f"Open {meta['url']}",
                        "Verify campaign is official (not phishing)",
                        "Connect wallet YOU control",
                        "Complete tasks manually",
                        "Never share seed phrase",
                    ],
                )
            )

        listings.sort(key=lambda x: x.reward_estimate_usd, reverse=True)
        return EthicalActionResult(
            accepted=True,
            mode="discover_only",
            platform=source or "all",
            message=f"Tracking {len(listings)} airdrop sources. Creator claims manually.",
            opportunities=[l.to_dict() for l in listings],
            next_steps=[
                "Verify URLs are official — beware phishing",
                "Use a dedicated hot wallet for claims",
                "Call prepare_claim() before any on-chain action",
            ],
            requires_human_confirmation=False,
            metadata={"wallet_required": True, "auto_claim": False},
        )

    def prepare_claim(self, listing_name: str, platform: str) -> EthicalActionResult:
        """Queue claim prep — creator must confirm and execute manually."""
        if not self.user_wallet:
            return EthicalActionResult(
                accepted=False,
                mode="prepare_session",
                platform=platform,
                message="Set user-owned wallet via set_wallet() before preparing claims.",
                next_steps=["set_wallet('0xYourAddress')", "Never store private keys in repo"],
            )

        key = platform.lower().replace(" ", "_")
        for src_key, meta in AIRDROP_SOURCES.items():
            if meta["name"].lower() in platform.lower() or src_key in key:
                action_id = f"AIRDROP-{int(time.time())}-{src_key}"
                details = {
                    "platform": meta["name"],
                    "listing": listing_name,
                    "url": meta["url"],
                    "wallet": self.user_wallet,
                    "auto_claim": False,
                }
                result = self.confirmation_gate.request(action_id, "manual_airdrop_claim", details)
                result.next_steps = [
                    f"Open {meta['url']}",
                    f"Verify campaign: {listing_name}",
                    f"Use wallet {self.user_wallet[:6]}...{self.user_wallet[-4:]} (yours only)",
                    "Complete all steps manually in browser",
                    f"Confirm action_id={action_id} after claim",
                ]
                return result

        return EthicalActionResult(
            accepted=False,
            mode="prepare_session",
            platform=platform,
            message=f"Unknown airdrop platform: {platform}",
        )

    def track_claim(
        self,
        platform: str,
        token: str,
        value_usd: float,
        tx_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = {
            "platform": platform,
            "token": token,
            "value_usd": round(value_usd, 2),
            "tx_hash": tx_hash,
            "wallet": self.user_wallet,
            "timestamp": time.time(),
            "source": "creator_reported",
        }
        self.claim_log.append(entry)
        return entry

    def earnings_summary(self) -> Dict[str, Any]:
        total = sum(c["value_usd"] for c in self.claim_log)
        return {
            "claims": len(self.claim_log),
            "total_usd": round(total, 2),
            "wallet": self.user_wallet,
        }