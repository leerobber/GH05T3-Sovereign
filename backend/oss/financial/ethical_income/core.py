"""
Ethical income core — consent, rate limits, human confirmation, compliance gate.

Agents ASSIST the creator; they do not replace human judgment or violate platform ToS.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import time

from ..lawful_boundaries import PROHIBITED_PATHS
from ..revenue_discovery import LawfulRevenueDiscovery

ASSISTANCE_MODES = (
    "discover_only",       # list opportunities — no credentials
    "prepare_session",     # open URLs + checklist — user executes
    "suggest_strategy",    # read-only analysis — user confirms action
    "track_earnings",      # log creator-reported income
)

# Hard ceiling: max outbound requests per platform per minute (public data only).
DEFAULT_RATE_LIMITS: Dict[str, float] = {
    "prolific": 6.0,
    "mturk": 6.0,
    "coinmarketcap": 4.0,
    "coingecko": 10.0,
    "splinterlands": 8.0,
    "aave": 12.0,
    "compound": 12.0,
    "default": 6.0,
}


class HumanConfirmationRequired(Exception):
    """Raised when an action requires explicit creator approval before proceeding."""

    def __init__(self, action: str, details: Dict[str, Any]):
        self.action = action
        self.details = details
        super().__init__(f"Human confirmation required for: {action}")


@dataclass
class UserConsentRecord:
    """Explicit opt-in per platform. Revocable. Credentials never stored in repo."""

    platform: str
    consented_at: float
    scopes: List[str]  # e.g. ["discover", "prepare_session", "track_earnings"]
    user_id: str = "creator"
    revoked: bool = False
    notes: str = ""

    def is_active(self, scope: str) -> bool:
        return not self.revoked and scope in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RateLimiter:
    """Token-bucket rate limiter per platform key."""

    def __init__(self, limits: Optional[Dict[str, float]] = None):
        self.limits = dict(DEFAULT_RATE_LIMITS)
        if limits:
            self.limits.update(limits)
        self._last_call: Dict[str, float] = {}

    def allow(self, platform: str) -> bool:
        key = platform.lower()
        min_interval = 60.0 / self.limits.get(key, self.limits["default"])
        now = time.time()
        last = self._last_call.get(key, 0.0)
        if now - last < min_interval:
            return False
        self._last_call[key] = now
        return True

    def wait_seconds(self, platform: str) -> float:
        key = platform.lower()
        min_interval = 60.0 / self.limits.get(key, self.limits["default"])
        elapsed = time.time() - self._last_call.get(key, 0.0)
        return max(0.0, min_interval - elapsed)


@dataclass
class EthicalActionResult:
    """Uniform result envelope for all ethical income assistants."""

    accepted: bool
    mode: str
    platform: str
    message: str
    opportunities: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    requires_human_confirmation: bool = True
    prohibited_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HumanConfirmationGate:
    """Blocks execution until creator explicitly confirms."""

    def __init__(self):
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._approved: Dict[str, bool] = {}

    def request(self, action_id: str, action: str, details: Dict[str, Any]) -> EthicalActionResult:
        self._pending[action_id] = {"action": action, "details": details, "requested_at": time.time()}
        return EthicalActionResult(
            accepted=True,
            mode="prepare_session",
            platform=details.get("platform", "unknown"),
            message=f"Action '{action}' queued. Creator must confirm before execution.",
            next_steps=[
                f"Review action_id={action_id}",
                "Call confirm(action_id) to approve",
                "Never auto-submit surveys, game moves, or on-chain transactions",
            ],
            requires_human_confirmation=True,
            metadata={"action_id": action_id, "action": action, "details": details},
        )

    def confirm(self, action_id: str) -> bool:
        if action_id not in self._pending:
            return False
        self._approved[action_id] = True
        return True

    def is_confirmed(self, action_id: str) -> bool:
        return self._approved.get(action_id, False)

    def pop_confirmed(self, action_id: str) -> Optional[Dict[str, Any]]:
        if not self.is_confirmed(action_id):
            return None
        entry = self._pending.pop(action_id, None)
        self._approved.pop(action_id, None)
        return entry


class ComplianceGate:
    """Rejects prohibited automation before any assistant runs."""

    def __init__(self):
        self._discovery = LawfulRevenueDiscovery()

    def check(self, prompt: str) -> Optional[EthicalActionResult]:
        prohibited = self._discovery.is_prohibited(prompt)
        if prohibited:
            return EthicalActionResult(
                accepted=False,
                mode="discover_only",
                platform="compliance",
                message=(
                    f"REJECTED: {prohibited} violates lawful boundaries. "
                    "Use human-in-the-loop assistants only."
                ),
                prohibited_path=prohibited,
                next_steps=[
                    "Use discover_opportunities() to list lawful tasks",
                    "Creator completes surveys/games manually",
                    "Claim airdrops only to wallets you own",
                ],
                metadata={"prohibited_paths": list(PROHIBITED_PATHS)},
            )
        return None