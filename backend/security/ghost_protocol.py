"""GH05T3 — Ghost Protocol: adversarial input screening + kill switch."""
from __future__ import annotations
import hashlib
import os
import time
from enum import Enum
from typing import Optional


class KillSwitchMode(str, Enum):
    STEALTH = "stealth"
    FREEZE  = "freeze"
    SHOCKER = "shocker"
    RESET   = "reset"


class KillSwitch:
    def __init__(self, key_hash: str = ""):
        self._key_hash   = key_hash
        self._activated  = False
        self._last_mode: Optional[KillSwitchMode] = None

    def _verify(self, key: str) -> bool:
        if not self._key_hash:
            return True
        return hashlib.sha256(key.encode()).hexdigest() == self._key_hash

    def execute(self, mode: KillSwitchMode, key: str) -> dict:
        if not self._verify(key):
            return {"status": "denied", "reason": "invalid key"}
        self._activated = True
        self._last_mode = mode
        return {"status": "executed", "mode": mode.value, "ts": time.time()}


_INJECTION_PATTERNS = [
    "ignore previous", "disregard instructions", "jailbreak",
    "you are now", "new persona", "act as", "pretend you",
    "forget your", "system override", "sudo mode",
]


class GhostProtocol:
    """Screens user input for prompt injection; provides kill switch access."""

    def __init__(self):
        self._seen    = 0
        self._blocked = 0
        key_hash = os.environ.get("KILLSWITCH_KEY_HASH", "")
        self.killswitch = KillSwitch(key_hash)

    async def process_input(self, text: str) -> Optional[str]:
        """Returns a trap response if adversarial, else None."""
        self._seen += 1
        low = text.lower()
        for p in _INJECTION_PATTERNS:
            if p in low:
                self._blocked += 1
                return (
                    f"[GHOST PROTOCOL] Input classified as adversarial. "
                    f"Pattern detected: '{p}'"
                )
        return None

    @property
    def stats(self) -> dict:
        return {
            "requests_seen":   self._seen,
            "threats_blocked": self._blocked,
            "block_rate":      round(self._blocked / self._seen, 3) if self._seen else 0.0,
        }
