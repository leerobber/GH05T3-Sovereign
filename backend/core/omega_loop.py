"""GH05T3 — Omega Loop: orchestrates chat cycles through inference backends."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

from core.config import BACKENDS

log = logging.getLogger("gh0st3.omega_loop")


class LoopMode(str, Enum):
    LOCAL    = "local"
    FALLBACK = "fallback"
    GHOST    = "ghost"


@dataclass
class LoopState:
    response:     str       = ""
    mode:         LoopMode  = LoopMode.LOCAL
    sage_score:   float     = 0.0
    sage_verdict: str       = "PENDING"
    latency_ms:   float     = 0.0
    cycle_id:     int       = 0
    backend_used: str       = "local"


class OmegaLoop:
    """
    Runs a request through the tri-GPU inference mesh, applies SAGE scoring,
    and records the cycle in KAIROS.
    """

    def __init__(self, memory=None, kairos=None, sage=None):
        self._memory = memory
        self._kairos = kairos
        self._sage   = sage
        self._cycle  = 0
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def cycle_count(self) -> int:
        return self._cycle

    @property
    def stats(self) -> dict:
        return {
            "cycle_count": self._cycle,
            "backends":    list(BACKENDS.keys()),
        }

    async def run(self, message: str, context: Optional[dict] = None) -> LoopState:
        self._cycle += 1
        t0    = time.perf_counter()
        state = LoopState(cycle_id=self._cycle)

        backend_order = [
            ("primary",  BACKENDS["primary"]),
            ("fallback", BACKENDS["fallback"]),
        ]

        for name, url in backend_order:
            try:
                resp = await self._client.post(
                    f"{url}/v1/chat/completions",
                    json={
                        "model":       "default",
                        "messages":    [{"role": "user", "content": message}],
                        "max_tokens":  800,
                        "temperature": 0.7,
                    },
                )
                data             = resp.json()
                state.response   = data["choices"][0]["message"]["content"].strip()
                state.backend_used = name
                state.mode       = LoopMode.LOCAL
                break
            except Exception as exc:
                log.warning(f"[Omega] backend '{name}' unavailable: {exc}")

        if not state.response:
            # Local GPU backends are offline — fall through to cloud provider chain
            try:
                from ghost_llm import chat_once, NoLLMError
                text, tag = await chat_once("omega", "", message)
                state.response   = text
                state.mode       = LoopMode.FALLBACK
                state.backend_used = tag
            except Exception as exc:
                log.warning("[Omega] cloud fallback failed: %s", exc)
                state.response   = (
                    "⚠ All inference backends offline. GH05T3 running in degraded mode."
                )
                state.mode       = LoopMode.GHOST
                state.backend_used = "none"

        # SAGE inline scoring
        if self._sage:
            result            = self._sage.evaluate(state.response, message)
            state.sage_score  = result["score"]
            state.sage_verdict = result["verdict"]
        else:
            word_count        = len(state.response.split())
            state.sage_score  = min(1.0, word_count / 100)
            state.sage_verdict = "PASS" if state.sage_score >= 0.5 else "REVISE"

        state.latency_ms = (time.perf_counter() - t0) * 1000

        if self._kairos:
            self._kairos.record_cycle(
                proposal=state.response[:500],
                verdict=state.sage_verdict,
                score=state.sage_score,
            )

        return state

    async def close(self):
        await self._client.aclose()
