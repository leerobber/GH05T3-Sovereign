"""GH05T3 — Omega Loop: orchestrates chat cycles through inference backends."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
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
    response:            str       = ""
    mode:                LoopMode  = LoopMode.LOCAL
    sage_score:          float     = 0.0
    sage_verdict:        str       = "PENDING"
    latency_ms:          float     = 0.0
    cycle_id:            int       = 0
    backend_used:        str       = "local"
    # Sentinel + entropy — populated after SAGE evaluation
    sentinel_viability:  float     = 0.0
    entropy_drift:       float     = 0.0
    agent_id:            str       = "unknown"


class OmegaLoop:
    """
    Runs a request through the tri-GPU inference mesh, applies SAGE scoring
    (with Sentinel gate + entropy drift), and records the cycle in KAIROS.
    """

    def __init__(self, memory=None, kairos=None, sage=None, entropy_tracker=None):
        self._memory          = memory
        self._kairos          = kairos
        self._sage            = sage
        self._entropy_tracker = entropy_tracker
        self._cycle           = 0
        self._client          = httpx.AsyncClient(timeout=30.0)

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

        agent_id = (context or {}).get("agent_id", "omega")
        state.agent_id = agent_id

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
                data               = resp.json()
                state.response     = data["choices"][0]["message"]["content"].strip()
                state.backend_used = name
                state.mode         = LoopMode.LOCAL
                break
            except Exception as exc:
                log.warning(f"[Omega] backend '{name}' unavailable: {exc}")

        if not state.response:
            try:
                from ghost_llm import chat_once, NoLLMError
                text, tag          = await chat_once("omega", "", message)
                state.response     = text
                state.mode         = LoopMode.FALLBACK
                state.backend_used = tag
            except Exception as exc:
                log.warning("[Omega] cloud fallback failed: %s", exc)
                state.response     = (
                    "⚠ All inference backends offline. GH05T3 running in degraded mode."
                )
                state.mode         = LoopMode.GHOST
                state.backend_used = "none"

        # Entropy drift — computed before SAGE so D_ε feeds the Sentinel gate
        if self._entropy_tracker and state.response:
            try:
                state.entropy_drift = await self._entropy_tracker.compute_drift(
                    state.response, agent_id, cycle_id=self._cycle
                )
            except Exception as exc:
                log.warning("[Omega] entropy tracking failed: %s", exc)

        # Kill switch signal from context (0 = block, 1 = normal)
        human_sig = int((context or {}).get("human_sig", 1))

        # SAGE inline scoring (now includes Sentinel gate)
        if self._sage:
            result                  = self._sage.evaluate(
                state.response, message,
                entropy_drift = state.entropy_drift,
                human_sig     = human_sig,
            )
            state.sage_score        = result["score"]
            state.sage_verdict      = result["verdict"]
            state.sentinel_viability = result.get("sentinel_viability", 0.0)
        else:
            word_count             = len(state.response.split())
            state.sage_score       = min(1.0, word_count / 100)
            state.sage_verdict     = "PASS" if state.sage_score >= 0.5 else "REVISE"

        state.latency_ms = (time.perf_counter() - t0) * 1000

        if self._kairos:
            self._kairos.record_cycle(
                proposal           = state.response[:500],
                verdict            = state.sage_verdict,
                score              = state.sage_score,
                sentinel_viability = state.sentinel_viability,
                entropy_drift      = state.entropy_drift,
                agent_id           = agent_id,
            )

        return state

    async def close(self):
        await self._client.aclose()
