"""
AethyroBridge — connects the Control Plane (SQLite/canonical) to the
Execution Plane (ChronosLedger binary mmap).

Responsibilities:
  - Sync genome objects → binary ledger slots on each cycle
  - Calculate dissent_boost for agents that diverge from population mean
  - Provide get_segment_for_user() for user-scoped ledger views

Control Plane (Stripe/billing, breakthroughs.db, learnings.jsonl) is
NEVER written here — this bridge only touches the Execution Plane ledger.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.oss.core.chronos_ledger import (
    DESIRE_NAMES,
    ChronosLedger,
    get_chronos_ledger,
)

LOG = logging.getLogger("ghost.aethyro_bridge")

# Desire molecule IDs in the same order as DESIRE_NAMES
_DESIRE_MOLS = [
    "M_DESIRE_KNOWLEDGE",
    "M_DESIRE_SKILL",
    "M_DESIRE_STATUS",
    "M_DESIRE_EXPERIENCE",
    "M_DESIRE_CREATION",
    "M_DESIRE_CONNECTION",
    "M_DESIRE_FREEDOM",
]


class AethyroBridge:
    """
    Mediator between genome objects and the binary execution ledger.

    Slot assignment is first-come-first-served. An agent_id → slot mapping
    is kept in memory; lost on restart (intentional — Execution Plane is
    volatile; canonical state lives in genome objects and Control Plane DBs).
    """

    def __init__(self, ledger: Optional[ChronosLedger] = None):
        self._ledger    = ledger or get_chronos_ledger()
        self._slots:    Dict[str, int] = {}    # agent_id → slot index
        self._next_slot: int = 0

    # ── Slot management ───────────────────────────────────────────────────────

    def get_slot(self, agent_id: str) -> Optional[int]:
        return self._slots.get(agent_id)

    def alloc_slot(self, agent_id: str) -> int:
        if agent_id not in self._slots:
            self._slots[agent_id] = self._next_slot
            self._next_slot += 1
        return self._slots[agent_id]

    def release_slot(self, agent_id: str) -> None:
        slot = self._slots.pop(agent_id, None)
        if slot is not None:
            self._ledger.write_zero(slot)

    # ── Sync: genome → ledger ─────────────────────────────────────────────────

    def sync_agent(self, agent_id: str, genome: Any, fitness: float = 0.5) -> int:
        """Pack desires + maturity + fitness from genome into a ledger slot."""
        slot = self.alloc_slot(agent_id)
        desires = self._extract_desires(genome)
        maturity = self._extract_maturity(genome)
        self._ledger.write_agent(slot, desires, maturity, fitness)
        return slot

    def sync_all(self, agents: Dict[str, Any]) -> int:
        """Sync every agent in a swarm dict. Returns count synced."""
        synced = 0
        for aid, agent in agents.items():
            fitness = agent.mean_fitness() if hasattr(agent, "mean_fitness") else 0.5
            self.sync_agent(aid, agent.genome, fitness)
            synced += 1
        return synced

    # ── Dissent calculation ───────────────────────────────────────────────────

    # ── Dissent coefficient ───────────────────────────────────────────────────

    _DISSENT_SENSITIVITY = 0.15   # controls how steeply the exponential rises
    _BOOST_CAP           = 10.0   # hard ceiling — prevents runaway from NaN→Inf

    @staticmethod
    def _sanitize(matrix: np.ndarray) -> np.ndarray:
        """Replace NaN / ±Inf with safe defaults before any arithmetic.

        float16 can produce NaN via 0*Inf or subnormal underflow in tight loops.
        Sanitizing once per pass protects every downstream calculation.
        """
        return np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)

    @staticmethod
    def calculate_dissent_boost(dist: float, sensitivity: float = 0.15) -> float:
        """
        Exponential dissent: e^(dist * sensitivity), capped at 10.0.

        Spawn trigger at boost > 2.0 → dist ≈ ln(2)/0.15 = 4.62.
        Cap prevents NaN-contaminated distance from producing Inf boost.
        """
        raw = float(np.exp(np.clip(dist, 0.0, 100.0) * sensitivity))
        return round(min(raw, AethyroBridge._BOOST_CAP), 4)

    def dissent_boost(self, agent_id: str, sensitivity: float = 0.15) -> float:
        """
        How much this agent diverges from the population desire mean.
        Returns e^(L2_distance * sensitivity). Outlier agents get rewarded.
        """
        slot = self._slots.get(agent_id)
        if slot is None or self._ledger.active_slots < 2:
            return 1.0
        matrix = self._sanitize(self._ledger.desires_matrix())   # NaN-safe
        if slot >= len(matrix):
            return 1.0
        mean          = matrix.mean(axis=0)
        agent_desires = matrix[slot]
        dist          = float(np.linalg.norm(agent_desires - mean))
        return self.calculate_dissent_boost(dist, sensitivity)

    def population_dissent_pass(
        self,
        agents:      Dict[str, Any],
        sensitivity: float = 0.15,
    ) -> Dict[str, float]:
        """
        Vectorized exponential dissent pass.

        1. Sanitize the full desires matrix (nan_to_num).
        2. Compute centroid and per-agent L2 distances in one numpy call.
        3. Clip boosts to _BOOST_CAP; flag agents above spawn threshold.
        Returns {agent_id: boost}.
        """
        if self._ledger.active_slots < 2:
            return {}

        matrix = self._sanitize(self._ledger.desires_matrix())   # (n, 7) float32
        centroid = matrix.mean(axis=0)                            # (7,)

        # Vectorized L2 distances for all active slots at once
        distances = np.linalg.norm(matrix - centroid, axis=1)    # (n,)
        raw_boosts = np.exp(np.clip(distances * sensitivity, 0, 100.0))
        clipped    = np.clip(raw_boosts, 1.0, self._BOOST_CAP)   # floor=1, cap=10

        boosts: Dict[str, float] = {}
        for aid, agent in agents.items():
            slot = self._slots.get(aid)
            if slot is None or slot >= len(clipped):
                continue
            boost = round(float(clipped[slot]), 4)
            boosts[aid] = boost
            if boost > 1.01 and hasattr(agent, "fitness_history"):
                current = agent.fitness_history[-1] if agent.fitness_history else 0.5
                agent.fitness_history.append(min(1.0, current * boost))
        return boosts

    # ── User segment (tiered compute) ─────────────────────────────────────────

    def get_segment_for_user(self, user_id: str, tier: str = "novice") -> memoryview:
        """
        Return a memoryview slice of the ledger for a user's swarm segment.
        Tier controls how many slots are allocated:
            novice → 50, specialist → 200, elite → 500, architect → 2000
        """
        from backend.oss.core.chronos_ledger import STRUCT_SIZE
        tier_slots = {"novice": 50, "specialist": 200, "elite": 500, "architect": 2000}
        n_slots  = tier_slots.get(str(tier).lower(), 50)
        slot     = self.alloc_slot(f"user_{user_id}")
        offset   = slot * STRUCT_SIZE
        end      = min(offset + n_slots * STRUCT_SIZE, self._ledger._size)
        self._ledger._map.seek(0)
        raw = self._ledger._map.read(self._ledger._size)
        return memoryview(bytearray(raw))[offset:end]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_desires(self, genome: Any) -> Tuple[float, ...]:
        desires = []
        for mol_id in _DESIRE_MOLS:
            try:
                v = genome.get_value("desire", mol_id)
            except Exception:
                v = 0.1
            desires.append(float(v))
        return tuple(desires)

    def _extract_maturity(self, genome: Any) -> int:
        try:
            from backend.oss.genomic.context_genome import score_context_maturity
            return score_context_maturity(genome)
        except Exception:
            return 1

    def ledger_stats(self) -> Dict[str, Any]:
        return self._ledger.stats()


# ── Singleton ─────────────────────────────────────────────────────────────────

_bridge: Optional[AethyroBridge] = None


def get_aethyro_bridge() -> AethyroBridge:
    global _bridge
    if _bridge is None:
        _bridge = AethyroBridge()
    return _bridge
