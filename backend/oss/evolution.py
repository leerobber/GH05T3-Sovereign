"""
Role-Adaptive Evolution (Phase 2+)

Different roles evolve differently, creating specialization pressure.
This is wired into the MVS via OmniDNA.

All evolution still goes through the MVS spine.
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional
from backend.oss.omni_dna import OmniDNA
from backend.oss.evolution_adaptive import AdaptiveMutationEngine
from backend.oss.speciation import get_speciation_engine


class RoleEvolutionManager:
    """Role-specific evolution rules for true speciation (now with adaptive meta-evolution)."""

    def __init__(self):
        self.adaptive = AdaptiveMutationEngine()

    def evolve_for_role(self, dna: OmniDNA, role: str, fitness: float, fitness_history: List[float] = None) -> None:
        role = role.upper()
        history = fitness_history or []

        # Speciation isolation: if this dna belongs to a species, only evolve within compatible group
        spec = get_speciation_engine()
        # For now we just let adaptive run; isolation is enforced at memetic/crossover level

        # Evolution-of-Evolution: use adaptive engine for elites + high performers
        if role in ("THEORIST_ELITE", "ARCHITECT_ELITE", "PHILOSOPHER_ELITE") or fitness > 0.78:
            changed = self.adaptive.mutate(dna, history)
            if changed:
                dna.evolve(strength=self.adaptive.compute_rate(history) * 0.3)
            return

        # v2.0 meta-DNA for others occasionally
        if fitness > 0.75 and random.random() < 0.12:
            dna.evolve_meta(strength=0.02)

        if role == "SCIENTIST":
            self._evolve_scientist(dna, fitness)
        elif role == "INVESTOR":
            self._evolve_investor(dna, fitness)
        elif role == "BUILDER":
            self._evolve_builder(dna, fitness)
        elif role == "OPERATOR":
            self._evolve_operator(dna, fitness)
        elif role == "GOVERNOR":
            self._evolve_governor(dna, fitness)
        elif role == "THEORIST_ELITE":
            self._evolve_theorist_elite(dna, fitness)
        elif role == "ARCHITECT_ELITE":
            self._evolve_architect_elite(dna, fitness)
        elif role == "PHILOSOPHER_ELITE":
            self._evolve_philosopher_elite(dna, fitness)
        else:
            dna.evolve()

    def _boost_trait(self, dna: OmniDNA, name: str, delta: float):
        traits = dna.get_traits()
        current = traits.get(name, 0.5)
        new_val = max(0.1, min(0.95, current + delta))
        # Since traits is a dict on dna, we mutate it directly for simplicity in MVS
        dna.traits[name] = new_val

    def _evolve_scientist(self, dna: OmniDNA, fitness: float):
        if fitness > 0.7:
            self._boost_trait(dna, "rigor", 0.05)
            self._boost_trait(dna, "novelty_seeking", 0.04)
        else:
            self._boost_trait(dna, "self_reflection", 0.03)
        dna.evolve(strength=0.06 if fitness > 0.7 else 0.1)

    def _evolve_investor(self, dna: OmniDNA, fitness: float):
        if fitness > 0.7:
            self._boost_trait(dna, "risk_tolerance", 0.04)
            self._boost_trait(dna, "market_intuition", 0.05)
        else:
            self._boost_trait(dna, "self_reflection", 0.04)
        dna.evolve(strength=0.05 if fitness > 0.7 else 0.09)

    def _evolve_builder(self, dna: OmniDNA, fitness: float):
        if fitness > 0.7:
            self._boost_trait(dna, "creativity", 0.05)
            self._boost_trait(dna, "empathy", 0.04)
        else:
            self._boost_trait(dna, "innovation", 0.03)
        dna.evolve(strength=0.06 if fitness > 0.7 else 0.1)

    def _evolve_operator(self, dna: OmniDNA, fitness: float):
        if fitness > 0.7:
            self._boost_trait(dna, "efficiency", 0.05)
            self._boost_trait(dna, "persistence", 0.04)
        else:
            self._boost_trait(dna, "self_reflection", 0.03)
        dna.evolve(strength=0.05 if fitness > 0.7 else 0.08)

    def _evolve_governor(self, dna: OmniDNA, fitness: float):
        if fitness > 0.7:
            self._boost_trait(dna, "alignment", 0.05)
            self._boost_trait(dna, "self_reflection", 0.04)
        else:
            self._boost_trait(dna, "empathy", 0.03)
        dna.evolve(strength=0.04 if fitness > 0.7 else 0.07)

    def _evolve_theorist_elite(self, dna: OmniDNA, fitness: float):
        """Very conservative, directed evolution for THEORIST_ELITE.
        Prioritizes stability and depth over random mutation.
        """
        target_traits = ["math", "pattern_detection", "self_reflection", "creativity", "alignment", "rigor"]
        if fitness > 0.8:
            # Elite performance: strong directed refinement
            for t in target_traits:
                delta = 0.025 if t in ["math", "pattern_detection", "self_reflection"] else 0.02
                self._boost_trait(dna, t, delta)
            dna.evolve(strength=0.02)  # minimal mutation
        elif fitness > 0.65:
            self._boost_trait(dna, random.choice(target_traits[:4]), 0.015)
            dna.evolve(strength=0.03)
        else:
            # Conservative recovery
            self._boost_trait(dna, "self_reflection", 0.01)
            dna.evolve(strength=0.04)

    def _evolve_architect_elite(self, dna: OmniDNA, fitness: float):
        if fitness > 0.75:
            self._boost_trait(dna, "creativity", 0.04)
            self._boost_trait(dna, "efficiency", 0.03)
            self._boost_trait(dna, "collaboration", 0.03)
        dna.evolve(strength=0.04 if fitness > 0.75 else 0.06)

    def _evolve_philosopher_elite(self, dna: OmniDNA, fitness: float):
        if fitness > 0.75:
            self._boost_trait(dna, "self_reflection", 0.05)
            self._boost_trait(dna, "alignment", 0.04)
        dna.evolve(strength=0.03 if fitness > 0.75 else 0.05)


# ── Phase 2 Week 5: Evolutionary pressure from world fitness ──────────────────

from dataclasses import dataclass, field as dc_field  # noqa: E402
from typing import Tuple as TypingTuple  # noqa: E402


@dataclass
class PressureStats:
    cycle: int
    top_fraction: float
    top_agents: List[str] = dc_field(default_factory=list)
    spawn_boost: int = 0
    phased_out: int = 0
    pareto_ratio: float = 0.0


class EvolutionaryPressureEngine:
    """Top 20% by volatility fitness get boosted reproduction."""

    TOP_FRACTION = 0.20
    SPAWN_BOOST_MULTIPLIER = 2

    def __init__(self) -> None:
        self._history: List[PressureStats] = []
        self._spawn_counts: Dict[str, int] = {}

    def rank_agents(self, results: List[TypingTuple[str, float, Any]]) -> List[TypingTuple[str, float]]:
        return sorted(results, key=lambda x: x[1], reverse=True)

    def apply_pressure(self, results: List[TypingTuple[str, float, Any]], cycle: int, spawn_fn) -> PressureStats:
        ranked = self.rank_agents(results)
        n = len(ranked)
        if n == 0:
            return PressureStats(cycle=cycle, top_fraction=self.TOP_FRACTION)

        top_n = max(1, int(n * self.TOP_FRACTION))
        top = ranked[:top_n]
        bottom = ranked[top_n:]
        top_ids = [gid for gid, _, _ in top]

        spawn_boost = 0
        for gid, score, _ in top:
            if score < 0.5:
                continue
            for _ in range(self.SPAWN_BOOST_MULTIPLIER):
                if spawn_fn(gid):
                    spawn_boost += 1
                    self._spawn_counts[gid] = self._spawn_counts.get(gid, 0) + 1

        phased_out = len([gid for gid, sc, _ in bottom if sc < 0.35])
        total_spawns = sum(self._spawn_counts.values()) or 1
        top_spawns = sum(self._spawn_counts.get(gid, 0) for gid in top_ids)
        pareto = top_spawns / total_spawns if total_spawns else 0.0

        stats = PressureStats(
            cycle=cycle,
            top_fraction=self.TOP_FRACTION,
            top_agents=top_ids,
            spawn_boost=spawn_boost,
            phased_out=phased_out,
            pareto_ratio=round(pareto, 4),
        )
        self._history.append(stats)
        return stats

    def last_stats(self) -> Dict[str, Any]:
        if not self._history:
            return {}
        s = self._history[-1]
        return {
            "cycle": s.cycle,
            "top_agents": s.top_agents,
            "spawn_boost": s.spawn_boost,
            "phased_out": s.phased_out,
            "pareto_ratio": s.pareto_ratio,
        }
