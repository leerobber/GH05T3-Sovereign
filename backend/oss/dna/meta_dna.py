"""Meta-DNA v2 — agents/species adapt their own evolution rules to environment (Phase 4).

Success: agents in stagnant fitness regimes raise mutation; high performers stabilize.
Wired to Species + RoleEvolutionManager + OmniDNA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import random
import time


@dataclass
class MetaDNA:
    """Self-adaptive evolution parameters. Per-species or per-agent lineage."""
    mutation_rate: float = 0.10
    crossover_rate: float = 0.35
    selection_pressure: float = 0.55
    retention: float = 0.72
    exploration_bonus: float = 0.08
    history: List[float] = field(default_factory=list)
    adaptation_count: int = 0
    last_adapted: float = field(default_factory=time.time)
    niche: str = "general"

    def evolve_rules(self, fitness: float, stagnation_window: int = 5, environment_pressure: float = 0.0) -> None:
        """Adapt the meta rules based on recent fitness trajectory + external pressure.
        Stagnation (low variance) -> increase mutation.
        High consistent fitness -> tighten / lower rates for exploitation.
        """
        self.history.append(max(0.0, min(1.0, fitness)))
        if len(self.history) > 64:
            self.history = self.history[-64:]

        self.adaptation_count += 1
        self.last_adapted = time.time()

        if len(self.history) >= stagnation_window:
            recent = self.history[-stagnation_window:]
            var = (max(recent) - min(recent))
            mean_f = sum(recent) / len(recent)

            if var < 0.025:  # stagnation
                self.mutation_rate = min(0.48, self.mutation_rate * 1.25 + environment_pressure * 0.03)
                self.exploration_bonus = min(0.22, self.exploration_bonus + 0.015)
            elif mean_f > 0.82:
                self.mutation_rate = max(0.04, self.mutation_rate * 0.88)
                self.selection_pressure = min(0.78, self.selection_pressure * 1.03)
                self.exploration_bonus = max(0.02, self.exploration_bonus * 0.92)
            elif mean_f < 0.35:
                self.mutation_rate = min(0.40, self.mutation_rate * 1.12)
                self.crossover_rate = min(0.55, self.crossover_rate * 1.08)

        # environment (niche) pressure e.g. volatility world
        if environment_pressure > 0.15:
            self.mutation_rate = min(0.5, self.mutation_rate + environment_pressure * 0.1)
            self.crossover_rate = min(0.6, self.crossover_rate + 0.02)

        # bounds
        self.mutation_rate = max(0.03, min(0.5, self.mutation_rate))
        self.crossover_rate = max(0.1, min(0.65, self.crossover_rate))
        self.selection_pressure = max(0.2, min(0.85, self.selection_pressure))

    def apply_rules(self) -> Dict[str, float]:
        """Return current effective evolution knobs for consumer (evo engine, substrate)."""
        return {
            "mutation_rate": round(self.mutation_rate, 4),
            "crossover_rate": round(self.crossover_rate, 4),
            "selection_pressure": round(self.selection_pressure, 4),
            "retention": round(self.retention, 4),
            "exploration_bonus": round(self.exploration_bonus, 4),
            "adaptation_count": self.adaptation_count,
        }

    def get_effective_mutation(self, base: float = 0.06) -> float:
        return max(0.02, min(0.5, base * (self.mutation_rate / 0.1) + self.exploration_bonus * 0.5))

    def clone_for_new_species(self, niche: str) -> "MetaDNA":
        """Create diverged copy for a new species."""
        return MetaDNA(
            mutation_rate=self.mutation_rate * random.uniform(0.7, 1.35),
            crossover_rate=self.crossover_rate * random.uniform(0.85, 1.2),
            selection_pressure=self.selection_pressure * random.uniform(0.75, 1.3),
            retention=self.retention,
            exploration_bonus=self.exploration_bonus * random.uniform(0.6, 1.4),
            niche=niche,
            history=list(self.history[-3:]),
        )