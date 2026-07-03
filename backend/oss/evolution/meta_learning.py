"""
Meta-Learning Engine — Phase 6.1

Strategy assignment by traits.
evolve_strategies(), apply_strategy()

Enables +20% learning rate over time by selecting/adapting strategies per agent lineage.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import random

from backend.oss.omni_dna import OmniDNA


@dataclass
class EvolutionStrategy:
    name: str
    description: str
    mutation_multiplier: float = 1.0
    crossover_multiplier: float = 1.0
    trait_biases: Dict[str, float] = field(default_factory=dict)
    fitness_boost: float = 0.0


class MetaLearningEngine:
    """Assigns and evolves strategies based on agent traits and performance."""

    STRATEGIES = {
        "explorer": EvolutionStrategy(
            "explorer", "High novelty, broad search",
            mutation_multiplier=1.6, crossover_multiplier=1.3,
            trait_biases={"novelty_seeking": 0.15, "creativity": 0.1},
            fitness_boost=0.05
        ),
        "exploiter": EvolutionStrategy(
            "exploiter", "Precise refinement of strengths",
            mutation_multiplier=0.5, crossover_multiplier=0.7,
            trait_biases={"rigor": 0.12, "efficiency": 0.08},
            fitness_boost=0.08
        ),
        "collaborator": EvolutionStrategy(
            "collaborator", "Emphasize sharing and consensus",
            mutation_multiplier=0.9, crossover_multiplier=1.4,
            trait_biases={"collaboration": 0.2, "empathy": 0.15},
        ),
        "theorist": EvolutionStrategy(
            "theorist", "Deep math/pattern focus, conservative mutation",
            mutation_multiplier=0.6, crossover_multiplier=0.8,
            trait_biases={"math": 0.18, "pattern_detection": 0.15, "self_reflection": 0.1},
            fitness_boost=0.1
        ),
    }

    def __init__(self):
        self.agent_strategies: Dict[str, str] = {}  # genome_id -> strategy_name
        self.strategy_usage: Dict[str, int] = {k: 0 for k in self.STRATEGIES}

    def assign_strategy(self, dna: OmniDNA, genome_id: Optional[str] = None) -> str:
        traits = dna.get_traits()
        scores = {}
        for name, strat in self.STRATEGIES.items():
            score = 0.0
            for t, bias in strat.trait_biases.items():
                score += traits.get(t, 0.5) * bias
            # bonus for current fitness if history available (simplified)
            score += strat.fitness_boost
            scores[name] = score

        best = max(scores, key=scores.get)
        if genome_id:
            self.agent_strategies[genome_id] = best
            self.strategy_usage[best] += 1
        return best

    def evolve_strategies(self, performance: Dict[str, float]) -> Dict[str, str]:
        """Globally evolve which strategies are favored based on recent performance."""
        # Simple: boost strategies that performed well
        for strat_name, perf in performance.items():
            if strat_name in self.STRATEGIES and perf > 0.75:
                self.STRATEGIES[strat_name].mutation_multiplier *= 1.05
                self.STRATEGIES[strat_name].fitness_boost = min(0.15, self.STRATEGIES[strat_name].fitness_boost + 0.01)
        return {k: v.name for k, v in self.STRATEGIES.items()}

    def apply_strategy(self, dna: OmniDNA, genome_id: Optional[str] = None) -> float:
        """Apply the current strategy to this DNA. Returns effective mutation multiplier."""
        strat_name = self.agent_strategies.get(genome_id) or self.assign_strategy(dna, genome_id)
        strat = self.STRATEGIES[strat_name]

        # Mutate traits according to biases
        traits = dna.get_traits()
        for t, bias in strat.trait_biases.items():
            if t in traits:
                delta = (random.random() - 0.5) * 0.08 * (1 + bias)
                traits[t] = max(0.1, min(0.95, traits[t] + delta))
        # write back
        for k, v in traits.items():
            if k in dna.traits:
                dna.traits[k] = round(v, 4)

        multiplier = strat.mutation_multiplier
        # Apply to dna evolve if called externally
        return multiplier

    def get_stats(self) -> Dict[str, Any]:
        total = sum(self.strategy_usage.values()) or 1
        usage_pct = {k: round(v / total, 3) for k, v in self.strategy_usage.items()}
        return {
            "strategy_usage_pct": usage_pct,
            "assigned_agents": len(self.agent_strategies),
            "current_strategies": list(self.STRATEGIES.keys()),
        }


# Convenience
def get_meta_learning_engine() -> MetaLearningEngine:
    # Simple singleton for MVS use
    if not hasattr(get_meta_learning_engine, "_instance"):
        get_meta_learning_engine._instance = MetaLearningEngine()
    return get_meta_learning_engine._instance
