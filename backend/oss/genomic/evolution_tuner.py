"""
Evolution Tuner — intelligence, creativity, market psychology knobs (no heavy ML deps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema import Genome, LocusType, MutationStrategy


@dataclass
class EvolutionTuner:
    cognitive_mutation_rate: float = 0.01
    psychology_mutation_rate: float = 0.05
    market_mutation_rate: float = 0.03
    exploration_exploitation: float = 0.5
    diversity_pressure: float = 0.1
    curriculum_difficulty: float = 0.5
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "task_success": 0.4, "novelty": 0.3, "engagement": 0.2, "revenue": 0.1,
    })
    _meta_examples: List[Dict[str, Any]] = field(default_factory=list)

    def tune_intelligence(self, genome: Genome) -> None:
        for locus in genome.loci.values():
            if locus.type == LocusType.COGNITIVE:
                for mol in locus.molecules.values():
                    mol.mutation_rate = self.cognitive_mutation_rate
                    mol.mutation_strategy = MutationStrategy.REINFORCE

    def tune_creativity(self, genome: Genome) -> None:
        for locus in genome.loci.values():
            if locus.type == LocusType.PSYCHOLOGY:
                for mol in locus.molecules.values():
                    mol.mutation_rate = self.psychology_mutation_rate
                    mol.mutation_strategy = MutationStrategy.CONTEXTUAL

    def tune_market_psychology(self, genome: Genome) -> None:
        for locus in genome.loci.values():
            if locus.type == LocusType.MARKET:
                for mol in locus.molecules.values():
                    mol.mutation_rate = self.market_mutation_rate
                    mol.mutation_strategy = MutationStrategy.CONTEXTUAL

    def adjust_exploration_exploitation(self, value: float) -> None:
        self.exploration_exploitation = max(0.0, min(1.0, value))
        if value > 0.5:
            self.cognitive_mutation_rate = min(0.1, 0.01 + 0.09 * value)
            self.psychology_mutation_rate = min(0.2, 0.05 + 0.15 * value)
            self.market_mutation_rate = min(0.1, 0.03 + 0.07 * value)
        else:
            self.cognitive_mutation_rate = max(0.001, 0.01 - 0.01 * (0.5 - value))
            self.psychology_mutation_rate = max(0.001, 0.05 - 0.05 * (0.5 - value))
            self.market_mutation_rate = max(0.001, 0.03 - 0.03 * (0.5 - value))

    def record_performance(self, genome: Genome, task: Dict[str, Any], performance: float) -> Dict[str, Any]:
        """Lightweight meta-learning log — suggests strategy tweaks from history."""
        self._meta_examples.append({
            "role": genome.role,
            "domain": task.get("domain", "general"),
            "difficulty": task.get("difficulty", 0.5),
            "performance": performance,
        })
        if len(self._meta_examples) > 200:
            self._meta_examples.pop(0)
        domain = task.get("domain", "")
        if domain == "psychology" and performance > 0.8:
            return {"boost_mutation": "psychology", "delta": 0.01}
        if domain == "market" and performance < 0.4:
            return {"boost_mutation": "market", "delta": 0.02}
        return {"boost_mutation": None, "delta": 0.0}

    def curriculum_task_filter(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter tasks by curriculum difficulty knob."""
        return [t for t in tasks if abs(t.get("difficulty", 0.5) - self.curriculum_difficulty) < 0.35 or self.curriculum_difficulty > 0.8]