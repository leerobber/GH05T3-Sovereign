"""
Evolution-of-Evolution (Adaptive Mutation Engine)

Extends the basic evolution with:
- Fitness-history driven mutation rates
- Trait-driven strategies (high math → smaller, more precise mutations)
- Lineage awareness (future: use mutation history)

This is the "meta" part of OmniDNA v2 / Omni-Evolution v2.
"""

from typing import List, Dict
import random
from backend.oss.omni_dna import OmniDNA


class AdaptiveMutationEngine:
    """
    Computes context-sensitive mutation strength and applies it.
    Used by RoleEvolutionManager for elite roles.
    """

    def __init__(self, base_rate: float = 0.025):
        self.base_rate = base_rate

    def compute_rate(self, fitness_history: List[float]) -> float:
        if not fitness_history or len(fitness_history) < 2:
            return self.base_rate

        recent = fitness_history[-5:]
        avg = sum(recent) / len(recent)
        trend = recent[-1] - recent[0] if len(recent) > 1 else 0

        rate = self.base_rate
        if avg > 0.82:
            rate *= 0.45          # exploit — very conservative
        elif avg > 0.65:
            rate *= 0.75
        elif avg < 0.35:
            rate *= 1.8           # explore
        else:
            rate *= 1.0

        # If improving, reduce rate further
        if trend > 0.05:
            rate *= 0.7

        return max(0.005, min(0.18, rate))

    def mutate(self, dna: OmniDNA, fitness_history: List[float] = None) -> List[str]:
        """
        Perform adaptive mutation on the DNA.
        Returns list of changed traits.
        """
        rate = self.compute_rate(fitness_history or [])
        traits = dna.get_traits()
        changed = []

        # Trait-driven bias
        math_bias = traits.get("math", 0.55)
        reflection = traits.get("self_reflection", 0.55)
        novelty = traits.get("novelty_seeking", 0.55)

        for name, value in list(traits.items()):
            # Base delta
            delta = random.gauss(0, rate)

            # Specialists mutate more precisely in their domain
            if name in ("math", "pattern_detection") and math_bias > 0.8:
                delta *= 0.6
            if name in ("self_reflection", "alignment") and reflection > 0.8:
                delta *= 0.55
            if name == "novelty_seeking" and novelty > 0.85:
                delta *= 1.3  # explorers explore harder

            new_val = max(0.10, min(0.95, value + delta))
            if abs(new_val - value) > 0.004:
                dna.traits[name] = round(new_val, 4)
                changed.append(name)

        # Occasionally evolve meta-DNA as well
        if random.random() < 0.18:
            try:
                dna.evolve_meta(strength=rate * 0.6)
            except Exception:
                pass

        return changed
