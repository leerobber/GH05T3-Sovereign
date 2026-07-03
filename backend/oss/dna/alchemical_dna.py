"""Alchemical DNA v2 — trait transmutation recipes + catalysts (Phase 4).

Success: ~30% of agents experience meaningful transmutation per full experiment cycle.
Recipes trade off opposing traits; catalysts (worlds, labs) modulate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
import random


@dataclass
class AlchemicalDNA:
    recipes: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "rigor_to_creativity": {"rigor": -0.09, "creativity": 0.13, "innovation": 0.04},
        "risk_to_intuition": {"risk_tolerance": -0.07, "market_intuition": 0.11},
        "persistence_to_novelty": {"persistence": -0.06, "novelty_seeking": 0.09},
        "efficiency_to_alignment": {"efficiency": -0.05, "alignment": 0.08, "self_reflection": 0.04},
        "math_to_empathy": {"math": -0.08, "empathy": 0.10, "collaboration": 0.05},
    })
    catalysts: List[str] = field(default_factory=lambda: ["theory_lab", "volatility_world", "alignment_world", "meta_architecture", "pressure"])
    transmutation_log: List[Dict[str, Any]] = field(default_factory=list)
    total_transmuted: int = 0

    def transmute_traits(self, traits: Dict[str, float], recipe: str, intensity: float = 1.0) -> Dict[str, float]:
        if recipe not in self.recipes:
            return dict(traits)
        out = dict(traits)
        changes = {}
        for name, delta in self.recipes[recipe].items():
            eff_delta = delta * intensity
            before = out.get(name, 0.5)
            after = max(0.0, min(1.0, before + eff_delta))
            if abs(after - before) > 0.005:
                changes[name] = round(after - before, 4)
            out[name] = after
        if changes:
            self.total_transmuted += 1
            self.transmutation_log.append({"recipe": recipe, "changes": changes, "ts": len(self.transmutation_log)})
        return out

    def catalytic_transmute(self, traits: Dict[str, float], catalyst: str, intensity: float = 1.0) -> Tuple[Dict[str, float], str]:
        if catalyst not in self.catalysts:
            return dict(traits), "no_catalyst"
        # Prefer a context-aware recipe
        recipe = self._choose_recipe_for_catalyst(catalyst)
        new_traits = self.transmute_traits(traits, recipe, intensity)
        return new_traits, recipe

    def _choose_recipe_for_catalyst(self, catalyst: str) -> str:
        mapping = {
            "theory_lab": "rigor_to_creativity",
            "volatility_world": "risk_to_intuition",
            "alignment_world": "efficiency_to_alignment",
            "meta_architecture": "persistence_to_novelty",
            "pressure": random.choice(list(self.recipes.keys())),
        }
        return mapping.get(catalyst, next(iter(self.recipes)))

    def apply_to_population(self, agents_traits: Dict[str, Dict[str, float]], catalyst: str, fraction: float = 0.30) -> int:
        """Apply transmutation to ~fraction of agents. Returns count changed."""
        ids = list(agents_traits.keys())
        k = max(1, int(len(ids) * fraction))
        chosen = random.sample(ids, min(k, len(ids)))
        count = 0
        for aid in chosen:
            before = agents_traits[aid]
            after, recipe = self.catalytic_transmute(before, catalyst, intensity=random.uniform(0.7, 1.1))
            if after != before:
                agents_traits[aid] = after
                count += 1
        return count

    def transmutation_rate(self, total_agents: int) -> float:
        if total_agents < 1:
            return 0.0
        return min(1.0, self.total_transmuted / max(1, total_agents * 3))  # rough per agent * cycles heuristic

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_transmuted": self.total_transmuted,
            "unique_recipes_used": len(set(e["recipe"] for e in self.transmutation_log)),
            "num_recipes": len(self.recipes),
        }