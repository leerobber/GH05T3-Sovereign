"""
OmniMind Emergent Goal Generator v2 (for OmniMind v1.5)

Uses canonical memory + weighted consensus signals + counters for stable emergent goals.
Wired into theory_lab + loop for real evolution pressure.
"""

from __future__ import annotations
from typing import Dict, Any, List
from collections import Counter
from backend.oss.omni_mind import OmniMind


class EmergentGoalEngine:
    def __init__(self, mind: OmniMind):
        self.mind = mind
        if not hasattr(self.mind, 'goals'):
            self.mind.goals = []

    def generate_goals(self) -> List[Dict[str, Any]]:
        # v2: use canonical + theory_lab tagged memories + mind seeds for stable goals
        memories = getattr(self.mind.state, 'shared_memory', []) if hasattr(self.mind, 'state') else getattr(self.mind, 'shared_memory', [])
        if isinstance(memories, dict):
            memories = list(memories.values())

        texts = [str(m) for m in memories]
        tokens = []
        for t in texts:
            tokens.extend([w for w in t.lower().split() if len(w) > 4])

        counts = Counter(tokens)
        goals = []

        # Pull seeds from v1.5 mind api if present
        if hasattr(self.mind, 'generate_goal_seeds'):
            for seed in self.mind.generate_goal_seeds()[:3]:
                goals.append({
                    "goal_id": f"goal_seed_{seed['seed']}_{len(self.mind.goals)}",
                    "description": seed.get("suggested", f"Deep theory on {seed['seed']}"),
                    "priority": 0.9,
                    "required_traits": ["math", "self_reflection", "pattern_detection"],
                    "source": "mind_v15"
                })

        if counts.get("volatility", 0) > 3 or counts.get("regime", 0) > 2:
            goals.append({
                "goal_id": f"goal_volatility_{len(getattr(self.mind, 'goals', []))}",
                "description": "Design a robust formal theory and mathematical model for high-volatility regimes using regime HMM + alignment regularizer.",
                "priority": 0.95,
                "required_traits": ["math", "pattern_detection", "self_reflection"],
                "source": "theory_lab"
            })

        if counts.get("alignment", 0) > 2 or counts.get("coherence", 0) > 2:
            goals.append({
                "goal_id": f"goal_governance_{len(getattr(self.mind, 'goals', []))}",
                "description": "Propose constitutional updates, drift detectors, and pareto multi-species alignment mechanisms.",
                "priority": 0.92,
                "required_traits": ["alignment", "self_reflection"],
                "source": "theory_lab"
            })

        for g in goals:
            if not hasattr(self.mind, 'goals'):
                self.mind.goals = []
            self.mind.goals.append(g)

        return goals
