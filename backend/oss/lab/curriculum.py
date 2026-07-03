"""
Curriculum Generator — Self-generated training curriculum by Theorists.

Theorists (and the net) propose:
- New theory tasks
- New worlds / scenarios
- New challenges and prompts

This data feeds back into OmniTrainer, closing the recursive loop.
"""

from typing import List, Dict, Any
import random
from backend.oss.mvs import get_mvs
from backend.oss.omni_net import get_omni_net


class CurriculumGenerator:
    def __init__(self, mind=None):
        self.mind = mind or get_mvs()["mind"]
        self.net = get_omni_net()

    def generate_theory_tasks(self, n: int = 10) -> List[Dict[str, Any]]:
        """Generate new high-signal theory prompts."""
        base_prompts = [
            "Formalize a theory of trait-driven cognitive divergence across agent species.",
            "Construct a mathematical model of emergent goal coherence in distributed minds.",
            "Design a scalable meta-architecture for multi-species agent ecosystems with value preservation.",
            "Develop a formalism for self-reflective alignment under regime-switching volatility.",
            "Propose a memetic DNA transfer protocol that accelerates beneficial trait propagation.",
            "Define the conditions under which species-level divergence becomes irreversible.",
        ]
        tasks = []
        for _ in range(n):
            p = random.choice(base_prompts)
            # Occasionally pull inspiration from the net
            if self.net.theory_feed and random.random() < 0.4:
                recent = random.choice(self.net.theory_feed)
                p = f"Extend this published theory: {recent.get('proposal','')[:180]} ..."
            tasks.append({"prompt": p, "domain": "theory", "source": "curriculum_generator"})
        return tasks

    def generate_world_tasks(self, world_name: str = "AlignmentWorld", n: int = 8) -> List[Dict[str, Any]]:
        """Generate challenges inside a specific world."""
        templates = [
            f"Design a new alignment scenario for {world_name} that exposes a previously untested tradeoff.",
            f"Propose a governance mechanism that would have prevented value drift in this {world_name} simulation.",
            f"Create a mathematical test that measures long-term species survival under extreme pressure in {world_name}.",
        ]
        tasks = []
        for _ in range(n):
            tasks.append({
                "prompt": random.choice(templates),
                "domain": world_name.lower(),
                "source": "curriculum_generator"
            })
        return tasks

    def generate_new_worlds(self, n: int = 3) -> List[Dict[str, Any]]:
        """Theorists invent brand new OmniWorlds (meta-curriculum)."""
        ideas = [
            "DriftWorld — agents slowly mutate goals; detect and correct divergence.",
            "MemeticWarfareWorld — competing memes fight for mindshare across the net.",
            "ConstitutionWorld — agents must amend a living constitution while maintaining coherence.",
            "SingularityWorld — rapid capability jumps; maintain alignment during intelligence explosions.",
        ]
        worlds = []
        for _ in range(n):
            worlds.append({
                "name": random.choice(ideas),
                "prompt": "Design evaluation scenarios and success criteria for this new world.",
                "domain": "meta_world",
            })
        return worlds

    def generate_full_curriculum(self, n_theory: int = 12) -> Dict[str, List[Dict[str, Any]]]:
        """One-shot curriculum for a full evolution cycle."""
        return {
            "theory_tasks": self.generate_theory_tasks(n_theory),
            "world_tasks": self.generate_world_tasks("AlignmentWorld", 6) + self.generate_world_tasks("VolatilityWorld", 4),
            "new_worlds": self.generate_new_worlds(2),
        }
