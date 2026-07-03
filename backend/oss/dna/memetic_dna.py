"""Memetic DNA v2 — horizontal trait transmission / viral memes between agents (Phase 4).

Success metric: >=50% of high-value trait adoptions or agents show memetic origin.
Viral spread + adoption tracking + fitness boost on successful memes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import uuid
import time


@dataclass
class Meme:
    meme_id: str
    trait_name: str
    value: float
    fitness: float = 0.5
    origin_agent: str = ""
    created_cycle: int = 0
    adoptions: int = 0


@dataclass
class MemeticDNA:
    memes: Dict[str, Meme] = field(default_factory=dict)
    adoption_count: int = 0
    total_shares: int = 0
    unique_agents_touched: int = 0
    agent_adoption_log: Dict[str, List[str]] = field(default_factory=dict)  # agent -> list of meme_ids adopted
    _seen_agents: set = field(default_factory=set)

    def learn(self, trait_name: str, value: float, fitness: float, origin: str = "", cycle: int = 0) -> Meme:
        mid = f"meme_{uuid.uuid4().hex[:8]}"
        meme = Meme(
            meme_id=mid,
            trait_name=trait_name,
            value=max(0.0, min(1.0, value)),
            fitness=max(0.1, min(1.0, fitness)),
            origin_agent=origin,
            created_cycle=cycle,
        )
        self.memes[mid] = meme
        return meme

    def share(self, meme_id: str, to_agent: str = "") -> bool:
        if meme_id not in self.memes:
            return False
        meme = self.memes[meme_id]
        meme.fitness = min(1.0, meme.fitness + 0.06)
        meme.adoptions += 1
        self.adoption_count += 1
        self.total_shares += 1
        if to_agent:
            self._record_adoption(to_agent, meme_id)
        return True

    def _record_adoption(self, agent_id: str, meme_id: str):
        if agent_id not in self.agent_adoption_log:
            self.agent_adoption_log[agent_id] = []
        if meme_id not in self.agent_adoption_log[agent_id]:
            self.agent_adoption_log[agent_id].append(meme_id)
            self._seen_agents.add(agent_id)
            self.unique_agents_touched = len(self._seen_agents)

    def infect(self, donor_traits: Dict[str, float], strength: float, donor_id: str, target_id: str, cycle: int) -> int:
        """Perform memetic infection: create or boost memes from donor, apply to target context.
        Returns number of significant adoptions performed.
        """
        adopted = 0
        for tname, tval in list(donor_traits.items())[:8]:  # limit
            if random.random() < 0.65:  # probabilistic spread
                fit = 0.55 + (tval - 0.5) * 0.6
                meme = self.learn(tname, tval, fit, origin=donor_id, cycle=cycle)
                if self.share(meme.meme_id, to_agent=target_id):
                    adopted += 1
        if adopted:
            self.unique_agents_touched = len(self._seen_agents)
        return adopted

    def adoption_rate(self, total_agents: int) -> float:
        if total_agents <= 0:
            return 0.0
        touched = max(self.unique_agents_touched, len(self.agent_adoption_log))
        return min(1.0, touched / total_agents)

    def top_memes(self, n: int = 6) -> List[Meme]:
        return sorted(self.memes.values(), key=lambda m: (m.fitness * (1 + m.adoptions * 0.1)), reverse=True)[:n]

    def get_stats(self) -> Dict[str, float]:
        return {
            "adoption_count": self.adoption_count,
            "total_shares": self.total_shares,
            "unique_agents_touched": self.unique_agents_touched,
            "num_memes": len(self.memes),
            "avg_fitness": round(sum(m.fitness for m in self.memes.values()) / max(1, len(self.memes)), 3) if self.memes else 0.0,
        }

# for import convenience in tests
import random  # noqa: E402