"""
Swarm Reasoning Engine — Phase 3.4

Multi-agent task execution with consensus integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from .consensus import WeightedConsensusEngine


@dataclass
class SwarmTask:
    task_id: str
    description: str
    complexity: float = 0.5
    domain: str = "theory"
    assigned_agents: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None


class SwarmReasoningEngine:
    """Coordinates swarms; integrates consensus for final decisions."""

    def __init__(self, mind: Any = None, economy: Any = None, substrate: Any = None) -> None:
        self.mind = mind
        self.economy = economy
        self.substrate = substrate
        self.consensus = WeightedConsensusEngine()
        self._swarms: Dict[str, SwarmTask] = {}
        self._contributions: Dict[str, Dict[str, float]] = {}

    def create_swarm(self, description: str, complexity: float = 0.5, domain: str = "theory") -> str:
        sid = f"swarm_{uuid.uuid4().hex[:8]}"
        self._swarms[sid] = SwarmTask(task_id=sid, description=description, complexity=complexity, domain=domain)
        return sid

    def assign_agents(self, swarm_id: str, agent_ids: List[str]) -> bool:
        task = self._swarms.get(swarm_id)
        if not task:
            return False
        task.assigned_agents = list(agent_ids)
        return True

    def execute_swarm(self, swarm_id: str, agent_outputs: Dict[str, str]) -> Dict[str, Any]:
        task = self._swarms.get(swarm_id)
        if not task:
            return {"error": "swarm_not_found"}

        topic = f"swarm_{swarm_id}"
        scores: Dict[str, float] = {}
        for aid, output in agent_outputs.items():
            quality = min(1.0, len(output) / 800 + 0.2)
            traits = {"math": 0.6, "pattern_detection": 0.7, "self_reflection": 0.5}
            if self.substrate:
                try:
                    agent = self.substrate.spawn_agent(aid)
                    traits = agent.dna.get_traits()
                except Exception:
                    pass
            self.consensus.cast_vote(topic, aid, quality, confidence=0.8, traits=traits)
            scores[aid] = quality

        consensus, reached, meta = self.consensus.get_consensus(topic, threshold=0.7)
        if not reached:
            resolved = self.consensus.resolve_dispute(topic)
            if resolved is not None:
                consensus = resolved

        self._contributions[swarm_id] = scores
        task.result = {"consensus_score": consensus, "reached": reached, "meta": meta, "agent_scores": scores}
        return task.result

    def distribute_rewards(self, swarm_id: str, base_reward: float = 100.0) -> Dict[str, float]:
        contrib = self._contributions.get(swarm_id, {})
        if not contrib or not self.economy:
            return {}
        total = sum(contrib.values()) or 1.0
        payouts = {}
        for aid, score in contrib.items():
            share = (score / total) * base_reward
            try:
                self.economy.reward(aid, share, reason=f"swarm_{swarm_id}")
            except Exception:
                pass
            payouts[aid] = round(share, 2)
        return payouts

    def benchmark_vs_solo(self, solo_score: float, swarm_score: float) -> Dict[str, Any]:
        delta = swarm_score - solo_score
        pct = (delta / solo_score * 100) if solo_score else 0.0
        return {"solo": solo_score, "swarm": swarm_score, "delta_pct": round(pct, 2), "swarm_wins": delta > 0}