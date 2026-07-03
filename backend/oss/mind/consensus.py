"""
Weighted Consensus Engine — Phase 3.1

Trait-weighted votes for high-stakes Theory Lab decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid


@dataclass
class Vote:
    agent_id: str
    value: float
    confidence: float
    traits: Dict[str, float] = field(default_factory=dict)
    vote_id: str = field(default_factory=lambda: f"vote_{uuid.uuid4().hex[:8]}")


class WeightedConsensusEngine:
    """Aggregates agent votes using trait-weighted confidence."""

    TRAIT_WEIGHTS = {
        "math": 0.25,
        "pattern_detection": 0.25,
        "self_reflection": 0.20,
        "rigor": 0.15,
        "novelty_seeking": 0.15,
    }

    def __init__(self) -> None:
        self._votes: Dict[str, List[Vote]] = {}
        self._disputes: List[Dict[str, Any]] = []

    def cast_vote(self, topic: str, agent_id: str, value: float, confidence: float, traits: Dict[str, float]) -> Vote:
        vote = Vote(agent_id=agent_id, value=value, confidence=confidence, traits=traits)
        self._votes.setdefault(topic, []).append(vote)
        return vote

    def _trait_weight(self, traits: Dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for name, w in self.TRAIT_WEIGHTS.items():
            t = traits.get(name, traits.get(name.replace("_", ""), 0.5))
            total += float(t) * w
            weight_sum += w
        return total / weight_sum if weight_sum else 0.5

    def get_consensus(self, topic: str, threshold: float = 0.7) -> Tuple[float, bool, Dict[str, Any]]:
        votes = self._votes.get(topic, [])
        if not votes:
            return 0.5, False, {"reason": "no_votes", "count": 0}

        weighted_sum = 0.0
        weight_total = 0.0
        for v in votes:
            tw = self._trait_weight(v.traits)
            w = max(0.01, v.confidence * tw)
            weighted_sum += v.value * w
            weight_total += w

        consensus = weighted_sum / weight_total if weight_total else 0.5
        agreement = sum(1 for v in votes if abs(v.value - consensus) < 0.15) / len(votes)
        reached = agreement >= threshold
        return consensus, reached, {
            "count": len(votes),
            "agreement": round(agreement, 4),
            "threshold": threshold,
        }

    def resolve_dispute(self, topic: str) -> Optional[float]:
        votes = self._votes.get(topic, [])
        if len(votes) < 2:
            return None
        values = [v.value for v in votes]
        spread = max(values) - min(values)
        if spread < 0.3:
            return None
        ranked = sorted(votes, key=lambda v: self._trait_weight(v.traits) * v.confidence, reverse=True)
        resolution = ranked[0].value
        self._disputes.append({"topic": topic, "resolution": resolution, "spread": spread})
        return resolution

    def clear_topic(self, topic: str) -> None:
        self._votes.pop(topic, None)