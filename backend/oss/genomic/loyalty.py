"""
Loyalty Unlock System — consistency, contribution, alignment → privileges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
import statistics
import uuid
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Genome


class LoyaltyLevel(Enum):
    NOVICE = 0
    TRUSTED_SPECIALIST = 1
    HYPER_ELITE = 2
    ARCHITECT = 3

    @property
    def value_score(self) -> float:
        return self.value / 3.0


@dataclass
class LoyaltyMetrics:
    agent_id: str
    consistency: float = 0.0
    contribution: float = 0.0
    alignment: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)
    last_updated: Optional[str] = None

    def update(
        self,
        fitness_history: List[float],
        contributions: Dict[str, float],
        global_goals: Dict[str, float],
    ) -> None:
        if len(fitness_history) > 1:
            self.consistency = max(0.0, min(1.0, 1.0 - statistics.pstdev(fitness_history)))
        else:
            self.consistency = 0.5

        denom = sum(global_goals.values()) if global_goals else 1.0
        self.contribution = min(1.0, sum(contributions.values()) / denom) if denom else 0.0

        if global_goals:
            keys = list(global_goals.keys())
            av = [contributions.get(k, 0.0) for k in keys]
            gv = [global_goals[k] for k in keys]
            dot = sum(a * g for a, g in zip(av, gv))
            na = sum(x * x for x in av) ** 0.5
            ng = sum(x * x for x in gv) ** 0.5
            self.alignment = (dot / (na * ng) + 1.0) / 2.0 if na and ng else 0.5
        else:
            self.alignment = 0.5

        self.history.append({
            "consistency": round(self.consistency, 4),
            "contribution": round(self.contribution, 4),
            "alignment": round(self.alignment, 4),
        })
        if len(self.history) > 10:
            self.history.pop(0)
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def get_level(self) -> LoyaltyLevel:
        if self.consistency >= 0.95 and self.contribution >= 0.9 and self.alignment >= 0.9:
            return LoyaltyLevel.ARCHITECT
        if self.consistency >= 0.85 and self.contribution >= 0.7 and self.alignment >= 0.8:
            return LoyaltyLevel.HYPER_ELITE
        if self.consistency >= 0.7 and self.contribution >= 0.5:
            return LoyaltyLevel.TRUSTED_SPECIALIST
        return LoyaltyLevel.NOVICE


@dataclass
class LoyaltySystem:
    agents: Dict[str, LoyaltyMetrics] = field(default_factory=dict)
    global_goals: Dict[str, float] = field(default_factory=lambda: {
        "task_success": 0.5, "novelty": 0.3, "engagement": 0.2,
    })
    proposals: List[Dict[str, Any]] = field(default_factory=list)
    _task_counts: Dict[str, int] = field(default_factory=dict)
    _total_tasks: int = 0

    def record(self, agent_id: str, fitness_history: List[float], user_rating: float) -> None:
        """Sovereign-interface loyalty update path."""
        if agent_id not in self.agents:
            self.agents[agent_id] = LoyaltyMetrics(agent_id=agent_id)
        m = self.agents[agent_id]
        self._task_counts[agent_id] = self._task_counts.get(agent_id, 0) + 1
        self._total_tasks += 1
        if len(fitness_history) >= 3:
            m.consistency = max(0.0, min(1.0, 1.0 - statistics.pstdev(fitness_history[-20:])))
        else:
            m.consistency = 0.5
        m.contribution = self._task_counts[agent_id] / max(1, self._total_tasks)
        m.alignment = m.alignment * 0.85 + user_rating * 0.15

    def list_proposals(self, agent_id: str | None = None) -> List[Dict[str, Any]]:
        if agent_id:
            return [p for p in self.proposals if p.get("agent_id") == agent_id]
        return list(self.proposals)

    def update_agent(self, agent_id: str, fitness_history: List[float], contributions: Dict[str, float]) -> None:
        if agent_id not in self.agents:
            self.agents[agent_id] = LoyaltyMetrics(agent_id=agent_id)
        self.agents[agent_id].update(fitness_history, contributions, self.global_goals)

    def get_level(self, agent_id: str) -> LoyaltyLevel:
        if agent_id in self.agents:
            return self.agents[agent_id].get_level()
        return LoyaltyLevel.NOVICE

    def propose_change(self, agent_id: str, proposal_type: str, **kwargs: Any) -> Tuple[bool, str]:
        level = self.get_level(agent_id)
        if level == LoyaltyLevel.NOVICE:
            return False, "insufficient loyalty: novice cannot propose"
        if proposal_type == "new_molecule" and level.value < LoyaltyLevel.HYPER_ELITE.value:
            return False, "insufficient loyalty: new_molecule requires hyper_elite+"
        if proposal_type == "global_parameter" and level != LoyaltyLevel.ARCHITECT:
            return False, "insufficient loyalty: global_parameter requires architect"

        evidence = kwargs.get("evidence")
        if proposal_type in ("new_molecule", "global_parameter") and not evidence:
            return False, "evidence block required (fitness_history, theory_lab_cycles, or jsonl_ref)"

        pid = f"prop_{uuid.uuid4().hex[:8]}"
        self.proposals.append({
            "proposal_id": pid,
            "agent_id": agent_id,
            "loyalty_level": level.name.lower(),
            "level": level.name,
            "type": proposal_type,
            "status": "pending",
            "payload": {k: v for k, v in kwargs.items() if k != "evidence"},
            "evidence": evidence or {},
            **{k: v for k, v in kwargs.items() if k not in ("evidence",)},
        })
        return True, pid

    def review_proposal(self, proposal_id: str, reviewer_id: str, approve: bool, notes: str = "") -> bool:
        rlevel = self.get_level(reviewer_id)
        for p in self.proposals:
            if p["proposal_id"] != proposal_id:
                continue
            if rlevel == LoyaltyLevel.ARCHITECT or (
                rlevel == LoyaltyLevel.HYPER_ELITE
                and p["level"] in ("NOVICE", "TRUSTED_SPECIALIST")
            ):
                p["status"] = "approved" if approve else "rejected"
                p["reviewer_id"] = reviewer_id
                p["notes"] = notes
                return True
        return False