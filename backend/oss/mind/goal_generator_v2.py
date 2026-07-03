"""
Goal Generator v2 — Phase 3.3

Internal, external, and world-driven emergent goals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import hashlib
import random
import uuid


@dataclass
class GeneratedGoal:
    goal_id: str
    description: str
    priority: float
    source: str
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class GoalGeneratorV2:
    """Produces deduplicated, prioritized goals from mind state and world pressure."""

    INTERNAL_TEMPLATES = [
        "Improve regime-switching model accuracy on VolatilityWorld challenges",
        "Cross-validate alignment safety scores with volatility fitness",
        "Discover canonical memory patterns for theorist lineage",
        "Optimize swarm consensus latency on high-stakes decisions",
        "Increase meta-export schema completeness per cycle",
    ]

    EXTERNAL_STUBS = [
        "Survey arXiv for stochastic volatility regime papers",
        "Extract PubMed cognitive bias patterns for persuasion agents",
        "Monitor DeFi yield curves for survival-mandate opportunities",
    ]

    def __init__(self, mind: Any = None) -> None:
        self.mind = mind
        self._seen: Set[str] = set()
        self._goals: List[GeneratedGoal] = []

    def _dedupe_key(self, description: str) -> str:
        return hashlib.sha256(description.lower().strip().encode()).hexdigest()[:16]

    def _add_goal(self, description: str, priority: float, source: str, domain: str = "general") -> Optional[GeneratedGoal]:
        key = self._dedupe_key(description)
        if key in self._seen:
            return None
        self._seen.add(key)
        goal = GeneratedGoal(
            goal_id=f"goal_{uuid.uuid4().hex[:8]}",
            description=description,
            priority=priority,
            source=source,
            domain=domain,
        )
        self._goals.append(goal)
        return goal

    def inject_volatility_challenge(self, series_id: str, difficulty: float) -> GeneratedGoal:
        desc = f"Build volatility model for series {series_id} (difficulty={difficulty:.2f})"
        g = self._add_goal(desc, priority=0.7 + difficulty * 0.2, source="volatility_world", domain="volatility")
        return g or GeneratedGoal(goal_id="dup", description=desc, priority=0.7, source="volatility_world")

    def generate_goals(self, limit: int = 10) -> List[Dict[str, Any]]:
        batch: List[GeneratedGoal] = []

        for tmpl in random.sample(self.INTERNAL_TEMPLATES, min(5, len(self.INTERNAL_TEMPLATES))):
            g = self._add_goal(tmpl, priority=random.uniform(0.5, 0.9), source="internal")
            if g:
                batch.append(g)

        for stub in random.sample(self.EXTERNAL_STUBS, min(3, len(self.EXTERNAL_STUBS))):
            g = self._add_goal(stub, priority=random.uniform(0.4, 0.7), source="external_stub")
            if g:
                batch.append(g)

        if self.mind and hasattr(self.mind, "shared_memory"):
            for key in list(self.mind.shared_memory.keys())[:3]:
                g = self._add_goal(f"Resolve shared memory gap: {key}", 0.6, source="mind_sync", domain="cognitive")
                if g:
                    batch.append(g)

        while len(batch) < limit:
            g = self._add_goal(
                f"Novel research thread #{len(batch)} — cross-domain theory synthesis",
                random.uniform(0.45, 0.85),
                source="synthetic",
            )
            if not g:
                break
            batch.append(g)

        batch.sort(key=lambda x: x.priority, reverse=True)
        return [
            {"goal_id": g.goal_id, "description": g.description, "priority": g.priority, "source": g.source, "domain": g.domain}
            for g in batch[:limit]
        ]