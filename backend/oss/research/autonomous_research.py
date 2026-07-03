"""
Autonomous Research Engine — Phase 7.2

propose_project, evaluate_project, execute_project.
Approval threshold >0.7.
50+ projects/month sim.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import random
import time
import uuid


@dataclass
class ResearchProject:
    project_id: str
    title: str
    hypothesis: str
    domain: str
    approval_score: float = 0.0
    status: str = "proposed"
    results: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class AutonomousResearchEngine:
    """Self-directed research for singularity."""

    DOMAINS = ["volatility", "alignment", "meta_architecture", "species", "economy"]

    def __init__(self, economy: Any = None):
        self.projects: List[ResearchProject] = []
        self.economy = economy
        self.completed = 0

    def propose_project(self, context: Dict[str, Any] = None) -> ResearchProject:
        domain = random.choice(self.DOMAINS)
        p = ResearchProject(
            project_id=f"proj_{uuid.uuid4().hex[:8]}",
            title=f"Autonomous study of {domain} dynamics",
            hypothesis=f"Higher {domain} pressure leads to emergent {random.choice(['resilience', 'divergence', 'alignment'])}",
            domain=domain,
            approval_score=round(0.6 + random.random() * 0.4, 3),
        )
        self.projects.append(p)
        return p

    def evaluate_project(self, project: ResearchProject, mind_state: Dict[str, Any] = None) -> float:
        """Evaluate with >0.7 approval threshold."""
        score = project.approval_score
        if mind_state:
            # boost based on state
            score = min(0.99, score + 0.1 * random.random())
        project.approval_score = round(score, 3)
        return score

    def execute_project(self, project: ResearchProject) -> Dict[str, Any]:
        if project.approval_score < 0.7:
            project.status = "rejected"
            return {"status": "rejected", "reason": "approval < 0.7"}
        # Simulate execution
        result = {
            "findings": f"Confirmed {project.hypothesis} with 87% confidence",
            "delta": round(random.uniform(0.05, 0.2), 3),
            "artifacts": [f"data/{project.domain}_report.json"]
        }
        project.results = result
        project.status = "completed"
        self.completed += 1
        if self.economy:
            try:
                self.economy.reward("research", 50, reason="autonomous_project")
            except:
                pass
        return result

    def run_month_sim(self, n: int = 60) -> Dict[str, Any]:
        """Simulate 50+ projects/month. Resilient to individual failures (survival-first)."""
        proposed = 0
        approved = 0
        executed = 0
        for i in range(n):
            try:
                p = self.propose_project()
                proposed += 1
                if self.evaluate_project(p) >= 0.7:
                    approved += 1
                    if self.execute_project(p):
                        executed += 1
            except Exception as e:
                # Resilience: one failed project must not stop research flywheel
                print(f"[Research] Project {i} resilient skip: {e}")
        return {
            "proposed": proposed,
            "approved": approved,
            "executed": executed,
            "rate": round(executed / max(1, proposed), 2),
        }


def get_autonomous_research_engine(economy: Any = None) -> AutonomousResearchEngine:
    if not hasattr(get_autonomous_research_engine, "_inst"):
        get_autonomous_research_engine._inst = AutonomousResearchEngine(economy)
    return get_autonomous_research_engine._inst
