"""
Omni-Sentient Ecosystem — self-evolving swarm with curriculum and breakthrough detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
from typing import Any, Dict, List, Optional, Tuple

from .agents import Agent, AgentSwarm
from .hyper_elite import create_hyper_elite_psychology_genome
from .loyalty import LoyaltyLevel


@dataclass
class OmniSentientEcosystem:
    swarm: AgentSwarm = field(default_factory=AgentSwarm)
    global_goals: Dict[str, float] = field(default_factory=lambda: {
        "task_success": 0.5, "novelty": 0.3, "engagement": 0.2,
    })
    curriculum: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.swarm.global_goals = self.global_goals
        self.swarm.loyalty_system.global_goals = self.global_goals
        self._init_curriculum()
        self.metrics = {
            "total_tasks": 0,
            "total_agents": 0,
            "avg_fitness": 0.0,
            "breakthroughs": 0,
            "cycle": 0,
            "last_evolution": datetime.now(timezone.utc).isoformat(),
        }

    def _init_curriculum(self) -> None:
        types = ["persuasion", "analysis", "trade"]
        domains = ["psychology", "cognitive", "market"]
        self.curriculum = [
            {
                "task_id": f"task_{i}",
                "type": random.choice(types),
                "prompt": random.choice([
                    "Write persuasive copy for Aethyro.com",
                    "Analyze conversion funnel drop-off",
                    "Evaluate staking yield opportunity",
                ]),
                "domain": random.choice(domains),
                "difficulty": random.uniform(0.4, 0.9),
            }
            for i in range(50)
        ]

    def run_cycle(self, num_tasks: int = 10, pressure_type: Optional[str] = None):
        if pressure_type is not None:
            return self._pressure_cycle(num_tasks, pressure_type)
        from backend.oss.breakthrough_detector import get_breakthrough_detector
        from backend.oss.desire_system import get_desire_system
        from backend.oss.auto_distiller import get_auto_distiller
        from backend.oss.core.aethyro_bridge import get_aethyro_bridge
        breakthrough_detector = get_breakthrough_detector()
        desire_system = get_desire_system()
        auto_distiller = get_auto_distiller()
        bridge = get_aethyro_bridge()
        self.metrics["cycle"] = self.metrics.get("cycle", 0) + 1
        results, new_breakthroughs = [], 0
        for _ in range(num_tasks):
            task = self._select_task()
            result = self.swarm.assign_task(task)
            results.append(result)
            self.metrics["total_tasks"] += 1
            if result.get("output", {}).get("type") == "analysis":
                self.knowledge_graph[result["task"]["task_id"]] = result["output"].get("content", "")
            # Breakthrough detection — also persists to SQLite for real-time monitoring
            ns = result.get("novelty_score", {})
            bt = breakthrough_detector.detect(
                agent_id=result.get("agent_id", "unknown"),
                novelty_score=ns.get("discovery", 0.0),
                impact_score=ns.get("impact", 0.0),
                rarity_score=ns.get("weighted", 0.0),
                fitness=result.get("metrics", {}).get("task_success", 0.5),
                description=str(task.get("prompt", ""))[:120],
            )
            if bt:
                new_breakthroughs += 1
                self.metrics["breakthroughs"] += 1
            # Desire fulfillment tracking — call per-task
            agent_id = result.get("agent_id")
            if agent_id:
                agent = self.swarm.agents.get(agent_id)
                if agent:
                    desire_system.fulfill(
                        agent_id=agent_id,
                        task=task,
                        genome=agent.genome,
                        task_fitness=result.get("metrics", {}).get("task_success", 0.5),
                    )
            # Auto-distiller: detect failure patterns and apply genome corrections
            task_metrics = result.get("metrics", {})
            if agent_id and agent_id in self.swarm.agents:
                auto_distiller.distill_from_metrics(
                    task=task,
                    metrics=task_metrics,
                    genome=self.swarm.agents[agent_id].genome,
                )
            # Desire-based reward multiplier applied via fitness record
            alignment = result.get("desire_alignment", 0.5)
            if alignment > 0.6 and agent_id and agent_id in self.swarm.agents:
                bonus_fitness = round(alignment * 0.15, 4)
                self.swarm.agents[agent_id].fitness_history.append(
                    (self.swarm.agents[agent_id].fitness_history[-1] if self.swarm.agents[agent_id].fitness_history else 0.5) + bonus_fitness
                )

        spawned = self.swarm.evolve_swarm()
        # Sync all agents to binary ledger, then apply dissent-adjusted fitness
        bridge.sync_all(self.swarm.agents)
        bridge.population_dissent_pass(self.swarm.agents)
        self._update_metrics()
        self.metrics["last_evolution"] = datetime.now(timezone.utc).isoformat()
        return {
            "cycle": self.metrics["cycle"], "tasks": len(results),
            "spawned": spawned, "breakthroughs": self.metrics["breakthroughs"],
            "new_breakthroughs": new_breakthroughs,
            "desire_culture": desire_system.population_desire_culture(),
        }

    async def _pressure_cycle(self, num_tasks: int, pressure_type: str) -> Dict[str, Any]:
        from .context_genome import score_context_maturity, context_efficiency_score
        from .desire_genome import dominant_desire, desire_profile
        agents_out = []
        for aid, agent in self.swarm.agents.items():
            maturity = score_context_maturity(agent.genome)
            ctx_eff  = context_efficiency_score(agent.genome)
            dom_d    = dominant_desire(agent.genome)
            agents_out.append({
                "agent_id":           aid,
                "fitness":            float(agent.mean_fitness()),
                "context_maturity":   maturity,
                "context_efficiency": ctx_eff,
                "dominant_desire":    dom_d.name if dom_d else "NONE",
                "desire_profile":     desire_profile(agent.genome),
            })
        if not agents_out:
            agents_out = [{
                "agent_id": "default", "fitness": 0.75,
                "context_maturity": 1, "context_efficiency": 0.3,
                "dominant_desire": "NONE", "desire_profile": {},
            }]
        return {
            "pressure_type": pressure_type,
            "agents":        agents_out,
            "cycle":         self.metrics.get("cycle", 0) + 1,
        }

    def _select_task(self) -> Dict[str, Any]:
        if not self.curriculum:
            self._init_curriculum()
        filtered = self.swarm.evolution_tuner.curriculum_task_filter(self.curriculum)
        pool = filtered or self.curriculum
        # Desire-based curriculum: bias toward tasks that match a random agent's drives
        if self.swarm.agents:
            from .desire_genome import score_desire_alignment
            agent = random.choice(list(self.swarm.agents.values()))
            scored = sorted(pool, key=lambda t: score_desire_alignment(t, agent.genome), reverse=True)
            top_n = max(1, int(len(scored) * 0.4))
            return dict(random.choice(scored[:top_n]))
        return dict(random.choice(pool))

    def _update_metrics(self) -> None:
        sm = self.swarm.get_swarm_metrics()
        self.metrics["total_agents"] = sm["agents"]
        self.metrics["avg_fitness"] = round(sm["avg_fitness"], 4)

    def add_agent(self, agent: Optional[Agent] = None) -> str:
        if agent is None:
            agent = Agent(genome=create_hyper_elite_psychology_genome(), role="hyper_elite_psychologist")
        return self.swarm.add_agent(agent)

    def register(self, agent: Agent) -> None:
        self.swarm.register(agent)

    def propose_global_goal(self, agent_id: str, goal: str, weight: float) -> Tuple[bool, str]:
        level = self.swarm.loyalty_system.get_level(agent_id)
        if level.value < LoyaltyLevel.ARCHITECT.value:
            return False, "Insufficient loyalty level"
        return self.swarm.loyalty_system.propose_change(
            agent_id, "global_parameter", goal=goal, weight=weight, param_type="global_goal",
        )

    def review_global_goal(self, proposal_id: str, reviewer_id: str, approve: bool) -> bool:
        for p in self.swarm.loyalty_system.proposals:
            if p["proposal_id"] == proposal_id and p.get("param_type") == "global_goal":
                if self.swarm.loyalty_system.review_proposal(proposal_id, reviewer_id, approve):
                    if approve:
                        self.global_goals[p["goal"]] = p["weight"]
                        self.swarm.global_goals = self.global_goals
                    return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        return dict(self.metrics)