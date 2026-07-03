"""
Omni-Sentient Agents — genome + novelty + loyalty + evolution integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import random
import uuid
from typing import Any, Dict, List, Optional

from .hyper_elite import create_hyper_elite_psychology_genome
from .schema import Genome
from .novelty import NoveltyRewardEngine
from .loyalty import LoyaltyLevel, LoyaltySystem
from .evolution_tuner import EvolutionTuner


@dataclass
class Agent:
    agent_id: str = field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    genome: Genome = field(default_factory=create_hyper_elite_psychology_genome)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    fitness_history: List[float] = field(default_factory=list)
    task_count: int = 0
    role: str = "hyper_elite_psychologist"

    def mean_fitness(self) -> float:
        if not self.fitness_history:
            return 0.5
        return sum(self.fitness_history[-20:]) / min(len(self.fitness_history), 20)

    def fitness_ema(self, alpha: float = 0.2) -> float:
        ema = 0.5
        for f in self.fitness_history:
            ema = alpha * f + (1 - alpha) * ema
        return ema

    def record_fitness(self, fitness: float) -> None:
        self.fitness_history.append(round(fitness, 4))
        self.task_count += 1
        if len(self.fitness_history) > 500:
            self.fitness_history = self.fitness_history[-250:]

    def profile(self, loyalty_system: Optional[LoyaltySystem] = None) -> dict:
        level = loyalty_system.get_level(self.agent_id) if loyalty_system else LoyaltyLevel.NOVICE
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "task_count": self.task_count,
            "mean_fitness": round(self.mean_fitness(), 4),
            "fitness_ema": round(self.fitness_ema(), 4),
            "generation": self.genome.generation,
            "loyalty_level": level.name,
        }

    def act(self, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        conditioned = self._condition_task(task)
        output = self._generate_output(conditioned)
        metrics = self._evaluate_output(output, task)
        fitness = self.genome.calculate_fitness(metrics)

        self.fitness_history.append(fitness)
        if len(self.fitness_history) > 100:
            self.fitness_history.pop(0)

        novelty_ctx = {"novelty_reward": metrics.get("novelty", 0.5), "performance": fitness, "fitness_history": self.fitness_history}
        self.genome.mutate(novelty_ctx)
        self.genome.apply_interactions()

        self.memory.append({
            "task": task,
            "output": output,
            "metrics": metrics,
            "fitness": fitness,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self.memory) > 200:
            self.memory.pop(0)

        return {
            "agent_id": self.agent_id,
            "task": task,
            "output": output,
            "metrics": metrics,
            "fitness": fitness,
            "raw_output": output.get("content", str(output)),
        }

    def _condition_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        conditioned = dict(task)
        depth = self.genome.get_value("cognitive", "M_REASONING_DEPTH")
        conditioned["difficulty"] = min(1.0, task.get("difficulty", 0.5) * (1.2 if depth > 0.8 else 0.85))
        curiosity = self.genome.get_value("psychology", "M_CURIOSITY_TRIGGER")
        if curiosity > 0.75:
            conditioned["prompt"] = f"[CURIOSITY={curiosity:.2f}] {task.get('prompt', '')}"
        return conditioned

    def _generate_output(self, task: Dict[str, Any]) -> Dict[str, Any]:
        ttype = task.get("type", "default")
        if ttype == "persuasion":
            trust = self.genome.get_value("psychology", "M_TRUST_TONE")
            emotion = self.genome.get_value("psychology", "M_EMOTIONAL_CHARGE")
            scarcity = self.genome.get_value("psychology", "M_SCARCITY_SIGNAL")
            content = task.get("prompt", "")
            if trust > 0.75:
                content += " [TRUST: authority + social proof]"
            if emotion > 0.7:
                content += " [EMOTION: high charge]"
            if scarcity > 0.6:
                content += " [SCARCITY: limited availability]"
            return {"type": "persuasion", "content": content, "trust_tone": trust, "emotional_charge": emotion, "scarcity_signal": scarcity}
        if ttype == "analysis":
            depth = self.genome.get_value("cognitive", "M_REASONING_DEPTH")
            pattern = self.genome.get_value("cognitive", "M_PATTERN_RECOGNITION")
            return {"type": "analysis", "content": f"{task.get('prompt', '')} [depth={depth:.2f}]", "reasoning_depth": depth, "pattern_recognition": pattern}
        if ttype == "trade":
            risk = self.genome.get_value("market", "M_RISK_TOLERANCE")
            trend = self.genome.get_value("market", "M_TREND_DETECTION")
            return {"type": "trade", "action": "buy" if trend > 0.7 else "hold", "confidence": (trend + risk) / 2}
        return {"type": "default", "content": task.get("prompt", "")}

    def _evaluate_output(self, output: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"task_success": 0.5, "novelty": 0.5, "impact": 0.5, "engagement": 0.5}
        if output.get("type") == "persuasion":
            metrics["click_through_rate"] = min(1.0, output.get("trust_tone", 0.5) * 1.1)
            metrics["engagement"] = min(1.0, output.get("emotional_charge", 0.5) * 1.1)
            metrics["conversion_rate"] = min(1.0, output.get("scarcity_signal", 0.5) * 1.2)
            metrics["emotions"] = {"joy": output.get("emotional_charge", 0.5), "trust": output.get("trust_tone", 0.5)}
        elif output.get("type") == "analysis":
            metrics["task_success"] = min(1.0, output.get("reasoning_depth", 0.5) * 1.2)
            metrics["novelty"] = min(1.0, output.get("pattern_recognition", 0.5) * 1.1)
        elif output.get("type") == "trade":
            metrics["conversion_rate"] = output.get("confidence", 0.5)
            metrics["revenue"] = output.get("confidence", 0.5) * 50
        return metrics


@dataclass
class AgentSwarm:
    agents: Dict[str, Agent] = field(default_factory=dict)
    loyalty_system: LoyaltySystem = field(default_factory=LoyaltySystem)
    novelty_engine: NoveltyRewardEngine = field(default_factory=NoveltyRewardEngine)
    evolution_tuner: EvolutionTuner = field(default_factory=EvolutionTuner)
    global_goals: Dict[str, float] = field(default_factory=lambda: {"task_success": 0.5, "novelty": 0.3, "engagement": 0.2})

    def __post_init__(self):
        self.loyalty_system.global_goals = self.global_goals

    def add_agent(self, agent: Optional[Agent] = None) -> str:
        a = agent or Agent()
        self.agents[a.agent_id] = a
        self.evolution_tuner.tune_creativity(a.genome)
        self.evolution_tuner.tune_intelligence(a.genome)
        return a.agent_id

    def register(self, agent: Agent) -> None:
        self.add_agent(agent)

    def assign_task_to_best_agent(self, task: Dict[str, Any]) -> str:
        """Return the agent_id whose genome desire locus best matches this task."""
        from .desire_genome import score_desire_alignment
        best_id, best_score = None, -1.0
        for aid, agent in self.agents.items():
            alignment = score_desire_alignment(task, agent.genome)
            if alignment > best_score:
                best_score, best_id = alignment, aid
        return best_id or random.choice(list(self.agents.keys()))

    def form_desire_complementary_swarm(self, task: Dict[str, Any], size: int = 3) -> List[str]:
        """Select agents with complementary desire profiles for a given task."""
        from .desire_genome import dominant_desire, score_desire_alignment
        candidates = sorted(
            self.agents.items(),
            key=lambda kv: score_desire_alignment(task, kv[1].genome),
            reverse=True,
        )
        chosen, seen = [], set()
        for aid, agent in candidates:
            dom = dominant_desire(agent.genome)
            if dom not in seen:
                chosen.append(aid)
                seen.add(dom)
                if len(chosen) >= size:
                    break
        return chosen

    def assign_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if not self.agents:
            self.add_agent()
        # Desire-aligned routing: prefer the agent whose drives match this task
        if len(self.agents) > 1:
            agent_id = self.assign_task_to_best_agent(task)
        else:
            agent_id = next(iter(self.agents))
        agent = self.agents[agent_id]
        result = agent.act(task)
        # Record desire alignment so callers can pass it to calculate_fitness
        from .desire_genome import score_desire_alignment
        alignment = round(score_desire_alignment(task, agent.genome), 4)
        result["metrics"]["desire_alignment"] = alignment
        result["desire_alignment"] = alignment
        novelty = self.novelty_engine.compute_reward(agent_id, result, result["metrics"])
        result["metrics"]["novelty"] = novelty.weighted
        result["novelty_score"] = novelty.to_dict()

        self.loyalty_system.update_agent(
            agent_id,
            agent.fitness_history,
            {k: result["metrics"].get(k, 0.0) for k in self.global_goals},
        )
        return result

    def evolve_swarm(self, top_fraction: float = 0.2) -> int:
        if len(self.agents) < 2:
            return 0
        ranked = sorted(self.agents.values(), key=lambda a: a.fitness_history[-1] if a.fitness_history else 0.0, reverse=True)
        n = max(1, int(len(ranked) * top_fraction))
        elites = ranked[:n]
        spawned = 0
        for _ in range(n):
            if len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
            else:
                p1 = p2 = elites[0]
            c1, c2 = p1.genome.crossover(p2.genome)
            child = Agent(genome=c1, role=p1.role)
            self.evolution_tuner.tune_creativity(child.genome)
            self.add_agent(child)
            spawned += 1
        return spawned

    def get_swarm_metrics(self) -> Dict[str, Any]:
        fitnesses = [a.fitness_history[-1] for a in self.agents.values() if a.fitness_history]
        return {
            "agents": len(self.agents),
            "avg_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "loyalty_levels": {aid: self.loyalty_system.get_level(aid).name for aid in self.agents},
        }