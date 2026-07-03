"""
Sovereign genomic stack — async ecosystem + agents for sovereign_interface.

Canonical path (backend/oss/genomic). Legacy backend/genomic/ tree removed after migration.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .agents import Agent, AgentSwarm
from .evolution_tuner import EvolutionTuner
from .loyalty import LoyaltySystem
from .novelty import NoveltyRewardEngine

log = logging.getLogger("oss.genomic.sovereign")


class SovereignEcosystem:
    """
    Async lifecycle controller for sovereign agent genome population.
    Used by sovereign_interface and volatility pressure tests.
    """

    def __init__(self) -> None:
        self.loyalty = LoyaltySystem()
        self.swarm = AgentSwarm(loyalty_system=self.loyalty)
        self.novelty = NoveltyRewardEngine()
        self.tuner = EvolutionTuner()
        self._cycle_log: List[dict] = []
        self._cycle = 0

    def register(self, agent: Agent) -> None:
        self.swarm.register(agent)

    async def run_cycle(
        self,
        pressure_fn: Optional[Callable[[Agent], Any]] = None,
        pressure_type: str = "general",
        tasks: Optional[List[str]] = None,
    ) -> dict:
        self._cycle += 1
        t0 = time.time()
        cycle_results = []
        vol_world = None
        vol_task: Dict[str, Any] = {}
        if pressure_type == "volatility":
            vol_world, vol_task = self._prepare_volatility_challenge()

        agents = list(self.swarm.agents.values())
        if not agents:
            return {"error": "no agents registered", "cycle": self._cycle}

        for i, agent in enumerate(agents):
            if pressure_fn is not None:
                try:
                    simulated_output = await asyncio.coroutine(pressure_fn)(agent) \
                        if asyncio.iscoroutinefunction(pressure_fn) \
                        else pressure_fn(agent)
                except Exception as e:
                    log.warning("pressure_fn failed for %s: %s", agent.agent_id, e)
                    simulated_output = f"[{agent.agent_id}] default pressure output"
            elif tasks:
                simulated_output = tasks[i % len(tasks)]
            else:
                simulated_output = self._default_pressure(agent, pressure_type, vol_task=vol_task)

            metrics = self._score_output(simulated_output, pressure_type, agent=agent, vol_world=vol_world, vol_task=vol_task)
            fitness = agent.genome.calculate_fitness(metrics)
            if agent.fitness_history is not None:
                agent.fitness_history.append(fitness)
            agent.task_count = getattr(agent, "task_count", 0) + 1

            out_payload = {"raw_output": simulated_output, "type": pressure_type}
            novelty = self.novelty.compute_reward(
                agent_id=agent.agent_id,
                output=out_payload,
                metrics=metrics,
                context={"user_rating": metrics["user_rating"]},
            )
            agent.genome.mutate(context={"novelty_reward": novelty.weighted})

            self.loyalty.record(
                agent_id=agent.agent_id,
                fitness_history=agent.fitness_history,
                user_rating=metrics["user_rating"],
            )

            cycle_results.append({
                "agent_id": agent.agent_id,
                "fitness": fitness,
                "novelty": novelty.weighted,
                "loyalty_level": self.loyalty.get_level(agent.agent_id).name,
                "generation": getattr(agent.genome, "generation", 0),
                "snapshot": agent.genome.snapshot() if hasattr(agent.genome, "snapshot") else {},
            })

        evolution_stats = {"mean_fitness": sum(r["fitness"] for r in cycle_results) / len(cycle_results)}

        cycle_summary = {
            "cycle": self._cycle,
            "pressure_type": pressure_type,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "agents": cycle_results,
            "evolution": evolution_stats,
        }
        self._cycle_log.append(cycle_summary)
        if len(self._cycle_log) > 100:
            self._cycle_log = self._cycle_log[-50:]
        return cycle_summary

    def _prepare_volatility_challenge(self) -> tuple[Any, Dict[str, Any]]:
        try:
            from backend.oss.world.volatility_world import VolatilityWorld
            world = VolatilityWorld(length=200, seed=self._cycle + 7)
            data = world.generate_series()
            challenge = world.generate_challenge(data["series_id"])
            task = {
                "world_data": data,
                "challenge": {"series_id": data["series_id"], "difficulty": challenge.difficulty},
                "world": world,
            }
            return world, task
        except Exception as e:
            log.warning("volatility challenge prep failed: %s", e)
            return None, {}

    def _default_pressure(self, agent: Agent, pressure_type: str, vol_task: Optional[Dict[str, Any]] = None) -> str:
        if pressure_type == "volatility" and vol_task and vol_task.get("world_data"):
            data = vol_task["world_data"]
            mean_v = sum(data["series"]) / max(len(data["series"]), 1)
            return (
                f"Agent {agent.agent_id} proposes a regime-switching stochastic volatility model "
                f"for series {data.get('series_id', 'unknown')} with mean={mean_v:.4f}. "
                "The model uses Markov transition dynamics between low/medium/high volatility regimes."
            )
        templates = {
            "psychology": f"Agent {agent.agent_id} demonstrates emotional intelligence framing.",
            "market": f"Agent {agent.agent_id} identifies margin improvement opportunity.",
            "cognitive": f"Agent {agent.agent_id} applies causal inference to churn drivers.",
        }
        return templates.get(pressure_type, f"Agent {agent.agent_id} processes task cycle {self._cycle}.")

    def _score_output(
        self,
        output: str,
        pressure_type: str,
        agent: Optional[Agent] = None,
        vol_world: Any = None,
        vol_task: Optional[Dict[str, Any]] = None,
    ) -> dict:
        import re
        from backend.oss.world.volatility_world import VolatilityModel

        numeric_density = len(re.findall(r"\d+", output)) / max(len(output.split()), 1)
        task_success = min(1.0, 0.4 + numeric_density * 2)
        user_rating = min(1.0, 0.5 + len(output) / 2000)
        coherence = 0.7 if pressure_type in {"psychology", "cognitive"} else 0.6

        if pressure_type == "volatility" and vol_world and vol_task and vol_task.get("world_data") and agent:
            try:
                data = vol_task["world_data"]
                model = VolatilityModel(agent_id=agent.agent_id, description=output[:600], code=output[:1200])
                mid = vol_world.submit_model(model)
                ev = vol_world._evaluate_by_ids(mid, data["series_id"])
                ws = ev.get("weighted_score", 0.5)
                task_success = ws
                user_rating = min(1.0, ws + 0.1)
                coherence = min(1.0, ev.get("metrics", {}).get("regime_awareness", 0.6))
            except Exception as e:
                log.warning("volatility score failed for %s: %s", agent.agent_id, e)

        return {
            "task_success": round(task_success, 4),
            "user_rating": round(user_rating, 4),
            "latency_ms": 100,
            "coherence": coherence,
        }

    def recent_cycles(self, n: int = 10) -> List[dict]:
        return self._cycle_log[-n:]

    def population_snapshot(self) -> dict:
        return {
            "cycle": self._cycle,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "mean_fitness": a.fitness_history[-1] if a.fitness_history else 0.5,
                    "generation": a.genome.generation,
                }
                for a in self.swarm.agents.values()
            ],
        }