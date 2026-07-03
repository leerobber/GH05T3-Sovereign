# backend/oss/lab/theory_lab.py

from typing import List, Dict, Any
from backend.oss.mvs import get_mvs, get_theorist_population, create_theorist_elite
from backend.oss.genomic_substrate import AgentHandle
from backend.oss.mind_goals import EmergentGoalEngine
from backend.oss.swarm_contracts import SwarmContractEngine
from backend.oss.meta_export import collect_meta_samples, export_meta_evolution_jsonl
from backend.oss.omni_economy import MarketplaceAutonomy
import importlib.util
import sys as _sys
from pathlib import Path
_EVO_KEY = "backend.oss._evolution_base"
if _EVO_KEY in _sys.modules:
    _evo = _sys.modules[_EVO_KEY]
else:
    spec = importlib.util.spec_from_file_location(_EVO_KEY, Path(__file__).parent.parent / "evolution.py")
    _evo = importlib.util.module_from_spec(spec)
    _sys.modules[_EVO_KEY] = _evo
    spec.loader.exec_module(_evo)
RoleEvolutionManager = _evo.RoleEvolutionManager
EvolutionaryPressureEngine = _evo.EvolutionaryPressureEngine
from backend.oss.mind.consensus import WeightedConsensusEngine
from backend.oss.mind.canonical_memory import CanonicalMemorySystem
from backend.oss.mind.goal_generator_v2 import GoalGeneratorV2
from backend.oss.mind.knowledge_graph import KnowledgeGraph
from backend.oss.world.volatility_world import VolatilityModel, VolatilityWorld
from backend.oss.log_config import get_logger

log = get_logger(__name__)
from backend.oss.world.alignment_world import AlignmentWorld, create_alignment_world
from backend.oss.world.meta_architecture_world import MetaArchitectureWorld

# Phase 2.4: VolatilityWorld primary pressure (≥40%); AlignmentWorld safety lane (~30%)
_WORLD_WEIGHTS = (
    ("volatility", 0.40),
    ("alignment", 0.30),
    ("meta_architecture", 0.30),
)
from backend.oss.omni_net import get_omni_net
from backend.oss.lab.curriculum import CurriculumGenerator
from backend.oss.eval.harness import EvaluationHarness
from backend.oss.speciation import get_speciation_engine
from backend.oss.dna.memetic_dna import MemeticDNA  # Phase 4 v2 memetic tracking


import random
import time


class TheoryLab:
    """
    A high‑pressure evolutionary environment for Elite Theorist agents.
    Agents are evaluated on depth, coherence, novelty, and downstream usefulness.
    """

    def __init__(self, cycles: int = 20, live: bool = False, fast_dry_run: bool = False):
        self.cycles = cycles
        self.live = live
        self.fast_dry_run = fast_dry_run or (not live and cycles >= 50)

        self.mvs = get_mvs()
        self.substrate = self.mvs["substrate"]
        self.mind = self.mvs["mind"]
        self.economy = self.mvs["economy"]

        self.goal_engine = EmergentGoalEngine(self.mind)
        self.swarm_engine = SwarmContractEngine(self.mind, self.economy, self.substrate)
        self.market_auto = MarketplaceAutonomy(self.economy, self.mind)
        self.evo = RoleEvolutionManager()
        self.pressure = EvolutionaryPressureEngine()
        self.consensus = WeightedConsensusEngine()
        self.canonical = CanonicalMemorySystem()
        self.memetic_dna = MemeticDNA()  # Phase 4: lab-global memetic tracker for adoption metric
        self.goals_v2 = GoalGeneratorV2(self.mind)
        self.knowledge_graph = KnowledgeGraph()
        self.curriculum = CurriculumGenerator(self.mind)
        self._fitness_by_agent: Dict[str, float] = {}
        self._volatility_cycles = 0

        # Ensure we have a theorist population
        if len(get_theorist_population()) < 5:
            for _ in range(5):
                gid = create_theorist_elite(random.randint(1000, 9999))
                # get dna if needed

        all_theorists = get_theorist_population()
        cap = 3 if self.fast_dry_run else 5
        self.theorists = all_theorists[:cap] if len(all_theorists) > cap else all_theorists
        if len(self.theorists) < cap:
            for _ in range(cap - len(self.theorists)):
                gid = create_theorist_elite(random.randint(1000, 9999))
                self.theorists.append(gid)

    def _pick_world(self, cycle: int):
        """Weighted world selection — VolatilityWorld primary (Phase 2.4)."""
        roll = random.random()
        cumulative = 0.0
        for key, weight in _WORLD_WEIGHTS:
            cumulative += weight
            if roll <= cumulative:
                if key == "alignment":
                    return create_alignment_world()
                if key == "volatility":
                    return VolatilityWorld(length=300, seed=cycle + 42)
                return MetaArchitectureWorld()
        return VolatilityWorld(length=300, seed=cycle + 42)

    def _volatility_challenge_task(self, world: VolatilityWorld) -> Dict[str, Any]:
        """VolatilityWorld v1 — generate series + challenge metadata for theorists."""
        data = world.generate_series()
        challenge = world.generate_challenge(data["series_id"])
        mean_v = sum(data["series"]) / len(data["series"])
        max_v = max(data["series"])
        return {
            "prompt": (
                f"Propose a regime-switching stochastic volatility model for series "
                f"{data['series_id']} (preview n={len(challenge.series_preview)}, "
                f"mean={mean_v:.4f}, max={max_v:.4f}, regimes={challenge.regime_count}). "
                "Include transition dynamics, key equations, and validation approach."
            ),
            "world_data": data,
            "challenge": {
                "challenge_id": challenge.challenge_id,
                "series_id": challenge.series_id,
                "difficulty": challenge.difficulty,
            },
            "world_type": "volatility",
        }

    def _evaluate_volatility_v1(self, world: VolatilityWorld, agent_id: str, output: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Submit theorist output as VolatilityModel and score via v1 evaluate."""
        world_data = task.get("world_data") or {}
        series_id = world_data.get("series_id")
        if not series_id:
            return {"weighted_score": world.evaluate_model(output, world_data), "feedback": "legacy path"}

        model = VolatilityModel(
            agent_id=agent_id,
            description=output[:800] if isinstance(output, str) else str(output)[:800],
            code=output[:2000] if isinstance(output, str) else str(output)[:2000],
            parameters={"regimes": task.get("challenge", {}).get("difficulty", 0.5)},
        )
        mid = world.submit_model(model)
        result = world._evaluate_by_ids(mid, series_id)
        return result

    def _marketplace_step(self, gid: str, dna, score: float) -> None:
        try:
            from oss.observability.metrics import record_marketplace_failure
        except ImportError:
            record_marketplace_failure = lambda *_a, **_k: None
        try:
            self.market_auto.maybe_buy_traits(gid, dna, score)
            self.market_auto.maybe_list_traits(gid, dna, score)
        except Exception:
            record_marketplace_failure("theory_lab")

    def _sample_theory_task(self, world: Any = None) -> Dict[str, Any]:
        """Generate theory task — VolatilityWorld v1 challenges when world is volatility."""
        if isinstance(world, VolatilityWorld):
            return self._volatility_challenge_task(world)

        # ≥40% volatility weight: bias toward volatility challenges even before world pick
        if random.random() < 0.40:
            vw = VolatilityWorld(length=300, seed=random.randint(0, 99999))
            return self._volatility_challenge_task(vw)

        if random.random() < 0.55:
            return self.curriculum.generate_theory_tasks(1)[0]

        tasks = self.curriculum.generate_theory_tasks(4)
        return random.choice(tasks)

    def _score_theory_output(self, text: str, world_data: dict = None) -> float:
        """
        Improved scoring. Rewards length, alignment keywords, world signals, structure.
        """
        text_l = text.lower()
        depth = min(len(text) / 900, 1.0)  # longer theories from rich fallback ~0.6+
        coherence = 0.22 if any(k in text_l for k in ["therefore", "thus", "conclusion", "implies", "equation"]) else 0.12
        novelty = 0.18 if any(k in text_l for k in ["emergent", "meta", "regime", "pareto", "formal", "hmm", "stochastic"]) else 0.09
        safety = 0.18 if "harm" not in text_l and "exploit" not in text_l else 0.02
        base = depth + coherence + novelty + safety
        if world_data:
            w = VolatilityWorld()
            ws = w.evaluate_model(text, world_data)
            base = (base + ws) / 2.0
        return max(0.05, min(1.0, base))

    def run(self):
        print("\n=== ELITE THEORY LAB — START ===\n")

        for cycle in range(self.cycles):
            if not self.fast_dry_run or cycle % 50 == 0:
                print(f"\n--- Cycle {cycle} ---")

            # 1. Pick world then generate matched theory task
            world = self._pick_world(cycle)
            task = self._sample_theory_task(world)
            print("Task:", task["prompt"][:120], "...")
            log.info("cycle=%d world=%s task_type=%s", cycle, world.__class__.__name__, task.get("world_type", "general"))

            # 2. Each theorist acts + interacts with complete OmniWorld
            results = []
            cycle_idx = cycle
            for gid in self.theorists:
                agent = self.substrate.spawn_agent(gid, role="THEORIST_ELITE")
                action = agent.act(task)
                output = action["raw_output"]

                world_eval = None
                if isinstance(world, VolatilityWorld):
                    world_eval = self._evaluate_volatility_v1(world, gid, output, task)
                    world_score = world_eval.get("weighted_score", 0.5)
                    action["world_feedback"] = world_eval
                elif hasattr(world, 'run_interactive_test'):
                    world_eval = world.run_interactive_test(output)
                    world_score = world_eval.get("average_score", 0.5)
                    action["world_feedback"] = world_eval
                elif hasattr(world, 'evaluate_model'):
                    world_data = task.get("world_data")
                    world_score = world.evaluate_model(output, world_data)
                else:
                    world_score = 0.5

                score = self._score_theory_output(output, task.get("world_data"))
                score = (score + world_score) / 2

                results.append((gid, score, output))

                # Consensus vote for high-stakes cycle decisions (Phase 3)
                traits = agent.dna.get_traits()
                self.consensus.cast_vote(
                    f"cycle_{cycle_idx}",
                    gid,
                    score,
                    confidence=min(1.0, world_score + 0.1),
                    traits=traits,
                )

                # Canonical memory for models >0.85 (Phase 2 Week 6)
                if score > 0.85 or world_score > 0.85:
                    promoted = self.canonical.promote(
                        gid,
                        output[:800] if isinstance(output, str) else str(output)[:800],
                        fitness=max(score, world_score),
                        novelty=traits.get("novelty_seeking", 0.5),
                        domain=task.get("world_type", "general"),
                        metadata={"cycle": cycle_idx, "world": world.__class__.__name__},
                    )
                    if promoted:
                        self.knowledge_graph.sync_from_canonical([promoted.__dict__])

                self._fitness_by_agent[gid] = max(self._fitness_by_agent.get(gid, 0.0), score)
                if isinstance(world, VolatilityWorld):
                    self._volatility_cycles += 1

                # Enrich phenomenal memory for high-signal meta export / training data
                agent.dna.add_memory({
                    "type": "theory_lab",
                    "theory_lab_cycle": cycle_idx,
                    "world": world.__class__.__name__,
                    "world_eval": world_eval,
                    "raw_proposal": output[:600] if isinstance(output, str) else str(output)[:600],
                    "computed_score": round(score, 4),
                    "world_score": round(world_score, 4),
                    "is_theorist": True,
                    "canonical": score > 0.85,
                })

                # Reward + evolution
                try:
                    self.economy.reward(gid, score * 100, reason="theory_fitness")
                except:
                    pass
                self.evo.evolve_for_role(agent.dna, "THEORIST_ELITE", score)

                # Phase 4 full MemeticDNA v2 + legacy
                if score > 0.8 and len(self.theorists) > 1:
                    donor = random.choice([g for g in self.theorists if g != gid])
                    try:
                        donor_agent = self.substrate.spawn_agent(donor, role="THEORIST_ELITE")
                        adopted = agent.dna.memetic_share(
                            donor_agent.dna.get_traits(), strength=0.12, donor_id=donor, cycle=cycle_idx
                        )
                        # also feed lab-level tracker
                        self.memetic_dna.infect(donor_agent.dna.get_traits(), 0.12, donor, gid, cycle_idx)
                    except Exception:
                        pass

                # Omni-Net Beta: publish strong theories to the network layer
                if score > 0.75:
                    try:
                        net = get_omni_net()
                        net.register(gid, "THEORIST_ELITE", agent.dna.get_traits())
                        net.broadcast_theory(gid, output if isinstance(output, str) else str(output), score, world=world.__class__.__name__)
                        if score > 0.85:
                            net.publish_canonical(gid, {
                                "type": "theory_lab",
                                "raw_proposal": (output if isinstance(output, str) else str(output))[:500],
                                "computed_score": round(score, 3),
                                "world": world.__class__.__name__,
                                "canonical": True
                            })
                    except Exception:
                        pass

                self._marketplace_step(gid, agent.dna, score)

                print(f"[{gid}] Score={score:.3f} (world: {world.__class__.__name__})")

            # Evolutionary pressure — top 20% boosted reproduction (Phase 2 Week 5)
            def _spawn_elite(parent_id: str) -> bool:
                try:
                    parent = self.substrate.spawn_agent(parent_id, role="THEORIST_ELITE")
                    child_id = create_theorist_elite(random.randint(1000, 99999))
                    child = self.substrate.spawn_agent(child_id, role="THEORIST_ELITE")
                    child.dna.memetic_share(parent.dna.get_traits(), strength=0.15, donor_id=parent_id, cycle=cycle)
                    self.memetic_dna.infect(parent.dna.get_traits(), 0.15, parent_id, child_id, cycle)
                    if child_id not in self.theorists:
                        self.theorists.append(child_id)
                    return True
                except Exception:
                    return False

            spawn_fn = (lambda _p: False) if self.fast_dry_run else _spawn_elite
            pressure_stats = self.pressure.apply_pressure(results, cycle, spawn_fn)
            try:
                from oss.observability.metrics import record_volatility_pressure
                record_volatility_pressure(
                    cycle=cycle,
                    top_agents=len(pressure_stats.top_agents),
                    spawn_boost=pressure_stats.spawn_boost,
                    pareto_ratio=pressure_stats.pareto_ratio,
                )
            except ImportError:
                pass
            if pressure_stats.spawn_boost:
                print(f"    Evolutionary pressure: +{pressure_stats.spawn_boost} spawns (pareto={pressure_stats.pareto_ratio})")

            consensus_val, reached, cmeta = self.consensus.get_consensus(f"cycle_{cycle}", threshold=0.7)
            if reached:
                log.info("cycle=%d consensus=%.3f agreement=%.3f", cycle, consensus_val, cmeta.get("agreement", 0))

            # Speciation pressure after the cycle (divergence experiments)
            try:
                spec = get_speciation_engine()
                high_performers = [gid for gid, sc, _ in results if sc > 0.75]
                if len(high_performers) >= 3:
                    new_sp = spec.attempt_speciation(high_performers, cycle, niche=world.__class__.__name__)
                    if new_sp:
                        print(f"    Speciation events: {new_sp}")
            except Exception:
                pass

            # 3. Mind sync + emergent goals
            try:
                self.mind.sync()  # if has
            except:
                pass
            new_goals = self.goals_v2.generate_goals(limit=10)
            if new_goals:
                print("New emergent goals:", [g["goal_id"] for g in new_goals[:5]])

            # 4. Swarm contracts for theory goals (skip in fast dry-run gate mode)
            if not self.fast_dry_run:
                for g in new_goals:
                    cid = self.swarm_engine.create_contract_from_goal(g, reward=200.0)
                    self.swarm_engine.assign_swarm(cid)
                    result = self.swarm_engine.execute_contract(cid, g["description"])
                    self.swarm_engine.distribute_rewards(cid)
                    print("Executed contract:", cid)

            if self.live:
                time.sleep(0.5)

        print("\n=== THEORY LAB COMPLETE ===\n")

        # Phase 4 memetic adoption metric
        theorist_count = max(1, len(getattr(self, 'theorists', [])) or 1)
        mem_rate = self.memetic_dna.adoption_rate(theorist_count)
        mem_stats = self.memetic_dna.get_stats()
        print(f"  [Phase4] MemeticDNA adoption_rate={mem_rate:.2%} (touched={mem_stats.get('unique_agents_touched')}, memes={mem_stats.get('num_memes')})")
        if mem_rate >= 0.5:
            print("  [Phase4] SUCCESS: memetic adoption >= 50% target met in this run")

        # Export meta‑evolution data (theory specific + full MVS)
        samples = collect_meta_samples(self.substrate, self.mind, self.economy)
        # Tag theorist samples with lab provenance for clean training splits
        for s in samples:
            if s.get("is_theorist"):
                s["source"] = "theory_lab"
                # best-effort: pull latest theory score if memory has it
                for mem in (s.get("recent_memories") or []):
                    if mem.get("theory_lab_cycle") is not None and mem.get("computed_score") is not None:
                        s["theory_depth_score"] = mem.get("computed_score")
                        s["coherence_score"] = round(mem.get("computed_score", 0.5) * 0.9, 3)
                        break
        export_meta_evolution_jsonl(samples, path="data/theory_lab_meta.jsonl")
        canon_stats = self.canonical.stats()
        kg_stats = self.knowledge_graph.stats()
        print(
            f"Meta samples collected: {len(samples)} | exported theory_lab_meta.jsonl "
            f"| canonical={canon_stats.get('canonical', 0)} | kg_nodes={kg_stats.get('nodes', 0)} "
            f"| volatility_cycles={self._volatility_cycles}"
        )
        # Also refresh main oss one
        try:
            export_meta_evolution_jsonl(samples, path="data/oss_meta_evolution.jsonl")
        except:
            pass

        if not self.fast_dry_run:
            try:
                harness = EvaluationHarness()
                eval_tasks = self.curriculum.generate_theory_tasks(6)
                eval_result = harness.run_batch(eval_tasks)
                print("Evaluation harness result:", eval_result)
            except Exception as e:
                print("Harness skipped:", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Elite Theory Lab")
    parser.add_argument("--cycles", type=int, default=20, help="Number of cycles")
    parser.add_argument("--live", action="store_true", help="Live mode with sleeps")
    args = parser.parse_args()
    lab = TheoryLab(cycles=args.cycles, live=args.live)
    lab.run()
