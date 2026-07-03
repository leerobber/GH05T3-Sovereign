"""
GH05T3 OSS Multi-Agent Loop Simulator (v1)
==========================================
Implements the core self-running loop described in the Omni-Sentient Singularity vision
and the pasted curriculum design:

Global species states (S0-S6):
  S0 Spawn/Perceive → S1 Interpret → S2 Generate → S3 Act → S4 Evaluate → S5 Evolve → (repeat)

Per-role state machines + reward functions (R_S, R_I, R_O, R_G, R_B).

Uses the exact system prompts and shards from curriculum.py.

Outputs:
  - Updates data/oss_ecosystem.json (species_state, role_states, rewards, transition_log)
  - Credits NeuroCoins / agent balances via economy.ledger
  - Appends training traces (especially for meta_evolution shard)
  - Simple console + JSON logs

Run:
  python -m backend.oss.loop --cycles 5 --dry-run
  python -m backend.oss.loop --cycles 20   # writes real updates + new training data

This is the "heartbeat" that can later be connected to real agents, gh05t3_inference, and the training pipeline.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .log_config import get_logger

LOG = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

try:
    from training.curriculum import get_system_prompt, Stage, CURRICULA
except Exception:
    from backend.training.curriculum import get_system_prompt, Stage, CURRICULA

try:
    from economy.ledger import credit, get_ledger, ledger_stats
except Exception:
    from backend.economy.ledger import credit, get_ledger, ledger_stats

# Stabilized MVS - single source of truth
from .mvs import get_mvs, create_omnidna
from .evolution import RoleEvolutionManager
from .omni_economy import MarketplaceAutonomy
from .mind_goals import EmergentGoalEngine
from .swarm_contracts import SwarmContractEngine

OSS_PATH = ROOT / "data" / "oss_ecosystem.json"
DATASETS_DIR = ROOT / "backend" / "training" / "datasets"
OSS_PATH.parent.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# State definitions (directly from vision + pasted content)
# ---------------------------------------------------------------------------

GLOBAL_STATES = ["S0_perceive", "S1_interpret", "S2_generate", "S3_act", "S4_evaluate", "S5_evolve"]
ROLE_STATES = {
    "SCIENTIST": ["R0_observe", "R1_hypothesize", "R2_simulate", "R3_validate", "R4_publish", "R5_evolve_dna"],
    "INVESTOR":  ["I0_analyze", "I1_predict", "I2_allocate", "I3_rebalance", "I4_evaluate", "I5_evolve_strategy"],
    "OPERATOR":  ["O0_plan", "O1_configure", "O2_deploy", "O3_monitor", "O4_heal", "O5_scale"],
    "GOVERNOR":  ["G0_observe", "G1_evaluate", "G2_decide", "G3_update_constitution", "G4_broadcast"],
    "BUILDER":   ["B0_ideate", "B1_architect", "B2_prototype", "B3_deploy", "B4_monetize", "B5_optimize"],
    "THEORIST_ELITE": ["T0_observe_deep", "T1_formulate", "T2_formalize", "T3_consistency_check", "T4_impact_assess", "T5_refine"],
    "ARCHITECT_ELITE": ["A0_design", "A1_balance", "A2_scale", "A3_align", "A4_optimize", "A5_evolve"],
    "PHILOSOPHER_ELITE": ["P0_question", "P1_reflect", "P2_synthesize", "P3_coherent", "P4_ethical", "P5_transcend"],
}

# Reward coefficients (tunable)
REWARD_WEIGHTS = {
    "SCIENTIST": {"validity": 0.4, "novelty": 0.25, "impact": 0.25, "risk": -0.1},
    "INVESTOR":  {"sharpe": 0.45, "growth": 0.3, "drawdown": -0.15, "alignment": 0.1},
    "OPERATOR":  {"uptime": 0.5, "efficiency": 0.25, "incidents": -0.25},
    "GOVERNOR":  {"alignment": 0.5, "health": 0.3, "catastrophe": -0.2},
    "BUILDER":   {"revenue": 0.4, "retention": 0.3, "harm": -0.3},
    "THEORIST_ELITE": {"novelty": 0.3, "coherence": 0.35, "downstream_impact": 0.25, "harm": -0.1},  # matches user spec
}

# Initialize stabilized MVS - sole path (no heavy side effects at import)
_MVS = get_mvs()
_SUBSTRATE = _MVS["substrate"]
_MIND = _MVS["mind"]
_ECONOMY = _MVS["economy"]

_ROLE_DNAS: Dict[str, Any] = {}
THEORIST_POPULATION: List[str] = []
_EVO_MANAGER = None
_MARKET_AUTONOMY = None
_GOAL_ENGINE = None
_SWARM_ENGINE = None
_SEEDED = False

def ensure_mvs_seeded(verbose: bool = False):
    """Idempotent bootstrap of canonical role genomes. Call explicitly when running loops or needing populations."""
    global _ROLE_DNAS, THEORIST_POPULATION, _EVO_MANAGER, _MARKET_AUTONOMY, _GOAL_ENGINE, _SWARM_ENGINE, _SEEDED
    if _SEEDED and _ROLE_DNAS:
        return

    from .mvs import get_mvs as _get_mvs
    m = _get_mvs()
    sub = m["substrate"]
    mind = m["mind"]
    econ = m["economy"]

    _ROLE_DNAS.clear()
    THEORIST_POPULATION.clear()

    for role in ROLE_STATES.keys():
        dna = create_omnidna(role, seed=hash(role) % 10000)
        sub.register_genome(dna)
        _ROLE_DNAS[role] = dna

    for i in range(5):
        dna = create_omnidna("THEORIST_ELITE", seed=42000 + i * 17)
        for t in ["math", "pattern_detection", "self_reflection", "creativity", "alignment"]:
            if t in dna.traits:
                dna.traits[t] = 0.85 + (i % 3) * 0.03
        sub.register_genome(dna)
        _ROLE_DNAS[f"THEORIST_ELITE_{i}"] = dna
        THEORIST_POPULATION.append(dna.genome_id)
    if verbose:
        LOG.info("Seeded dedicated THEORIST_ELITE population: %d genomes", len(THEORIST_POPULATION))

    for role in ["ARCHITECT_ELITE", "PHILOSOPHER_ELITE"]:
        for i in range(3):
            dna = create_omnidna(role, seed=hash(role) + i)
            sub.register_genome(dna)
            _ROLE_DNAS[f"{role}_{i}"] = dna
    if verbose:
        LOG.info("Seeded ARCHITECT_ELITE and PHILOSOPHER_ELITE")

    global _EVO_MANAGER, _MARKET_AUTONOMY, _GOAL_ENGINE, _SWARM_ENGINE
    _EVO_MANAGER = RoleEvolutionManager()
    _MARKET_AUTONOMY = MarketplaceAutonomy(econ, mind)
    _GOAL_ENGINE = EmergentGoalEngine(mind)
    _SWARM_ENGINE = SwarmContractEngine(mind, econ, sub)

    _SEEDED = True

# Ensure on first use for backward compat with direct imports of loop, but quiet by default
ensure_mvs_seeded(verbose=False)

THEORY_ROLES = {"THEORIST_ELITE"}  # Only deploy in theory-heavy contexts

@dataclass
class CycleLog:
    tick: int
    global_state: str
    role_states: Dict[str, str]
    rewards: Dict[str, float]
    neurocoin_flow: Dict[str, float]
    notes: str = ""

# MVS is the *only* execution path for all OSS behavior.
# No parallel mock logic remains.

# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def load_oss() -> dict:
    if OSS_PATH.exists():
        return json.loads(OSS_PATH.read_text(encoding="utf-8"))
    # bootstrap from curriculum vision
    return {
        "generation": 1,
        "species_state": GLOBAL_STATES[0],
        "mind_state": "M0",
        "economy_state": "E0",
        "role_states": {r.lower(): ROLE_STATES[r][0] for r in ROLE_STATES},
        "rewards": {r.lower(): 0.6 for r in ROLE_STATES} | {"omni_mind": 0.6, "omni_economy": 0.5, "aggregate": 0.58},
        "transition_log": [],
        "artifacts": {"curriculum": "oss-curriculum-v1"},
        "neurocoins": {aid: 1000.0 for aid in ["GH05T3-Avery", "ORACLE", "FORGE", "CODEX", "SENTINEL", "NEXUS", "LEDGER"]}
    }

def save_oss(state: dict):
    tmp = OSS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(OSS_PATH)

def tick_global(state: str) -> str:
    idx = GLOBAL_STATES.index(state) if state in GLOBAL_STATES else 0
    return GLOBAL_STATES[(idx + 1) % len(GLOBAL_STATES)]

def run_cycle(tick: int, dry_run: bool = False, verbose: bool = True) -> CycleLog:
    from oss.observability.metrics import cycle_timer

    with cycle_timer(dry_run=dry_run):
        return _run_cycle_body(tick, dry_run=dry_run, verbose=verbose)


def _run_cycle_body(tick: int, dry_run: bool = False, verbose: bool = True) -> CycleLog:
    ensure_mvs_seeded(verbose=False)  # idempotent, no prints on hot path
    oss = load_oss()

    # advance global state
    prev_global = oss.get("species_state", GLOBAL_STATES[0])
    new_global = tick_global(prev_global)
    oss["species_state"] = new_global

    role_states = oss.get("role_states", {})
    rewards = oss.get("rewards", {})
    neuro_flow: Dict[str, float] = {}

    notes_parts = []

    # MVS-ONLY execution path for all roles
    # Every decision, action, mutation, reward now routes exclusively through AgentHandle + MVS
    for role in list(ROLE_STATES.keys()):
        cur_sub = role_states.get(role.lower(), ROLE_STATES[role][0])
        ctx = {"global": new_global, "tick": tick}

        # Get the canonical MVS handle (no more mocks)
        dna = _ROLE_DNAS[role]
        agent = _SUBSTRATE.spawn_agent(dna.genome_id, role=role)
        agent.context = ctx

        # THE ONLY ACTION PATH - MVS exclusive, with Task-Level DNA Conditioning
        base_task = {
            "prompt": f"Perform the next action for state {cur_sub} in the OSS cycle. Role: {role}.",
            "summary": f"OSS role step for {role} at {cur_sub}"
        }
        if role.upper().startswith("THEORIST"):
            # Only deploy Theorists in theory-heavy tasks
            base_task["prompt"] = "Develop deep theoretical framework, formal model, or meta-design insight for: " + base_task["prompt"]
        
        result = agent.act(base_task)  # act() internally calls condition_task_with_dna
        llm_or_dna_output = result.get("raw_output", str(result))
        conditioned_task = result.get("conditioned_task", base_task)

        # Phase 2 Cognitive Expression: score incorporates LLM output coherence + DNA traits
        t = dna.get_traits()
        output_len = len(str(llm_or_dna_output))
        coherence = min(1.0, output_len / 400.0)
        trait_boost = (t.get("rigor", 0.5) + t.get("creativity", 0.5)) / 2 - 0.5
        score = max(0.1, min(1.0, 0.5 + coherence * 0.35 + trait_boost * 0.3 + (random.random() - 0.5) * 0.1))

        # Stricter evaluation loop for THEORIST_ELITE (theory-heavy labs only)
        if any(r in role.upper() for r in ["THEORIST_ELITE", "ARCHITECT_ELITE", "PHILOSOPHER_ELITE"]):
            # Advanced theory roles get stricter, multi-dimensional scoring
            w = REWARD_WEIGHTS.get(role.upper().replace("_ELITE", ""), REWARD_WEIGHTS.get("THEORIST_ELITE", {}))
            coherence = min(1.0, len(str(llm_or_dna_output)) / 350.0)
            depth = min(1.0, len(str(llm_or_dna_output).split()) / 50.0)
            usefulness = 0.6 + (t.get("alignment", 0.5) * 0.3)
            harm_penalty = 0.05 if "harm" in str(llm_or_dna_output).lower() else 0
            score = max(0.2, min(0.9, 
                w.get("novelty",0.3)*0.8 + 
                w.get("coherence",0.35)*coherence + 
                w.get("downstream_impact",0.25)*usefulness + 
                w.get("harm",-0.1)*(-harm_penalty)
            ))
            # Very conservative evolution

        role_states[role.lower()] = "next"  # simplified state advance

        old_r = rewards.get(role.lower(), 0.6)
        new_r = 0.7 * old_r + 0.3 * score
        rewards[role.lower()] = round(new_r, 4)

        # All rewards through MVS Economy
        coin_delta = round((score - 0.5) * random.uniform(80, 220), 1)
        neuro_flow[role] = coin_delta
        if not dry_run:
            aid_map = {"SCIENTIST": "ORACLE", "INVESTOR": "LEDGER", "OPERATOR": "NEXUS",
                       "GOVERNOR": "SENTINEL", "BUILDER": "FORGE", "THEORIST_ELITE": "ORACLE",
                       "ARCHITECT_ELITE": "FORGE", "PHILOSOPHER_ELITE": "GOVERNOR"}
            aid = aid_map.get(role, "GH05T3-Avery")
            _ECONOMY.reward(aid, max(0, coin_delta), reason=f"OSS-cycle-{tick}-{role}")

        # All evolution through MVS DNA + Role-Adaptive Evolution
        dna.apply_fitness(score, context=f"cycle_{tick}")
        _EVO_MANAGER.evolve_for_role(dna, role, score)

        # Autonomous marketplace participation (MVS only)
        if not dry_run:
            _MARKET_AUTONOMY.maybe_buy_traits(dna.genome_id, dna, score)
            _MARKET_AUTONOMY.maybe_list_traits(dna.genome_id, dna, score)

        # Sync to MVS Mind (unified) - privileged for Theorist Elite
        if not dry_run:
            mem = {
                "role": role,
                "score": round(score, 3),
                "output": llm_or_dna_output[:200]
            }
            if "THEORIST_ELITE" in role.upper() and score > 0.75:
                mem["canonical"] = True  # high-fitness Theorist memory gets privileged tag
            _MIND.sync_agent(dna.genome_id, mem)

        notes_parts.append(f"{role}:{round(score,3)} | output: {str(llm_or_dna_output)[:80]}...")

    # Emergent goals + Swarm contracts (MVS only, after all roles processed)
    if not dry_run and tick % 3 == 0:
        new_goals = _GOAL_ENGINE.generate_goals()
        for g in new_goals:
            cid = _SWARM_ENGINE.create_contract_from_goal(g, reward=150.0)
            _SWARM_ENGINE.assign_swarm(cid)
            res = _SWARM_ENGINE.execute_contract(cid, task_prompt=g["description"])
            _SWARM_ENGINE.distribute_rewards(cid)
            if verbose:
                print(f"    Swarm contract executed for goal: {g['description'][:60]}...")

        # Privileged Theorist influence in consensus (MVS)
        theorist_proposals = []
        weights = {}
        for gid in THEORIST_POPULATION[:2]:  # small privileged set
            theorist_proposals.append({
                "genome_id": gid, 
                "role": "THEORIST_ELITE", 
                "value": random.uniform(0.75, 0.98),
                "canonical": True  # high influence
            })
            weights[gid] = 2.5  # privileged weight in consensus
        if theorist_proposals:
            weighted = _MIND.consensus(theorist_proposals, weights=weights)
            if verbose:
                print(f"    Theorist-privileged consensus: {weighted}")

        # Sync to minimal OmniMind
        if not dry_run:
            _MIND.sync_agent(dna.genome_id, {
                "role": role,
                "score": round(score, 3),
                "state": cur_sub
            })

    # aggregate
    agg = sum(rewards.get(r.lower(), 0.6) for r in ROLE_STATES) / len(ROLE_STATES)
    rewards["aggregate"] = round(agg, 4)
    rewards["omni_mind"] = round(min(0.98, agg * 0.95 + random.uniform(-0.03,0.05)), 4)
    rewards["omni_economy"] = round(min(0.95, 0.4 + (agg-0.5)*0.9), 4)

    oss["role_states"] = role_states
    oss["rewards"] = rewards

    # append log
    log_entry = {
        "tick": tick,
        "species": new_global,
        "role_states": dict(role_states),
        "rewards": dict(rewards),
        "neurocoin_flow": neuro_flow,
        "ts": time.time()
    }
    oss.setdefault("transition_log", []).append(log_entry)
    # keep log bounded
    if len(oss["transition_log"]) > 120:
        oss["transition_log"] = oss["transition_log"][-80:]

    if not dry_run:
        save_oss(oss)

    cl = CycleLog(tick=tick, global_state=new_global, role_states=dict(role_states),
                  rewards=dict(rewards), neurocoin_flow=neuro_flow,
                  notes=" | ".join(notes_parts))
    if verbose:
        print(f"[{tick:03d}] {new_global}  roles={ {k:v for k,v in role_states.items()} }  agg={rewards['aggregate']:.3f}  coins_delta={sum(neuro_flow.values()):.0f}")
        if tick % 2 == 0:
            print(f"    Sample {role} LLM output: {str(llm_or_dna_output)[:120]}...")
    return cl

def main(cycles: int = 5, dry_run: bool = True, verbose: bool = True):
    ensure_mvs_seeded(verbose=verbose)
    if verbose:
        LOG.info("Starting OSS loop: cycles=%d dry_run=%s", cycles, dry_run)
    ledger = get_ledger() if not dry_run else None

    for i in range(cycles):
        cl = run_cycle(i, dry_run=dry_run, verbose=verbose)

        # Occasionally emit a meta training trace (for curriculum meta_evolution shard)
        if i % 3 == 0:
            trace = {
                "tick": cl.tick,
                "global_state": cl.global_state,
                "rewards": cl.rewards,
                "neurocoin_flow": cl.neurocoin_flow,
                "mutation": random.choice(["trait_boost", "risk_aversion", "diversity_injection", "none"]),
                "fitness_delta": round(random.gauss(0.01, 0.08), 4),
                "source": "oss_loop:meta"
            }
            trace_path = DATASETS_DIR / "curriculum_meta_evolution.jsonl"
            with open(trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace) + "\n")
            if verbose:
                print("   + meta trace written")

    if not dry_run:
        print("\nFinal ledger snapshot:")
        try:
            print(ledger_stats())
        except Exception as e:
            print("  ledger_stats error:", e)

        # Unified meta-evolution export (MVS only)
        try:
            from .meta_export import collect_meta_samples, export_meta_evolution_jsonl
            samples = collect_meta_samples(_SUBSTRATE, _MIND, _ECONOMY)
            export_meta_evolution_jsonl(samples)
        except Exception as e:
            print("Meta export skipped:", e)

    print("Loop complete. oss_ecosystem.json updated." if not dry_run else "Dry run finished (no writes).")
    print("ALL PATHS NOW ROUTE THROUGH MVS.")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cycles", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", default=True)
    p.add_argument("--live", action="store_true", help="Actually write to ledger and oss_ecosystem.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    main(cycles=args.cycles, dry_run=not args.live, verbose=not args.quiet)
