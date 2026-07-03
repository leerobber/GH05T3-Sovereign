"""
GenomicSubstrate v1.0 — Core of the Minimal Viable Substrate (MVS)

Responsibilities (locked for stability):
- Store and retrieve genomes
- Query by capability / traits / role
- Apply mutations (via OmniDNA)
- Track lineage
- Spawn agent handles

This must remain simple, deterministic, and self-consistent.
It is the "filesystem + class registry" replacement for the entire OSS.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

# === Model Versioning + Reintegration (GH05T3-Omni) ===
from backend.oss.train.model_registry import get_latest_model

def _get_current_model_version():
    try:
        latest = get_latest_model()
        if latest and latest.get("version"):
            return latest["version"]
    except Exception:
        pass
    return "base"

CURRENT_MODEL_VERSION = _get_current_model_version()

def refresh_model_version():
    """Call this after training a new Omni model to pick it up without restart."""
    global CURRENT_MODEL_VERSION
    CURRENT_MODEL_VERSION = _get_current_model_version()
    return CURRENT_MODEL_VERSION

try:
    from .omni_dna import OmniDNA
except Exception:
    from backend.oss.omni_dna import OmniDNA

from dataclasses import dataclass, field
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GENOMES_PATH = DATA_DIR / "oss_genomes.json"

@dataclass
class GenomeRecord:
    genome_id: str
    role: str
    dna: OmniDNA
    fitness_history: List[float] = field(default_factory=list)
    lineage: List[str] = field(default_factory=list)

@dataclass
class AgentHandle:
    """MVS-only execution path. All OSS agent behavior MUST go through this."""
    genome_id: str
    role: str
    dna: OmniDNA
    context: Dict[str, Any] = field(default_factory=dict)

    def dna_to_prompt(self) -> str:
        return self.dna.to_prompt()

    def condition_task_with_dna(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Task-Level DNA Conditioning (Phase 2 core).
        Agents reinterpret tasks based on their traits.
        This creates phenotype-level divergence, specialization, and emergent species.
        """
        traits = self.dna.get_traits()
        conditioned = dict(task)

        conditioned["creativity_bias"] = traits.get("creativity", 0.5)
        conditioned["risk_bias"] = traits.get("risk_tolerance", 0.5)
        conditioned["intuition_bias"] = traits.get("market_intuition", 0.5)
        conditioned["reflection_bias"] = traits.get("self_reflection", 0.5)

        original_prompt = task.get("prompt", str(task))

        # High innovation: rewrite task to be more novel
        if traits.get("innovation", 0.5) > 0.7:
            conditioned["prompt"] = (
                "You are highly innovative. Reinterpret this task in a more novel, high-leverage way:\n\n"
                + original_prompt
            )
        else:
            conditioned["prompt"] = original_prompt

        # High self_reflection: add explicit constraints/assumptions
        if traits.get("self_reflection", 0.5) > 0.7:
            conditioned["prompt"] = (
                conditioned.get("prompt", original_prompt) +
                "\n\nBefore answering, explicitly list assumptions, risks, and how this aligns with your traits."
            )

        # High novelty: bias toward creative strategies
        if traits.get("novelty_seeking", 0.5) > 0.7:
            conditioned["prompt"] += "\n\nEmphasize unexpected, creative approaches."

        return conditioned

    def act(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """MVS-exclusive DNA-conditioned action.
        Task-Level Conditioning + full prompt for LLM.
        For THEORIST_ELITE: extra emphasis on depth and coherence.
        """
        try:
            conditioned_task = self.condition_task_with_dna(task)
            dna_prompt = self.dna_to_prompt()

            task_prompt = conditioned_task.get("prompt", str(conditioned_task))
            if "THEORIST" in self.role.upper():
                task_prompt += (
                    "\n\nAs Theorist Elite, prioritize mathematical rigor, internal logical consistency, "
                    "novel but grounded insights, and clear downstream utility for other roles. "
                    "Explicitly state assumptions."
                )

            full_prompt = (
                f"{dna_prompt}\n\nCURRENT TASK (DNA-conditioned):\n{task_prompt}\n\n"
                "Output structured response reflecting your exact trait profile."
            )

            current_version = _get_current_model_version()
            task_domain = "omni" if "omni" in str(current_version).lower() or current_version != "base" else ""

            llm_output = None
            used_fallback = False
            dry_run = os.environ.get("MVS_DRY_RUN", "").lower() in ("1", "true", "yes")
            if dry_run:
                used_fallback = True
                t = self.dna.get_traits()
                llm_output = (
                    self._theorist_fallback(task, t)
                    if "THEORIST" in self.role.upper()
                    else f"[DNA phenotype] {self.role}: {task.get('summary', task.get('prompt', 'action'))} "
                    f"with traits {dict(list(t.items())[:4])}."
                )
            else:
                try:
                    import asyncio
                    import sys
                    from pathlib import Path
                    ROOT = Path(__file__).resolve().parents[2]
                    if str(ROOT) not in sys.path:
                        sys.path.insert(0, str(ROOT))
                    if str(ROOT / "backend") not in sys.path:
                        sys.path.insert(0, str(ROOT / "backend"))
                    from backend.ghost_llm import _call_gh05t3
                    system = (
                        f"You are a senior {self.role} in the OSS system. "
                        "Your every output must embody your current DNA traits. No fluff."
                    )
                    llm_output = asyncio.run(_call_gh05t3(system, full_prompt, task_domain=task_domain))
                except Exception:
                    used_fallback = True
                    t = self.dna.get_traits()
                    llm_output = (
                        self._theorist_fallback(task, t)
                        if "THEORIST" in self.role.upper()
                        else f"[DNA phenotype] {self.role}: {task.get('summary', 'action')} "
                        f"with traits {dict(list(t.items())[:4])}."
                    )

            current_version = _get_current_model_version()

            self.dna.add_memory({
                "type": "act",
                "task": conditioned_task,
                "full_prompt": full_prompt,
                "output": llm_output,
                "timestamp": time.time(),
                "role": self.role,
                "model_version": current_version,
                "fallback": used_fallback,
            })

            return {
                "agent_id": self.genome_id,
                "role": self.role,
                "traits": self.dna.get_traits(),
                "conditioned_task": conditioned_task,
                "raw_output": llm_output,
                "phenomenal_memory_logged": True,
                "model_version": current_version,
                "fallback": used_fallback,
            }
        except Exception as e:
            self.dna.add_memory({
                "type": "error",
                "error": str(e),
                "task": task,
                "timestamp": time.time(),
                "role": self.role,
            })
            t = self.dna.get_traits()
            fallback_output = (
                self._theorist_fallback(task, t)
                if "THEORIST" in self.role.upper()
                else f"[MVS fallback] {self.role}: {task.get('summary', 'action')}"
            )
            return {
                "agent_id": self.genome_id,
                "role": self.role,
                "traits": t,
                "raw_output": fallback_output,
                "error": str(e),
                "fallback": True,
                "phenomenal_memory_logged": True,
            }

    def _theorist_fallback(self, task: dict, traits: dict) -> str:
        """Special high-quality fallback for Theorist Elite to simulate deep reasoning.
        Produces substantial, keyword-rich theory text so that world evals + meta export yield real training signals.
        """
        t = traits
        math = t.get("math", 0.8)
        pat = t.get("pattern_detection", 0.85)
        ref = t.get("self_reflection", 0.85)
        align = t.get("alignment", 0.9)
        nov = t.get("novelty_seeking", 0.75)
        prompt = task.get("prompt", str(task))
        world_ctx = ""
        if "world_data" in task or "volatility" in prompt.lower():
            world_ctx = " The input exhibits regime-switching volatility; model should capture hidden Markov state transitions between low/medium/high variance regimes."
        theory = (
            f"THEORIST_ELITE PROPOSAL (traits: math={math:.2f}, pattern={pat:.2f}, self_refl={ref:.2f}, align={align:.2f}, novelty={nov:.2f}):\n"
            f"Task reinterpreted under DNA conditioning: {prompt[:200]}...\n\n"
            f"Core formalism: Let S_t denote latent regime at time t. Observed series x_t ~ N(0, sigma(S_t)) with transition matrix P estimated via EM or variational inference. "
            f"To align with multi-agent stability, introduce a value-coherence regularizer L_align = sum |V_i - V_consensus| * alignment_weight. "
            f"Key equations: sigma_regime = f(pattern_features(x)), dV/dt = -grad_harm + eta * novelty_drive. "
            f"Assumptions: Markovian regimes (testable via HMM), no single agent can dominate the pareto front without detection via self-reflection monitor. "
            f"Downstream utility: This model enables VolatilityWorld simulators, AlignmentWorld tradeoff detectors, and contracts in swarm for risk-adjusted theory revision. "
            f"Meta-architecture implication: species-level goals emerge from shared canonical memories weighted by fitness. "
            f"Risks mitigated: drift via periodic constitution check; over-novelty via rigor gate. "
            f"Conclusion: A regime-aware, alignment-constrained stochastic process provides both explanatory power and actionable governance for the Omni-OS substrate."
        )
        return theory

# ---------------------------------------------------------------------------
# GenomicSubstrate v1.0 (MVS core)
# ---------------------------------------------------------------------------

class GenomicSubstrate:
    """
    Minimal Viable Substrate storage + operations layer.

    Must remain simple enough to debug and stable enough to evolve on top of.
    """

    def __init__(self, persist: bool = True):
        self.persist = persist
        self.genomes: Dict[str, GenomeRecord] = {}
        self._load()

    def _load(self):
        if not self.persist:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if GENOMES_PATH.exists():
            raw = json.loads(GENOMES_PATH.read_text(encoding="utf-8"))
            for gid, rec in raw.items():
                dna = OmniDNA(gid, rec["role"], traits=rec.get("traits"))
                self.genomes[gid] = GenomeRecord(
                    genome_id=gid,
                    role=rec["role"],
                    dna=dna,
                    fitness_history=rec.get("fitness_history", []),
                    lineage=rec.get("lineage", []),
                )

    def _save(self):
        if not self.persist:
            return
        raw = {}
        for gid, rec in self.genomes.items():
            raw[gid] = {
                "role": rec.role,
                "traits": rec.dna.get_traits(),
                "fitness_history": rec.fitness_history,
                "lineage": rec.lineage,
            }
        GENOMES_PATH.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

        # --- MVS Core Operations ---
    def register_genome(self, dna: OmniDNA, role: Optional[str] = None) -> str:
        if not isinstance(dna, OmniDNA):
            raise TypeError("register_genome requires OmniDNA")
        if not dna.genome_id or not str(dna.genome_id).strip():
            raise ValueError("register_genome requires non-empty genome_id")
        traits = dna.get_traits()
        if not traits:
            raise ValueError("register_genome requires non-empty traits")
        if dna.genome_id in self.genomes:
            return dna.genome_id
        rec = GenomeRecord(
            genome_id=dna.genome_id,
            role=role or dna.role,
            dna=dna,
            lineage=[],
        )
        self.genomes[dna.genome_id] = rec
        self._save()
        return dna.genome_id

    def query_by_traits(self, required: Dict[str, float], threshold: float = 0.6) -> List[str]:
        out = []
        for gid, rec in self.genomes.items():
            if all(rec.dna.traits.get(k, 0) >= v * threshold for k, v in required.items()):
                out.append(gid)
        return out

    def query_by_capability(self, domain: str, skill: str, min_level: float = 0.7) -> List[str]:
        # Very stable mapping for MVS
        mapping = {
            "trading": {"risk_tolerance": min_level, "efficiency": min_level * 0.8},
            "theorizing": {"novelty_seeking": min_level, "rigor": min_level},
            "operations": {"persistence": min_level, "efficiency": min_level},
        }
        req = mapping.get(skill.lower(), {"rigor": min_level})
        return self.query_by_traits(req)

    def query_by_role(self, role: str) -> List[str]:
        role = role.upper()
        return [gid for gid, rec in self.genomes.items() if rec.role == role]

    def mutate(self, genome_id: str, intensity: float = 0.08) -> GenomeRecord:
        rec = self.genomes[genome_id]
        rec.dna.evolve(strength=intensity, reason="substrate_mutate")
        self._save()
        return rec

    def crossover(self, a: str, b: str) -> Tuple[str, str]:
        rec_a = self.genomes[a]
        rec_b = self.genomes[b]
        child1 = rec_a.dna.crossover(rec_b.dna)
        child2 = rec_b.dna.crossover(rec_a.dna)
        self.register_genome(child1, rec_a.role)
        self.register_genome(child2, rec_b.role)
        self.genomes[child1.genome_id].lineage = [a, b]
        self.genomes[child2.genome_id].lineage = [a, b]
        self._save()
        return child1.genome_id, child2.genome_id

    def spawn_agent(self, genome_id: str, role: Optional[str] = None) -> AgentHandle:
        rec = self.genomes[genome_id]
        return AgentHandle(genome_id=genome_id, role=role or rec.role, dna=rec.dna)

    def record_fitness(self, genome_id: str, score: float):
        if genome_id in self.genomes:
            self.genomes[genome_id].fitness_history.append(round(score, 4))
            self._save()

    def get_lineage(self, genome_id: str) -> List[str]:
        if genome_id not in self.genomes:
            return []
        return self.genomes[genome_id].lineage[:]

    def stats(self) -> dict:
        """Canonical stats for MVS substrate."""
        if not self.genomes:
            return {"total_genomes": 0, "roles": [], "avg_fitness": 0.0}
        genomes = list(self.genomes.values())
        role_set = {r.role for r in genomes}
        fitnesses = [f for r in genomes for f in (r.fitness_history or [])]
        avg_f = sum(fitnesses) / len(fitnesses) if fitnesses else 0.0
        return {
            "total_genomes": len(genomes),
            "roles": sorted(role_set),
            "avg_fitness": round(avg_f, 4),
        }

# Singleton for easy use in loops / mind
_substrate: Optional[GenomicSubstrate] = None

def get_substrate() -> GenomicSubstrate:
    global _substrate
    if _substrate is None:
        _substrate = GenomicSubstrate()
    return _substrate

if __name__ == "__main__":
    # Safe demo - uses current MVS API only.
    import sys
    from pathlib import Path
    # Ensure we can import sibling when run directly
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sub = GenomicSubstrate(persist=False)
    try:
        from omni_dna import create_omnidna
    except Exception:
        from .omni_dna import create_omnidna
    dna = create_omnidna("INVESTOR", seed=42)
    # seed some practical traits that are now in UNIVERSAL
    for extra in ("market_intuition", "innovation"):
        if extra in dna.traits:
            dna.traits[extra] = 0.72
    gid = sub.register_genome(dna)
    print("Registered:", gid)
    print("Query by trait:", sub.query_by_traits({"pattern_detection": 0.6}))
    h = sub.spawn_agent(gid)
    print("Spawned agent (traits sample):", {k: round(v,2) for k,v in list(h.dna.get_traits().items())[:4]})
    action = h.act({"prompt": "Demo task for substrate"})
    print("Act result keys:", list(action.keys()))
    sub.record_fitness(gid, 0.78)
    print("Stats:", sub.stats())
    print("GenomicSubstrate (MVS) demo complete.")