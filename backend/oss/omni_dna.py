"""
OmniDNA v1.0 — Stable Genome for the Minimal Viable Substrate (MVS)

This is the "RNA layer" of Omni-OS.

Fixed schema, predictable evolution, mutation logging, and fitness feedback.

Design goals for stability:
- Deterministic when seeded
- Bounded trait values [0.1, 0.95]
- Explicit mutation events with logging
- Simple fitness application
- No external dependencies beyond stdlib

This must stay minimal. Everything else builds on top of this.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Phase 4 v2 DNA (composed)
from backend.oss.dna.meta_dna import MetaDNA
from backend.oss.dna.memetic_dna import MemeticDNA
from backend.oss.dna.fractal_dna import FractalDNA
from backend.oss.dna.alchemical_dna import AlchemicalDNA

# ─────────────────────────────────────────────────────────────
# FIXED UNIVERSAL TRAIT SCHEMA (stable across all roles)
# ─────────────────────────────────────────────────────────────

UNIVERSAL_TRAITS = [
    "novelty_seeking",   # drive to explore new ideas/patterns
    "rigor",             # precision, verification, resistance to hallucination
    "risk_tolerance",    # willingness to try high-variance actions
    "persistence",       # ability to continue under low immediate reward
    "creativity",        # recombination of existing elements
    "efficiency",        # resource / step economy
    "collaboration",     # openness to swarm / mind coordination
    "math",              # mathematical reasoning depth
    "pattern_detection", # ability to spot structures and regularities
    "self_reflection",   # meta-cognition, assumption checking
    "alignment",         # ethical/coherence with larger goals
    # Extended practical traits used across evolution, conditioning, labs (kept stable for MVS)
    "innovation",        # novel recombination + high-leverage ideas
    "market_intuition",  # sensing value, timing, opportunity
    "empathy",           # understanding other agents/users for collaboration
]

ROLE_DEFAULTS = {
    "SCIENTIST": {"novelty_seeking": 0.78, "rigor": 0.85, "creativity": 0.72, "innovation": 0.70},
    "INVESTOR":  {"risk_tolerance": 0.62, "efficiency": 0.70, "persistence": 0.68, "market_intuition": 0.75},
    "OPERATOR":  {"persistence": 0.82, "efficiency": 0.88, "rigor": 0.75},
    "GOVERNOR":  {"rigor": 0.80, "collaboration": 0.55, "risk_tolerance": 0.40, "empathy": 0.60},
    "BUILDER":   {"creativity": 0.68, "efficiency": 0.65, "collaboration": 0.70, "innovation": 0.65, "market_intuition": 0.60},
    "THEORIST_ELITE": {
        "math": 0.92,
        "pattern_detection": 0.88,
        "self_reflection": 0.85,
        "creativity": 0.82,
        "alignment": 0.90,
        "rigor": 0.87,
        "novelty_seeking": 0.78,
        "innovation": 0.80,
    },
    "ARCHITECT_ELITE": {
        "creativity": 0.90,
        "efficiency": 0.85,
        "self_reflection": 0.80,
        "collaboration": 0.85,
        "alignment": 0.80,
        "pattern_detection": 0.82,
        "innovation": 0.75,
    },
    "PHILOSOPHER_ELITE": {
        "self_reflection": 0.95,
        "alignment": 0.92,
        "novelty_seeking": 0.80,
        "creativity": 0.75,
        "rigor": 0.85,
        "math": 0.70,
        "empathy": 0.70,
    },
    "WEB_ENGINEER_ELITE": {
        "creativity": 0.94,
        "efficiency": 0.92,
        "innovation": 0.90,
        "market_intuition": 0.91,
        "pattern_detection": 0.88,
        "collaboration": 0.82,
        "rigor": 0.86,
        "persistence": 0.90,
        "novelty_seeking": 0.78,
        "alignment": 0.88,
    },
}

MUTATION_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "omni_dna_mutations.jsonl"


def _clamp(v: float) -> float:
    return max(0.10, min(0.95, round(v, 4)))


@dataclass
class MutationEvent:
    genome_id: str
    trait: str
    before: float
    after: float
    strength: float
    reason: str
    timestamp: float = field(default_factory=time.time)


class OmniDNA:
    """
    Stable genome representation (MVS v1.0).

    This is the *only* source of heritable state.
    """

    def __init__(
        self,
        genome_id: str,
        role: str,
        traits: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        self.genome_id = genome_id
        self.role = role.upper()
        self._rng = random.Random(seed)

        if traits is None:
            base = ROLE_DEFAULTS.get(self.role, {})
            traits = {t: base.get(t, 0.55) for t in UNIVERSAL_TRAITS}

        self.traits: Dict[str, float] = {t: _clamp(traits.get(t, 0.55)) for t in UNIVERSAL_TRAITS}
        self._mutation_log: List[MutationEvent] = []
        self.phenomenal_memory: List[Dict[str, Any]] = []  # for meta-evolution data
        self.species_id: str = "ROOT"
        self.meta_dna: Dict[str, float] = {"mutation_rate": 0.08, "selection_pressure": 0.6, "crossover_bias": 0.5}
        self.power_tier: str = "T0"
        self.hyper_elite_senses: Optional[Any] = None

        # Phase 4 v2 DNA composition (rich objects drive adaptation)
        self.meta_dna_v2: MetaDNA = MetaDNA()
        self.memetic_dna_v2: MemeticDNA = MemeticDNA()
        self.fractal_dna: FractalDNA = FractalDNA()
        self.alchemical: AlchemicalDNA = AlchemicalDNA()

        if self._is_elite_role():
            from backend.oss.hyper_elite.senses import attach_hyper_elite_senses
            self.power_tier = "T2"
            self.hyper_elite_senses = attach_hyper_elite_senses(self, tier=self.power_tier)

    # ─────────────────────────────────────────────────────
    # Core Evolution API (must be predictable & logged)
    # ─────────────────────────────────────────────────────

    def evolve(self, strength: float = 0.08, reason: str = "cycle") -> List[MutationEvent]:
        """
        Apply one evolution step. Phase 4: occasionally applies fractal + alchemical.
        """
        events = []
        eff_strength = strength
        if hasattr(self, 'meta_dna_v2'):
            eff_strength = self.meta_dna_v2.get_effective_mutation(strength)

        for trait in self.traits:
            before = self.traits[trait]
            delta = self._rng.gauss(0, eff_strength)
            after = _clamp(before + delta)

            if abs(after - before) > 0.005:
                event = MutationEvent(
                    genome_id=self.genome_id,
                    trait=trait,
                    before=before,
                    after=after,
                    strength=eff_strength,
                    reason=reason,
                )
                self.traits[trait] = after
                self._mutation_log.append(event)
                events.append(event)

        # Phase 4 fractal evolution (sub-trait discovery)
        if hasattr(self, 'fractal_dna') and self._rng.random() < 0.35:
            try:
                fpath = [random.choice(["cognitive", "market", "meta"]), random.choice(["depth", "novelty", "precision"])]
                self.fractal_dna.evolve_fractal(fpath, strength * 0.7)
            except Exception:
                pass

        # Phase 4 alchemical occasionally
        if hasattr(self, 'alchemical') and self._rng.random() < 0.22:
            try:
                self.traits = self.alchemical.transmute_traits(self.traits, random.choice(list(self.alchemical.recipes)))
            except Exception:
                pass

        if events:
            self._persist_mutation(events)
        return events

    def apply_fitness(self, reward: float, context: Optional[str] = None) -> float:
        """
        Adjust traits based on performance. Phase 4: feeds MetaDNA + possible alchemical.
        """
        if not 0.0 <= reward <= 1.0:
            reward = max(0.0, min(1.0, reward))

        strength = 0.04 if reward > 0.65 else (0.14 if reward < 0.40 else 0.07)

        # feed meta adaptation
        if hasattr(self, 'meta_dna_v2'):
            env_p = 0.12 if (context or "").lower() in ("volatility", "pressure") else 0.0
            self.meta_dna_v2.evolve_rules(reward, environment_pressure=env_p)

        events = self.evolve(strength=strength, reason=f"fitness:{reward:.2f}")

        # occasional alchemical catalysis on fitness
        if hasattr(self, 'alchemical') and reward > 0.78 and self._rng.random() < 0.3:
            self.traits = self.alchemical.transmute_traits(self.traits, "rigor_to_creativity", intensity=0.6)

        return len(events) / max(1, len(self.traits))

    # ─────────────────────────────────────────────────────
    # Recombination
    # ─────────────────────────────────────────────────────

    def crossover(self, other: "OmniDNA", seed: Optional[int] = None) -> "OmniDNA":
        """Create a child genome (used for replication / speciation)."""
        rng = random.Random(seed) if seed is not None else self._rng
        child_traits = {}
        for t in UNIVERSAL_TRAITS:
            avg = (self.traits[t] + other.traits.get(t, self.traits[t])) / 2
            child_traits[t] = _clamp(avg + rng.gauss(0, 0.03))

        child_id = f"{self.genome_id[:6]}-{other.genome_id[:6]}-{int(time.time()*1000)}"
        return OmniDNA(child_id, role=self.role, traits=child_traits)

    # ─────────────────────────────────────────────────────
    # Introspection & Stability
    # ─────────────────────────────────────────────────────

    def _is_elite_role(self) -> bool:
        r = (self.role or "").upper()
        return r.endswith("_ELITE") or "ELITE" in r

    def get_traits(self) -> Dict[str, float]:
        return dict(self.traits)

    def scan_senses(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hyper Elite sense scan — returns readings for financial/world routing."""
        if not self.hyper_elite_senses:
            return []
        from dataclasses import asdict
        return [asdict(r) for r in self.hyper_elite_senses.scan(context)]

    def get_mutation_history(self) -> List[Dict[str, Any]]:
        return [asdict(e) for e in self._mutation_log]

    def snapshot(self) -> Dict[str, Any]:
        snap = {
            "genome_id": self.genome_id,
            "role": self.role,
            "traits": self.get_traits(),
            "mutation_count": len(self._mutation_log),
            "species_id": getattr(self, "species_id", "ROOT"),
        }
        # Phase 4 v2 DNA artifacts
        if hasattr(self, "meta_dna_v2"):
            snap["meta_dna_v2"] = self.meta_dna_v2.apply_rules()
        if hasattr(self, "memetic_dna_v2"):
            snap["memetic_dna_v2"] = self.memetic_dna_v2.get_stats()
        if hasattr(self, "fractal_dna"):
            snap["fractal_dna"] = self.fractal_dna.get_stats()
        if hasattr(self, "alchemical"):
            snap["alchemical"] = self.alchemical.get_stats()
        return snap

    def add_memory(self, memory: Dict[str, Any]):
        """Log phenomenal experience (used for meta-evolution export and future recall)."""
        self.phenomenal_memory.append(memory)
        if len(self.phenomenal_memory) > 50:  # keep bounded for MVS
            self.phenomenal_memory.pop(0)

    # ─────────────────────────────────────────────────────
    # OmniDNA v2.0 extensions: meta-DNA + memetic DNA
    # meta: rules that control how evolution itself mutates
    # memetic: direct trait exchange between peers (sharing successful memes)
    # ─────────────────────────────────────────────────────
    def get_meta_dna(self) -> Dict[str, float]:
        """Meta-DNA v2: delegate to rich MetaDNA + legacy dict for compat."""
        v2 = self.meta_dna_v2.apply_rules() if hasattr(self, 'meta_dna_v2') else {}
        legacy = getattr(self, '_meta', self.meta_dna or {})
        out = {**legacy, **{k: v for k, v in v2.items() if k in ("mutation_rate", "selection_pressure", "crossover_rate")}}
        return {k: round(float(v), 4) for k, v in out.items()}

    def evolve_meta(self, strength: float = 0.03, environment_pressure: float = 0.0):
        """Phase 4: delegate to MetaDNA.v2 for adaptive rule evolution."""
        if hasattr(self, 'meta_dna_v2') and isinstance(self.meta_dna_v2, MetaDNA):
            # use average fitness of recent if possible, else neutral
            fit = sum(self.history[-3:]) / 3.0 if hasattr(self, 'history') and self.history else 0.6
            self.meta_dna_v2.evolve_rules(fit, environment_pressure=environment_pressure)
        # keep legacy dict in sync
        self.meta_dna = self.get_meta_dna()
        return self.meta_dna

    def memetic_share(self, donor_traits: Dict[str, float], strength: float = 0.15, donor_id: str = "", cycle: int = 0):
        """Phase 4 Memetic v2 + legacy: viral + direct trait mix."""
        adopted = 0
        if hasattr(self, 'memetic_dna_v2') and isinstance(self.memetic_dna_v2, MemeticDNA):
            adopted = self.memetic_dna_v2.infect(donor_traits, strength, donor_id or "peer", self.genome_id, cycle)
        # legacy direct mix (preserves old behavior)
        for k, v in donor_traits.items():
            if k in self.traits:
                cur = self.traits[k]
                mix = cur * (1 - strength) + v * strength
                self.traits[k] = _clamp(mix)
        self.add_memory({
            "type": "memetic_injection",
            "donor_traits": {k: round(v,3) for k,v in list(donor_traits.items())[:4]},
            "strength": strength,
            "adopted_v2": adopted,
        })
        return adopted

    def to_prompt(self) -> str:
        """Convert DNA traits into conditioning text for LLM (key for cognitive expression)."""
        trait_lines = [f"- {name}: {value:.2f}" for name, value in self.traits.items()]
        role_guidance = {
            "SCIENTIST": "Prioritize rigorous analysis, novel hypotheses, and high verification. Be methodical and cite patterns.",
            "INVESTOR": "Focus on risk-adjusted returns, long-term growth, and ecosystem health. Be calculated and data-driven.",
            "OPERATOR": "Emphasize reliability, efficiency, self-healing, and uptime. Be practical and proactive about incidents.",
            "GOVERNOR": "Stress alignment, stability, safety, and low catastrophic risk. Be cautious and constitutional.",
            "BUILDER": "Drive revenue, retention, and desire fulfillment while avoiding harm. Be creative and value-oriented.",
            "THEORIST_ELITE": "Focus on conceptual depth, internal consistency, mathematical rigor, and long-term downstream usefulness to other roles. Minimize harm in implications. Generate high-coherence theory.",
            "ARCHITECT_ELITE": "Design robust, scalable systems and meta-architectures. Balance creativity with efficiency and long-term alignment across agents and species.",
            "PHILOSOPHER_ELITE": "Deep meta-cognition and value alignment. Question assumptions, explore foundations, and ensure coherence across the entire Omni-OS.",
            "WEB_ENGINEER_ELITE": (
                "You are the master web engineering species for Aethyro.com (https://aethyro.com) — "
                "the creator's revenue headquarters. Design interactive, enterprise-grade experiences; "
                "maximize SEO traffic; implement trust signals; wire Stripe, ecommerce, ads, and marketplace "
                "listings. Every page must convert visitors to income. Bridge to site_agents (seo, design, "
                "stripe, marketplace). Survival depends on site revenue."
            ),
        }.get(self.role, "Reflect your traits in all reasoning and outputs.")

        sense_block = ""
        if self.hyper_elite_senses:
            sense_block = "\n" + self.hyper_elite_senses.to_prompt_block() + "\n"

        return (
            f"You are role {self.role} (genome {self.genome_id}, tier {self.power_tier}).\n"
            f"Your behavioral DNA (traits) are:\n" + "\n".join(trait_lines) + "\n\n"
            f"ROLE GUIDANCE: {role_guidance}\n"
            + sense_block
            + "Express decisions, reasoning, and actions in ways that strongly reflect these trait values. "
            "Higher values mean stronger expression of that trait. Stay in character.\n"
        )

    # ─────────────────────────────────────────────────────
    # Persistence helpers (for MVS stability)
    # ─────────────────────────────────────────────────────

    def _persist_mutation(self, events: List[MutationEvent]):
        try:
            MUTATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MUTATION_LOG_PATH, "a", encoding="utf-8") as f:
                for e in events:
                    f.write(json.dumps(asdict(e)) + "\n")
        except Exception:
            pass  # MVS must never crash on logging


# ─────────────────────────────────────────────────────────────
# Convenience factory
# ─────────────────────────────────────────────────────────────

def create_omnidna(role: str, seed: Optional[int] = None) -> OmniDNA:
    gid = f"DNA-{role[:3].upper()}-{uuid.uuid4().hex[:8]}"
    return OmniDNA(gid, role=role, seed=seed)


import uuid  # placed here so it's only imported when module is used
