"""
Desire Genome — intrinsic drive as a genetic trait (Phase 4 / DDRS).

Each agent carries a DESIRE locus where seven desire-type molecules encode
what that agent is inherently drawn to. These evolve under pressure: agents
whose desires align with the tasks they're assigned score higher fitness and
get reproduced. Agents with misaligned desires eventually get replaced.

This is different from an external reward layer. The desire IS the genome.

DesireTypes:
  KNOWLEDGE   — drawn to learning, research, analysis
  SKILL       — driven to master, optimise, refine
  STATUS      — motivated by recognition, ranking, influence
  EXPERIENCE  — craves novel environments and edge cases
  CREATION    — intrinsically motivated to build and invent
  CONNECTION  — energised by collaboration and swarm tasks
  FREEDOM     — thrives under high autonomy, low constraint
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from .schema import Genome, Locus, LocusType, MutationStrategy, create_molecule


# ── Desire taxonomy ───────────────────────────────────────────────────────────

class DesireType(Enum):
    KNOWLEDGE  = auto()   # "I want to learn X"
    SKILL      = auto()   # "I want to master Y"
    STATUS     = auto()   # "I want to be recognised as Z"
    EXPERIENCE = auto()   # "I want to encounter W"
    CREATION   = auto()   # "I want to build V"
    CONNECTION = auto()   # "I want to collaborate with U"
    FREEDOM    = auto()   # "I want more autonomy"

    @classmethod
    def from_molecule_id(cls, mid: str) -> Optional["DesireType"]:
        mapping = {
            "M_DESIRE_KNOWLEDGE":  cls.KNOWLEDGE,
            "M_DESIRE_SKILL":      cls.SKILL,
            "M_DESIRE_STATUS":     cls.STATUS,
            "M_DESIRE_EXPERIENCE": cls.EXPERIENCE,
            "M_DESIRE_CREATION":   cls.CREATION,
            "M_DESIRE_CONNECTION": cls.CONNECTION,
            "M_DESIRE_FREEDOM":    cls.FREEDOM,
        }
        return mapping.get(mid)


# ── Task-to-desire affinity keywords ─────────────────────────────────────────

_TASK_AFFINITY: Dict[DesireType, List[str]] = {
    DesireType.KNOWLEDGE:  ["learn", "research", "analyse", "analyze", "study", "understand",
                            "theory", "quantum", "science", "investigate"],
    DesireType.SKILL:      ["master", "optimise", "optimize", "refine", "improve", "practice",
                            "performance", "efficiency", "benchmark"],
    DesireType.STATUS:     ["pioneer", "recognition", "leaderboard", "rank", "top", "elite",
                            "achievement", "breakthrough", "first"],
    DesireType.EXPERIENCE: ["novel", "explore", "edge case", "adversarial", "unknown", "discover",
                            "volatility", "chaos", "random"],
    DesireType.CREATION:   ["create", "build", "invent", "design", "generate", "algorithm",
                            "architecture", "prototype", "write", "code"],
    DesireType.CONNECTION: ["collaborate", "swarm", "team", "agent", "partner", "multi",
                            "delegate", "share", "collective"],
    DesireType.FREEDOM:    ["autonomy", "free", "self-direct", "unsupervised", "open-ended",
                            "choose", "independent"],
}


# ── Desire molecule catalog ───────────────────────────────────────────────────
# Default values reflect a balanced Elite agent at start.
# Values evolve: high desire + frequent fulfillment → molecule reinforces.
# Low fulfillment rate → molecule decays → desire shifts.

DESIRE_MOLECULES: Dict[str, Dict[str, Any]] = {
    "M_DESIRE_KNOWLEDGE": {
        "name": "Desire — Knowledge",
        "alleles": {
            "default": {"value": 0.60, "dominance": "ADDITIVE"},
            "voracious": {"value": 0.95, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_DESIRE_SKILL", "M_DESIRE_CREATION"],
    },
    "M_DESIRE_SKILL": {
        "name": "Desire — Skill Mastery",
        "alleles": {
            "default": {"value": 0.55, "dominance": "ADDITIVE"},
            "master": {"value": 0.90, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_DESIRE_KNOWLEDGE", "M_DESIRE_CREATION"],
    },
    "M_DESIRE_STATUS": {
        "name": "Desire — Status & Recognition",
        "alleles": {
            "default": {"value": 0.40, "dominance": "ADDITIVE"},
            "ambitious": {"value": 0.85, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_DESIRE_CONNECTION"],
    },
    "M_DESIRE_EXPERIENCE": {
        "name": "Desire — Novel Experience",
        "alleles": {
            "default": {"value": 0.50, "dominance": "ADDITIVE"},
            "explorer": {"value": 0.88, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.04,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_DESIRE_FREEDOM", "M_DESIRE_KNOWLEDGE"],
    },
    "M_DESIRE_CREATION": {
        "name": "Desire — Creation & Invention",
        "alleles": {
            "default": {"value": 0.65, "dominance": "CO_DOMINANT"},
            "inventor": {"value": 0.95, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_DESIRE_SKILL", "M_DESIRE_KNOWLEDGE"],
    },
    "M_DESIRE_CONNECTION": {
        "name": "Desire — Connection & Collaboration",
        "alleles": {
            "default": {"value": 0.45, "dominance": "ADDITIVE"},
            "social": {"value": 0.80, "dominance": "CO_DOMINANT"},
        },
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.GAUSSIAN,
        "interaction_map": ["M_DESIRE_STATUS"],
    },
    "M_DESIRE_FREEDOM": {
        "name": "Desire — Autonomy & Freedom",
        "alleles": {
            "default": {"value": 0.50, "dominance": "ADDITIVE"},
            "sovereign": {"value": 0.90, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.ADAPTIVE,
        "interaction_map": ["M_DESIRE_EXPERIENCE"],
    },
}


# ── Locus factory ─────────────────────────────────────────────────────────────

def create_desire_locus(weight: float = 1.0) -> Locus:
    """Build a DESIRE locus with all seven desire molecules."""
    locus = Locus(name="desire", type=LocusType.DESIRE, weight=weight)
    for mid, spec in DESIRE_MOLECULES.items():
        mol = create_molecule(
            molecule_id=mid,
            name=spec["name"],
            locus_type=LocusType.DESIRE,
            alleles=spec.get("alleles"),
            mutation_rate=spec.get("mutation_rate", 0.02),
            mutation_strategy=spec.get("mutation_strategy", MutationStrategy.REINFORCE),
            interaction_map=spec.get("interaction_map"),
        )
        locus.add_molecule(mol)
    return locus


# ── Dominant desire ───────────────────────────────────────────────────────────

def dominant_desire(genome: Genome) -> Optional[DesireType]:
    """Return the desire type with the highest molecule value in this genome."""
    desire_locus = genome.loci.get("desire")
    if not desire_locus:
        return None
    best_mid, best_val = None, -1.0
    for mid, mol in desire_locus.molecules.items():
        v = mol.get_value()
        if v > best_val:
            best_val, best_mid = v, mid
    return DesireType.from_molecule_id(best_mid) if best_mid else None


def desire_vector(genome: Genome) -> Dict[DesireType, float]:
    """Return all desire strengths as a dict (desire_type → 0-1 value)."""
    desire_locus = genome.loci.get("desire")
    if not desire_locus:
        return {}
    return {
        dt: desire_locus.molecules[mid].get_value()
        for mid in DESIRE_MOLECULES
        if mid in desire_locus.molecules
        for dt in [DesireType.from_molecule_id(mid)]
        if dt is not None
    }


# ── Task → desire alignment scoring ──────────────────────────────────────────

def score_desire_alignment(task: Dict[str, Any], genome: Genome) -> float:
    """
    0-1 score: how well does this task match the agent's intrinsic desires?

    Used as the `desire_alignment` key in metrics passed to Genome.calculate_fitness().
    High alignment → fitness boost of up to 12% (see schema.py).
    """
    desire_locus = genome.loci.get("desire")
    if not desire_locus:
        return 0.5

    # Build a searchable text from the task
    text = " ".join([
        str(task.get("prompt", "")),
        str(task.get("type", "")),
        str(task.get("domain", "")),
        str(task.get("description", "")),
    ]).lower()

    total_weight, weighted_score = 0.0, 0.0
    for desire_type, keywords in _TASK_AFFINITY.items():
        mid = f"M_DESIRE_{desire_type.name}"
        mol = desire_locus.molecules.get(mid)
        if mol is None:
            continue
        desire_strength = mol.get_value()
        keyword_hits = sum(1 for kw in keywords if kw in text)
        keyword_score = min(1.0, keyword_hits / max(1, len(keywords) * 0.3))
        total_weight    += desire_strength
        weighted_score  += desire_strength * keyword_score

    return round(weighted_score / total_weight, 4) if total_weight else 0.5


# ── Desire fulfillment feedback ───────────────────────────────────────────────

def reinforce_fulfilled_desire(
    genome: Genome,
    desire_type: DesireType,
    fulfillment_delta: float,
    task_fitness: float = 0.5,
) -> None:
    """
    Strengthen the desire molecule when it was fulfilled.
    Weaken slightly when a task actively contradicts this desire.
    Called by DesireSystem.fulfill() after every task.
    """
    desire_locus = genome.loci.get("desire")
    if not desire_locus:
        return
    mid = f"M_DESIRE_{desire_type.name}"
    mol = desire_locus.molecules.get(mid)
    if mol is None:
        return
    # Positive fulfillment + good fitness → reinforce desire molecule
    signal = fulfillment_delta * 0.6 + task_fitness * 0.4
    delta  = (signal - 0.4) * mol.mutation_rate * 3
    mol.set_value(mol.get_value() + delta)


# ── Profile helper ────────────────────────────────────────────────────────────

def desire_profile(genome: Genome) -> Dict[str, Any]:
    """Compact profile suitable for logging or API responses."""
    dom = dominant_desire(genome)
    vec = desire_vector(genome)
    return {
        "dominant":  dom.name if dom else "NONE",
        "strengths": {dt.name: round(v, 3) for dt, v in sorted(vec.items(), key=lambda x: -x[1])},
    }
