"""
Context Genome — context-awareness as a genetic trait (Phase 3 / Level 5-8 upgrade).

Context depth, efficiency, compression, synthesis, and anticipation are encoded as
molecules in a CONTEXT locus. The pressure engine selects agents that use context
better — not just agents that produce better raw output.

8-Level Context Maturity (from framework):
  1-2  SURFACE    (value 0.00–0.25) — you are the context, tab-complete era
  3-4  SHALLOW    (value 0.25–0.50) — curated context, rules files, CLAUDE.md
  5-6  MODERATE   (value 0.50–0.75) — context layer, MCP, harness engineering
  7-8  DEEP       (value 0.75–1.00) — background agents, autonomous handoff
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional

from .schema import Genome, Locus, LocusType, MutationStrategy, create_molecule


# ── Enum representations of context traits ───────────────────────────────────

class ContextDepth(Enum):
    SURFACE  = 0  # levels 1-2
    SHALLOW  = 1  # levels 3-4
    MODERATE = 2  # levels 5-6
    DEEP     = 3  # levels 7-8

    @classmethod
    def from_value(cls, v: float) -> "ContextDepth":
        if v < 0.25:  return cls.SURFACE
        if v < 0.50:  return cls.SHALLOW
        if v < 0.75:  return cls.MODERATE
        return cls.DEEP


class ContextEfficiency(Enum):
    LOW     = 0  # wastes context, repeats work
    MEDIUM  = 1  # uses context but misses opportunities
    HIGH    = 2  # maximises context utility
    OPTIMAL = 3  # extracts max value from minimal context

    @classmethod
    def from_value(cls, v: float) -> "ContextEfficiency":
        if v < 0.25:  return cls.LOW
        if v < 0.50:  return cls.MEDIUM
        if v < 0.75:  return cls.HIGH
        return cls.OPTIMAL


class ContextMaturityLevel(Enum):
    LEVEL_1 = 1   # Unaware: no context usage
    LEVEL_2 = 2   # Reactive: uses context only when forced
    LEVEL_3 = 3   # Basic: surface-level context
    LEVEL_4 = 4   # Proactive: seeks context when needed
    LEVEL_5 = 5   # Composable: combines multiple sources
    LEVEL_6 = 6   # Optimised: uses context efficiently
    LEVEL_7 = 7   # Predictive: anticipates needed context
    LEVEL_8 = 8   # Autonomous: self-optimises context usage


# ── Context molecule catalog ──────────────────────────────────────────────────

CONTEXT_MOLECULES: Dict[str, Dict[str, Any]] = {
    # How deep the agent reasons about context (surface → deep)
    "M_CONTEXT_DEPTH": {
        "name": "Context Depth",
        "alleles": {
            "default": {"value": 0.50, "dominance": "ADDITIVE"},
            "deep":    {"value": 0.85, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_CONTEXT_SYNTHESIS", "M_CONTEXT_ANTICIPATION"],
    },
    # How efficiently the agent utilises available context
    "M_CONTEXT_EFFICIENCY": {
        "name": "Context Efficiency",
        "alleles": {
            "default": {"value": 0.50, "dominance": "ADDITIVE"},
            "optimal": {"value": 0.90, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.ADAPTIVE,
        "interaction_map": ["M_CONTEXT_COMPRESSION", "M_CONTEXT_DEPTH"],
    },
    # Extracts maximum meaning from sparse / minimal context
    "M_CONTEXT_COMPRESSION": {
        "name": "Context Compression",
        "alleles": {
            "default": {"value": 0.50, "dominance": "ADDITIVE"},
            "expert":  {"value": 0.80, "dominance": "CO_DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_CONTEXT_EFFICIENCY"],
    },
    # Cross-source synthesis (Levels 5-6 trait)
    "M_CONTEXT_SYNTHESIS": {
        "name": "Context Synthesis",
        "alleles": {
            "default": {"value": 0.40, "dominance": "ADDITIVE"},
            "multi_source": {"value": 0.85, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_CONTEXT_DEPTH", "M_CONTEXT_ANTICIPATION"],
    },
    # Predicts what context will be needed before acting (Levels 7-8)
    "M_CONTEXT_ANTICIPATION": {
        "name": "Context Anticipation",
        "alleles": {
            "default":    {"value": 0.30, "dominance": "ADDITIVE"},
            "predictive": {"value": 0.90, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.04,
        "mutation_strategy": MutationStrategy.ADAPTIVE,
        "interaction_map": ["M_CONTEXT_SYNTHESIS", "M_CONTEXT_DEPTH"],
    },
    # How quickly stale context loses weight (higher = faster decay)
    "M_CONTEXT_DECAY": {
        "name": "Context Decay Rate",
        "alleles": {
            "default": {"value": 0.50, "dominance": "ADDITIVE"},
            "slow":    {"value": 0.20, "dominance": "CO_DOMINANT"},
        },
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.GAUSSIAN,
        "interaction_map": [],
    },
}


# ── Locus factory ─────────────────────────────────────────────────────────────

def create_context_locus(weight: float = 1.1) -> Locus:
    """Build a CONTEXT locus populated with all context molecules."""
    locus = Locus(name="context", type=LocusType.CONTEXT, weight=weight)
    for mid, spec in CONTEXT_MOLECULES.items():
        mol = create_molecule(
            molecule_id=mid,
            name=spec["name"],
            locus_type=LocusType.CONTEXT,
            alleles=spec.get("alleles"),
            mutation_rate=spec.get("mutation_rate", 0.02),
            mutation_strategy=spec.get("mutation_strategy", MutationStrategy.ADAPTIVE),
            interaction_map=spec.get("interaction_map"),
        )
        locus.add_molecule(mol)
    return locus


# ── Maturity scoring ──────────────────────────────────────────────────────────

def score_context_maturity(genome: Genome) -> int:
    """
    Read the CONTEXT locus and return a maturity level 1-8.

    Combines depth, efficiency, synthesis, and anticipation into a
    single normalised score, then maps to the 8-level framework.
    """
    ctx = genome.loci.get("context")
    if ctx is None:
        return 1  # no context locus → LEVEL_1

    depth        = ctx.molecules.get("M_CONTEXT_DEPTH",       None)
    efficiency   = ctx.molecules.get("M_CONTEXT_EFFICIENCY",  None)
    synthesis    = ctx.molecules.get("M_CONTEXT_SYNTHESIS",   None)
    anticipation = ctx.molecules.get("M_CONTEXT_ANTICIPATION",None)

    d = depth.get_value()        if depth        else 0.3
    e = efficiency.get_value()   if efficiency   else 0.3
    s = synthesis.get_value()    if synthesis    else 0.2
    a = anticipation.get_value() if anticipation else 0.1

    combined = d * 0.30 + e * 0.30 + s * 0.20 + a * 0.20
    level = max(1, min(8, round(combined * 8)))
    return level


def context_maturity_enum(genome: Genome) -> ContextMaturityLevel:
    return ContextMaturityLevel(score_context_maturity(genome))


# ── Fitness feedback ──────────────────────────────────────────────────────────

def update_context_fitness(genome: Genome, hit_rate: float, task_success: float = 0.5) -> None:
    """
    Reinforce context molecules when context was actually useful.

    hit_rate: 0-1, fraction of injected context that contributed to the output
    task_success: 0-1, task outcome quality
    """
    ctx = genome.loci.get("context")
    if ctx is None:
        return
    signal = (hit_rate * 0.6 + task_success * 0.4)
    for mol in ctx.molecules.values():
        current = mol.get_value()
        # Positive signal → nudge up; negative → nudge down
        delta = (signal - 0.5) * mol.mutation_rate * 2
        mol.set_value(current + delta)


# ── Context efficiency metric for fitness signal ──────────────────────────────

def context_tier_for_loyalty(loyalty_level: int) -> Dict[str, Any]:
    """
    Returns the context access tier for a given LoyaltyLevel int.
    Tiers gate which context locus molecules are readable / writable by the agent.

    loyalty_level corresponds to LoyaltyLevel enum values:
        0 = NOVICE, 1 = TRUSTED_SPECIALIST, 2 = HYPER_ELITE, 3 = ARCHITECT
    """
    _TIERS: Dict[int, Dict[str, Any]] = {
        0: {
            "tier":       "surface",
            "readable":   ["M_CONTEXT_DEPTH", "M_CONTEXT_EFFICIENCY"],
            "writable":   [],
            "max_depth":  3,
            "description": "Read-only surface context. No cross-source synthesis.",
        },
        1: {
            "tier":       "shallow",
            "readable":   ["M_CONTEXT_DEPTH", "M_CONTEXT_EFFICIENCY",
                           "M_CONTEXT_COMPRESSION", "M_CONTEXT_SYNTHESIS"],
            "writable":   ["M_CONTEXT_EFFICIENCY"],
            "max_depth":  5,
            "description": "Context assembly + compression. Can tune efficiency molecule.",
        },
        2: {
            "tier":       "moderate",
            "readable":   ["M_CONTEXT_DEPTH", "M_CONTEXT_EFFICIENCY",
                           "M_CONTEXT_COMPRESSION", "M_CONTEXT_SYNTHESIS",
                           "M_CONTEXT_ANTICIPATION"],
            "writable":   ["M_CONTEXT_EFFICIENCY", "M_CONTEXT_COMPRESSION",
                           "M_CONTEXT_SYNTHESIS"],
            "max_depth":  7,
            "description": "Full context locus read. Anticipation molecule active.",
        },
        3: {
            "tier":       "deep",
            "readable":   list(CONTEXT_MOLECULES.keys()),
            "writable":   list(CONTEXT_MOLECULES.keys()),
            "max_depth":  8,
            "description": "Autonomous context: full read+write, decay control.",
        },
    }
    return _TIERS.get(loyalty_level, _TIERS[0])


def context_efficiency_score(genome: Genome) -> float:
    """
    Returns 0-1 representing how well this genome uses context.
    Used as the `context_efficiency` key in metrics passed to Genome.calculate_fitness().
    """
    ctx = genome.loci.get("context")
    if ctx is None:
        return 0.3
    eff = ctx.molecules.get("M_CONTEXT_EFFICIENCY")
    comp = ctx.molecules.get("M_CONTEXT_COMPRESSION")
    syn = ctx.molecules.get("M_CONTEXT_SYNTHESIS")
    e = eff.get_value()  if eff  else 0.3
    c = comp.get_value() if comp else 0.3
    s = syn.get_value()  if syn  else 0.3
    return round(e * 0.5 + c * 0.25 + s * 0.25, 4)
