"""
Hyper-Elite Psychology Genome — Attention, Language, Desire, Aesthetic genes.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List

from .schema import (
    Genome,
    Locus,
    LocusType,
    MutationStrategy,
    create_molecule,
)

PSYCHOLOGY_MOLECULES: Dict[str, Dict[str, Any]] = {
    "M_VISUAL_SALIENCE": {
        "name": "Visual Salience",
        "alleles": {"default": {"value": 0.7, "dominance": "CO_DOMINANT"}, "high_contrast": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_COLOR_VALENCE", "M_LAYOUT_FLOW"],
    },
    "M_COLOR_VALENCE": {
        "name": "Color Valence",
        "alleles": {"default": {"value": 0.6, "dominance": "ADDITIVE"}, "warm_colors": {"value": 0.8, "dominance": "DOMINANT"}},
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.GAUSSIAN,
        "interaction_map": ["M_VISUAL_SALIENCE", "M_EMOTIONAL_CHARGE", "M_PREMIUM_FEEL"],
    },
    "M_LAYOUT_FLOW": {
        "name": "Layout Flow",
        "alleles": {"default": {"value": 0.8, "dominance": "DOMINANT"}, "f_pattern": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_VISUAL_SALIENCE", "M_MEMORABILITY"],
    },
    "M_CURIOSITY_TRIGGER": {
        "name": "Curiosity Trigger",
        "alleles": {"default": {"value": 0.5, "dominance": "ADDITIVE"}, "high_open_loop": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.05,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_TRUST_TONE", "M_SCARCITY_SIGNAL", "M_LAYOUT_FLOW", "M_MEMORABILITY"],
    },
    "M_TRUST_TONE": {
        "name": "Trust Tone",
        "alleles": {"default": {"value": 0.5, "dominance": "ADDITIVE"}, "high_authority": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_CURIOSITY_TRIGGER", "M_EMOTIONAL_CHARGE"],
    },
    "M_EMOTIONAL_CHARGE": {
        "name": "Emotional Charge",
        "alleles": {"default": {"value": 0.7, "dominance": "CO_DOMINANT"}, "high_joy": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.04,
        "mutation_strategy": MutationStrategy.GAUSSIAN,
        "interaction_map": ["M_COLOR_VALENCE", "M_TRUST_TONE"],
    },
    "M_SCARCITY_SIGNAL": {
        "name": "Scarcity Signal",
        "alleles": {"default": {"value": 0.6, "dominance": "ADDITIVE"}, "high_urgency": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_VALUE_FRAMING", "M_CURIOSITY_TRIGGER"],
    },
    "M_VALUE_FRAMING": {
        "name": "Value Framing",
        "alleles": {"default": {"value": 0.7, "dominance": "ADDITIVE"}, "high_roi": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_SCARCITY_SIGNAL", "M_IDENTITY_ALIGNMENT"],
    },
    "M_IDENTITY_ALIGNMENT": {
        "name": "Identity Alignment",
        "alleles": {"default": {"value": 0.8, "dominance": "DOMINANT"}, "high_personalization": {"value": 0.95, "dominance": "DOMINANT"}},
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.GAUSSIAN,
        "interaction_map": ["M_VALUE_FRAMING", "M_PREMIUM_FEEL"],
    },
    "M_PREMIUM_FEEL": {
        "name": "Premium Feel",
        "alleles": {"default": {"value": 0.5, "dominance": "CO_DOMINANT"}, "luxury": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_IDENTITY_ALIGNMENT", "M_COHERENCE"],
    },
    "M_COHERENCE": {
        "name": "Coherence",
        "alleles": {"default": {"value": 0.8, "dominance": "DOMINANT"}, "high_consistency": {"value": 0.95, "dominance": "DOMINANT"}},
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_PREMIUM_FEEL", "M_MEMORABILITY"],
    },
    "M_MEMORABILITY": {
        "name": "Memorability",
        "alleles": {"default": {"value": 0.7, "dominance": "ADDITIVE"}, "high_storytelling": {"value": 0.9, "dominance": "DOMINANT"}},
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.GAUSSIAN,
        "interaction_map": ["M_LAYOUT_FLOW", "M_COHERENCE"],
    },
}

COGNITIVE_MOLECULES: Dict[str, Dict[str, Any]] = {
    "M_REASONING_DEPTH": {"name": "Reasoning Depth", "alleles": {"default": {"value": 0.7}}, "mutation_rate": 0.01, "mutation_strategy": MutationStrategy.REINFORCE},
    "M_PATTERN_RECOGNITION": {"name": "Pattern Recognition", "alleles": {"default": {"value": 0.8}}, "mutation_rate": 0.02, "mutation_strategy": MutationStrategy.GAUSSIAN},
    "M_LEARNING_RATE": {"name": "Learning Rate", "alleles": {"default": {"value": 0.7}}, "mutation_rate": 0.03, "mutation_strategy": MutationStrategy.ADAPTIVE},
    "M_ADAPTABILITY": {"name": "Adaptability", "alleles": {"default": {"value": 0.8}}, "mutation_rate": 0.02, "mutation_strategy": MutationStrategy.CONTEXTUAL},
}

MARKET_MOLECULES: Dict[str, Dict[str, Any]] = {
    "M_RISK_TOLERANCE": {"name": "Risk Tolerance", "alleles": {"default": {"value": 0.5}}, "mutation_rate": 0.03, "mutation_strategy": MutationStrategy.CONTEXTUAL},
    "M_TREND_DETECTION": {"name": "Trend Detection", "alleles": {"default": {"value": 0.7}}, "mutation_rate": 0.02, "mutation_strategy": MutationStrategy.GAUSSIAN},
    "M_PRICING_INTUITION": {"name": "Pricing Intuition", "alleles": {"default": {"value": 0.6}}, "mutation_rate": 0.01, "mutation_strategy": MutationStrategy.REINFORCE},
}

LOYALTY_MOLECULES: Dict[str, Dict[str, Any]] = {
    "M_CONSISTENCY": {"name": "Consistency", "alleles": {"default": {"value": 0.6}}, "mutation_rate": 0.01, "mutation_strategy": MutationStrategy.REINFORCE},
    "M_CONTRIBUTION": {"name": "Contribution", "alleles": {"default": {"value": 0.5}}, "mutation_rate": 0.01, "mutation_strategy": MutationStrategy.REINFORCE},
    "M_ALIGNMENT": {"name": "Alignment", "alleles": {"default": {"value": 0.7}}, "mutation_rate": 0.01, "mutation_strategy": MutationStrategy.GAUSSIAN},
}


def _build_locus(name: str, locus_type: LocusType, catalog: Dict[str, Dict[str, Any]], weight: float = 1.0) -> Locus:
    locus = Locus(name=name, type=locus_type, weight=weight)
    for mid, spec in catalog.items():
        mol = create_molecule(
            molecule_id=mid,
            name=spec["name"],
            locus_type=locus_type,
            alleles=spec.get("alleles"),
            mutation_rate=spec.get("mutation_rate", 0.01),
            mutation_strategy=spec.get("mutation_strategy", MutationStrategy.GAUSSIAN),
            interaction_map=spec.get("interaction_map"),
        )
        locus.add_molecule(mol)
    return locus


def create_hyper_elite_psychology_genome(seed: int | None = None, role: str = "hyper_elite_psychologist") -> Genome:
    """Factory for Hyper-Elite Psychology agents (Aethyro web/revenue species)."""
    from .context_genome import create_context_locus   # local import avoids circular at module level
    from .desire_genome import create_desire_locus

    if seed is not None:
        random.seed(seed)

    lineage = f"HELIX-{role[:4].upper()}-{uuid.uuid4().hex[:8]}"
    genome = Genome(
        lineage_id=lineage,
        role=role,
        loci={
            "psychology": _build_locus("psychology", LocusType.PSYCHOLOGY, PSYCHOLOGY_MOLECULES, weight=1.2),
            "cognitive":  _build_locus("cognitive",  LocusType.COGNITIVE,  COGNITIVE_MOLECULES,  weight=1.0),
            "market":     _build_locus("market",     LocusType.MARKET,     MARKET_MOLECULES,     weight=0.9),
            "loyalty":    _build_locus("loyalty",    LocusType.LOYALTY,    LOYALTY_MOLECULES,    weight=0.8),
            "context":    create_context_locus(weight=1.1),   # Phase 3 — context as genetic trait
            "desire":     create_desire_locus(weight=1.0),    # Phase 4 — DDRS intrinsic drives
        },
        mutation_map={
            "psychology": {"mutation_rate_boost": 0.02, "context_sensitivity": True, "novelty_boost": 0.75},
            "cognitive":  {"mutation_rate_boost": 0.01, "reinforcement_learning": True},
            "market":     {"mutation_rate_boost": 0.015},
            "context":    {"mutation_rate_boost": 0.02, "reinforcement_learning": True, "context_driven": True},
            "desire":     {"mutation_rate_boost": 0.02, "reinforcement_learning": True, "desire_driven": True},
        },
        metadata={"species": "HYPER_ELITE_PSYCHOLOGY", "power_tier": "T2"},
    )
    genome.apply_interactions()
    return genome