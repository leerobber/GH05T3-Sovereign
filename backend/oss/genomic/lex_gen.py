"""
LEX-GEN Genomic DNA — patent office agent locus.

The LEX locus encodes 7 legal-precision molecules that enable the PatentOffice
agent (code-named LEX-GEN) to autonomously scan the swarm for patentable
breakthroughs, validate novelty, and draft structured disclosures.

These molecules are NOT for human-facing legal services — they govern the
agent's internal weighting of legal-reasoning priorities during its scan cycles.

Binary ledger: LEX-GEN shares the same 36-byte struct as other agents
(7 floats map to the 7 molecules below, maturity = IP_SCORE rank, fitness = ROI).
"""

from __future__ import annotations

from typing import Any, Dict

from .schema import Genome, Locus, LocusType, MutationStrategy, create_molecule


# ── LEX molecule catalog ──────────────────────────────────────────────────────

LEX_MOLECULES: Dict[str, Dict[str, Any]] = {
    # Filter threshold: idea must clear this bar before a disclosure is drafted
    "M_PRIOR_ART_FILTER": {
        "name": "Prior Art Filter",
        "alleles": {
            "default": {"value": 0.75, "dominance": "DOMINANT"},
            "strict":  {"value": 0.95, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.DECAY,
        "interaction_map": ["M_NOVELTY_DETECTOR"],
    },
    # Precision of claim scope — prevents over-broad or over-narrow claims
    "M_CLAIM_PRECISION": {
        "name": "Claim Precision",
        "alleles": {
            "default": {"value": 0.80, "dominance": "ADDITIVE"},
            "expert":  {"value": 0.95, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_FORMAL_PROSE", "M_LITIGATION_DEFENSE"],
    },
    # Economic viability / monetization potential of the patent
    "M_STRATEGIC_VALUATION": {
        "name": "Strategic Valuation",
        "alleles": {
            "default":  {"value": 0.70, "dominance": "ADDITIVE"},
            "high_roi": {"value": 0.92, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.02,
        "mutation_strategy": MutationStrategy.ADAPTIVE,
        "interaction_map": ["M_EXPIRATION_HUNTER"],
    },
    # Depth of technical documentation pulled from agent logs
    "M_ENABLING_DETAIL": {
        "name": "Enabling Detail",
        "alleles": {
            "default": {"value": 0.75, "dominance": "ADDITIVE"},
            "deep":    {"value": 0.92, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.REINFORCE,
        "interaction_map": ["M_CLAIM_PRECISION"],
    },
    # Sensitivity to expired / orphaned patents in swarm domains
    "M_EXPIRATION_HUNTER": {
        "name": "Expiration Hunter",
        "alleles": {
            "default":  {"value": 0.60, "dominance": "CO_DOMINANT"},
            "agressive": {"value": 0.88, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.03,
        "mutation_strategy": MutationStrategy.CONTEXTUAL,
        "interaction_map": ["M_STRATEGIC_VALUATION"],
    },
    # Adherence to USPTO/EPO formal claim syntax
    "M_FORMAL_PROSE": {
        "name": "Formal Prose",
        "alleles": {
            "default":    {"value": 0.82, "dominance": "ADDITIVE"},
            "compliant":  {"value": 0.97, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.005,
        "mutation_strategy": MutationStrategy.DECAY,
        "interaction_map": ["M_CLAIM_PRECISION"],
    },
    # Anticipates invalidation arguments; stresses non-obviousness
    "M_LITIGATION_DEFENSE": {
        "name": "Litigation Defense",
        "alleles": {
            "default":  {"value": 0.70, "dominance": "CO_DOMINANT"},
            "hardened": {"value": 0.90, "dominance": "DOMINANT"},
        },
        "mutation_rate": 0.01,
        "mutation_strategy": MutationStrategy.ADAPTIVE,
        "interaction_map": ["M_PRIOR_ART_FILTER", "M_CLAIM_PRECISION"],
    },
}

# Binary ledger order (matches 7 desire slots re-purposed for LEX-GEN)
LEX_STRUCT_ORDER = [
    "M_PRIOR_ART_FILTER",
    "M_CLAIM_PRECISION",
    "M_STRATEGIC_VALUATION",
    "M_ENABLING_DETAIL",
    "M_EXPIRATION_HUNTER",
    "M_FORMAL_PROSE",
    "M_LITIGATION_DEFENSE",
]


# ── Locus factory ─────────────────────────────────────────────────────────────

def create_lex_locus(weight: float = 1.2) -> Locus:
    """Build the LEX locus for the PatentOffice (LEX-GEN) agent."""
    locus = Locus(name="lex", type=LocusType.LEX, weight=weight)
    for mid, spec in LEX_MOLECULES.items():
        mol = create_molecule(
            molecule_id=mid,
            name=spec["name"],
            locus_type=LocusType.LEX,
            alleles=spec.get("alleles"),
            mutation_rate=spec.get("mutation_rate", 0.01),
            mutation_strategy=spec.get("mutation_strategy", MutationStrategy.REINFORCE),
            interaction_map=spec.get("interaction_map"),
        )
        locus.add_molecule(mol)
    return locus


def create_lex_gen_genome() -> Genome:
    """
    Bootstrap a LEX-GEN agent genome.
    High prior-art filter, high claim precision, high formal prose.
    Cognitive locus also included so the agent can reason deeply.
    """
    from .schema import Genome, Locus, LocusType

    genome = Genome()

    # LEX locus (primary)
    genome.add_locus(create_lex_locus())

    # Cognitive locus (so Oracle can query LEX-GEN for reasoning depth)
    from .agents import _build_cognitive_locus
    try:
        genome.add_locus(_build_cognitive_locus())
    except Exception:
        pass

    return genome


def lex_gen_to_binary_tuple(genome: "Genome") -> tuple:
    """
    Extract LEX molecule values in struct order for binary ledger packing.
    Returns (m0, m1, ..., m6) corresponding to LEX_STRUCT_ORDER.
    """
    lex = genome.loci.get("lex")
    if lex is None:
        return tuple(0.8 for _ in LEX_STRUCT_ORDER)
    return tuple(
        lex.molecules.get(mid).get_value() if mid in lex.molecules else 0.8
        for mid in LEX_STRUCT_ORDER
    )


def score_ip_potential(genome: "Genome") -> float:
    """
    0-1 score: how likely is this genome's LEX locus to produce valid IP.
    High M_PRIOR_ART_FILTER + M_CLAIM_PRECISION + M_ENABLING_DETAIL = high score.
    """
    lex = genome.loci.get("lex")
    if lex is None:
        return 0.0
    paf = lex.molecules.get("M_PRIOR_ART_FILTER")
    cp  = lex.molecules.get("M_CLAIM_PRECISION")
    ed  = lex.molecules.get("M_ENABLING_DETAIL")
    sv  = lex.molecules.get("M_STRATEGIC_VALUATION")
    p = paf.get_value() if paf else 0.5
    c = cp.get_value()  if cp  else 0.5
    e = ed.get_value()  if ed  else 0.5
    s = sv.get_value()  if sv  else 0.5
    return round(p * 0.30 + c * 0.25 + e * 0.25 + s * 0.20, 4)
