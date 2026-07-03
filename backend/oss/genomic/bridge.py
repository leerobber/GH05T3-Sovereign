"""
Bridge between Genomic schema and MVS OmniDNA (trait dict layer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

from .hyper_elite import create_hyper_elite_psychology_genome
from .schema import Genome

if TYPE_CHECKING:
    from backend.oss.omni_dna import OmniDNA

# Map genomic molecules → MVS UNIVERSAL_TRAITS keys
MOLECULE_TO_TRAIT: Dict[str, str] = {
    "M_REASONING_DEPTH": "rigor",
    "M_PATTERN_RECOGNITION": "pattern_detection",
    "M_LEARNING_RATE": "novelty_seeking",
    "M_ADAPTABILITY": "creativity",
    "M_RISK_TOLERANCE": "risk_tolerance",
    "M_TREND_DETECTION": "market_intuition",
    "M_PRICING_INTUITION": "efficiency",
    "M_CURIOSITY_TRIGGER": "novelty_seeking",
    "M_TRUST_TONE": "alignment",
    "M_EMOTIONAL_CHARGE": "empathy",
    "M_VALUE_FRAMING": "market_intuition",
    "M_COHERENCE": "rigor",
    "M_MEMORABILITY": "creativity",
    "M_CONSISTENCY": "persistence",
    "M_ALIGNMENT": "alignment",
}


def sync_genome_to_omni_dna(genome: Genome, dna: "OmniDNA") -> int:
    """Push molecule values into OmniDNA traits. Returns count updated."""
    updated = 0
    for locus in genome.loci.values():
        for mid, mol in locus.molecules.items():
            trait = MOLECULE_TO_TRAIT.get(mid)
            if trait and trait in dna.traits:
                dna.traits[trait] = max(dna.traits[trait], mol.get_value())
                updated += 1
    dna.power_tier = genome.metadata.get("power_tier", getattr(dna, "power_tier", "T0"))
    return updated


def genome_from_omni_dna(dna: "OmniDNA") -> Genome:
    """Create a genomic layer snapshot from existing OmniDNA."""
    genome = create_hyper_elite_psychology_genome(role=dna.role)
    genome.lineage_id = dna.genome_id
    traits = dna.get_traits()
    reverse = {v: k for k, v in MOLECULE_TO_TRAIT.items()}
    for trait_name, value in traits.items():
        mid = reverse.get(trait_name)
        if mid:
            for locus in genome.loci.values():
                mol = locus.get_molecule(mid)
                if mol:
                    mol.set_value(float(value))
    return genome