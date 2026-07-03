"""
Omni-Sentient Genomic Layer — alleles, molecules, loci, evolution, loyalty, novelty.

Sits alongside MVS OmniDNA (trait dict); use bridge.sync_genome_to_omni_dna() to unify.
"""

from .schema import (
    Allele,
    Molecule,
    Locus,
    Genome,
    MutationRule,
    DominanceType,
    MutationStrategy,
    LocusType,
    create_molecule,
)
from .hyper_elite import create_hyper_elite_psychology_genome, PSYCHOLOGY_MOLECULES
from .novelty import NoveltyRewardEngine, NoveltyScore
from .loyalty import LoyaltySystem, LoyaltyLevel, LoyaltyMetrics
from .evolution_tuner import EvolutionTuner
from .agents import Agent, AgentSwarm
from .ecosystem import OmniSentientEcosystem
from .bridge import sync_genome_to_omni_dna, genome_from_omni_dna

__all__ = [
    "Allele",
    "Molecule",
    "Locus",
    "Genome",
    "MutationRule",
    "DominanceType",
    "MutationStrategy",
    "LocusType",
    "create_molecule",
    "create_hyper_elite_psychology_genome",
    "PSYCHOLOGY_MOLECULES",
    "NoveltyRewardEngine",
    "NoveltyScore",
    "LoyaltySystem",
    "LoyaltyLevel",
    "LoyaltyMetrics",
    "EvolutionTuner",
    "Agent",
    "AgentSwarm",
    "OmniSentientEcosystem",
    "sync_genome_to_omni_dna",
    "genome_from_omni_dna",
]