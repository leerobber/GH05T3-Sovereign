from .meta_dna import MetaDNA
from .memetic_dna import Meme, MemeticDNA
from .fractal_dna import FractalDNA
from .alchemical_dna import AlchemicalDNA

__all__ = ["MetaDNA", "Meme", "MemeticDNA", "FractalDNA", "AlchemicalDNA"]

def get_dna_v2_suite():
    """Factory for a full v2 suite (used by integrations)."""
    return {
        "meta": MetaDNA(),
        "memetic": MemeticDNA(),
        "fractal": FractalDNA(),
        "alchemical": AlchemicalDNA(),
    }