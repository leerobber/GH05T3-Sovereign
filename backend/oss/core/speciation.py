"""Groups genomes into species by a real signature over specific
architectural traits (num_layers, dim, stabilizer) -- genomes sharing
these traits are the same species, genomes differing on any of them are
not. Replaces the original spec's SpeciationEngine, which mapped every
genome to the same hardcoded "default_species" string regardless of
content -- not real clustering, just a placeholder that always agreed
with itself.
"""
from __future__ import annotations

from typing import Any

_DEFAULT_SPECIES_TRAITS = ("num_layers", "dim", "stabilizer")


class SpeciationEngine:
    def __init__(self, species_traits: tuple[str, ...] = _DEFAULT_SPECIES_TRAITS) -> None:
        self.species_traits = species_traits

    def species_signature(self, traits: dict[str, Any]) -> str:
        parts = [f"{key}={traits.get(key, '?')}" for key in self.species_traits]
        return "|".join(parts)

    def assign_species(self, genomes: list) -> dict[str, str]:
        return {genome.id: self.species_signature(genome.traits) for genome in genomes}
