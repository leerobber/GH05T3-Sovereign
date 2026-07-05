"""Registry + encoding layer for genomes. GenomePlane.encode/decode is a
real round trip (see genome_encoding.py) -- the original rebuild spec's
decode() raised NotImplementedError, which would make "back up and
restore a genome" a one-way trip that silently loses data the moment
anyone actually exercises the restore half.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.oss.dna.genome_encoding import decode_genome, encode_genome
from backend.oss.dna.mutation_operators import Mutation


@dataclass
class Genome:
    id: str
    traits: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class GenomePlane:
    def __init__(self) -> None:
        self._genomes: dict[str, Genome] = {}

    def add_genome(self, genome_id: str, traits: dict[str, Any], metadata: dict[str, Any] | None = None) -> Genome:
        genome = Genome(id=genome_id, traits=dict(traits), metadata=dict(metadata or {}))
        self._genomes[genome.id] = genome
        return genome

    def list_genomes(self) -> list[Genome]:
        return list(self._genomes.values())

    def get_genome(self, genome_id: str) -> Genome:
        return self._genomes[genome_id]

    def has_genome(self, genome_id: str) -> bool:
        return genome_id in self._genomes

    def apply_mutation(self, mutation: Mutation) -> Genome:
        base = self._genomes[mutation.base_id]
        new_traits = mutation.apply(base.traits)
        new_genome = Genome(
            id=mutation.new_id,
            traits=new_traits,
            metadata={"parent": base.id, "mutation": mutation.description},
        )
        self._genomes[new_genome.id] = new_genome
        return new_genome

    def encode(self, genome_id: str) -> bytes:
        genome = self._genomes[genome_id]
        return encode_genome(genome.id, genome.traits, genome.metadata)

    def decode(self, data: bytes) -> Genome:
        """Real inverse of encode(): reconstructs and registers the
        genome from encoded bytes. Overwrites any existing genome with
        the same id -- a decode is meant to restore a genome's exact
        recorded state, not merge with whatever is already registered.
        """
        genome_id, traits, metadata = decode_genome(data)
        genome = Genome(id=genome_id, traits=traits, metadata=metadata)
        self._genomes[genome.id] = genome
        return genome
