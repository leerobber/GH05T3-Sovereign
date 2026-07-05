"""Top-level orchestrator composing the genome subsystem's real pieces:
GenomePlane (dna/genome_plane.py), OmniEvolutionEngine, SpeciesMemory,
ChronosLedger, and BMEBridge/SwarmRuntime (the real bridge to
gh05t3_binary). Each run_evolution_cycle() call is a real, measurable
step: read real recorded performance, decide mutations, apply them,
evaluate the results with a real forward pass, and record the real
outcome -- no stubbed scores anywhere in this path.
"""
from __future__ import annotations

from typing import Any

from backend.oss.core.chronos_ledger import ChronosLedger
from backend.oss.core.omni_evolution import OmniEvolutionEngine
from backend.oss.core.species_memory import SpeciesMemory
from backend.oss.dna.genome_plane import Genome, GenomePlane
from backend.oss.swarm.swarm_runtime import SwarmRuntime


class GenomicSubstrate:
    def __init__(
        self,
        genome_plane: GenomePlane,
        evolution_engine: OmniEvolutionEngine,
        species_memory: SpeciesMemory,
        chronos_ledger: ChronosLedger,
        swarm_runtime: SwarmRuntime,
    ):
        self.genome_plane = genome_plane
        self.evolution_engine = evolution_engine
        self.species_memory = species_memory
        self.chronos_ledger = chronos_ledger
        self.swarm_runtime = swarm_runtime

    def register_initial_genomes(self, genomes: list[tuple[str, dict[str, Any]]]) -> None:
        """genomes: (genome_id, traits) pairs -- matches GenomePlane.add_genome's
        real signature directly rather than a separate GenomeConfig
        wrapper class duplicating Genome's fields."""
        for genome_id, traits in genomes:
            self.genome_plane.add_genome(genome_id, traits)

    def evaluate_all(self) -> None:
        """Real evaluation pass: every registered genome gets a real
        SwarmRuntime.evaluate_genome() score, recorded to both the full
        history (ChronosLedger) and the latest-score view (SpeciesMemory,
        used for selection)."""
        for genome in self.genome_plane.list_genomes():
            result = self.swarm_runtime.evaluate_genome(genome.traits)
            self.chronos_ledger.record_result(genome.id, result)
            self.species_memory.update(genome.id, result)

    def run_evolution_cycle(self) -> list[Genome]:
        """One real evolution step: propose mutations from real recorded
        performance, apply and evaluate each with a real forward pass,
        record the real outcome. Returns the genomes actually created
        this cycle (empty if no genome was a mutation candidate)."""
        genomes = self.genome_plane.list_genomes()
        genome_ids = [g.id for g in genomes]
        stats = self.chronos_ledger.get_recent_stats(genome_ids)

        mutations = self.evolution_engine.propose_mutations(genomes, stats)

        new_genomes = []
        for mutation in mutations:
            new_genome = self.genome_plane.apply_mutation(mutation)
            result = self.swarm_runtime.evaluate_genome(new_genome.traits)
            self.chronos_ledger.record_result(new_genome.id, result)
            self.species_memory.update(new_genome.id, result)
            new_genomes.append(new_genome)

        return new_genomes

    def route_task(self, task: dict[str, Any]) -> Genome:
        """Selects and returns the real best genome for a task, per
        SpeciesMemory's configured selection strategy."""
        genome_id = self.species_memory.select_best_genome(task)
        return self.genome_plane.get_genome(genome_id)
