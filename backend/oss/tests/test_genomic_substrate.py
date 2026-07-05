"""End-to-end test for GenomicSubstrate: the real correctness gate for
the whole genome subsystem -- registers genomes, evaluates them with a
real forward pass, mutates the underperforming ones based on real
recorded scores, evaluates the mutants, and routes a task to the real
best genome. No stubbed scores anywhere in this path.
"""
from __future__ import annotations

from backend.oss.core.chronos_ledger import ChronosLedger
from backend.oss.core.genomic_substrate import GenomicSubstrate
from backend.oss.core.omni_evolution import OmniEvolutionEngine
from backend.oss.core.species_memory import SpeciesMemory
from backend.oss.dna.genome_plane import GenomePlane
from backend.oss.dna.mutation_operators import StabilizerSwitchMutation
from backend.oss.swarm.swarm_runtime import SwarmRuntime


def _small_substrate(score_threshold: float = float("inf")) -> GenomicSubstrate:
    # score_threshold=inf by default -- every genome with a recorded
    # score is a mutation candidate, so tests don't depend on real loss
    # magnitudes happening to land on one side of an arbitrary cutoff.
    return GenomicSubstrate(
        genome_plane=GenomePlane(),
        evolution_engine=OmniEvolutionEngine([StabilizerSwitchMutation()], score_threshold=score_threshold),
        species_memory=SpeciesMemory(),
        chronos_ledger=ChronosLedger(),
        swarm_runtime=SwarmRuntime(seq_len=8, batch_size=2),
    )


_TINY_TRAITS = {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 16, "stabilizer": "mgc"}


def test_register_and_evaluate_all_records_real_scores():
    substrate = _small_substrate()
    substrate.register_initial_genomes([("g1", dict(_TINY_TRAITS))])

    substrate.evaluate_all()

    stats = substrate.chronos_ledger.get_recent_stats(["g1"])
    assert "g1" in stats
    assert stats["g1"]["loss"] > 0.0
    assert substrate.species_memory.scores_snapshot()["g1"] == stats["g1"]


def test_evolution_cycle_mutates_genomes_with_recorded_history_and_evaluates_them():
    substrate = _small_substrate()
    substrate.register_initial_genomes([("g1", dict(_TINY_TRAITS))])
    substrate.evaluate_all()  # g1 now has a real recorded score

    new_genomes = substrate.run_evolution_cycle()

    assert len(new_genomes) == 1
    mutant = new_genomes[0]
    assert mutant.metadata["parent"] == "g1"
    assert mutant.traits["stabilizer"] == "damg"  # flipped from the base's "mgc"

    # The mutant must have its OWN real recorded score, not the parent's.
    mutant_stats = substrate.chronos_ledger.get_recent_stats([mutant.id])
    assert mutant.id in mutant_stats
    assert mutant_stats[mutant.id]["loss"] > 0.0


def test_evolution_cycle_skips_genomes_with_no_recorded_history():
    substrate = _small_substrate()
    substrate.register_initial_genomes([("g1", dict(_TINY_TRAITS))])
    # Deliberately NOT calling evaluate_all() -- g1 has no recorded score.

    new_genomes = substrate.run_evolution_cycle()

    assert new_genomes == []


def test_route_task_returns_the_real_best_genome():
    substrate = _small_substrate()
    substrate.register_initial_genomes([
        ("g1", dict(_TINY_TRAITS)),
        ("g2", {**_TINY_TRAITS, "num_layers": 3}),
    ])
    substrate.evaluate_all()

    best = substrate.route_task(task={})

    scores = substrate.species_memory.scores_snapshot()
    best_by_score = max(scores.items(), key=lambda kv: kv[1]["score"])[0]
    assert best.id == best_by_score
