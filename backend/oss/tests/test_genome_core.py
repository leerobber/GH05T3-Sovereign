"""Tests for the genome bookkeeping layer: ChronosLedger, SpeciesMemory,
SpeciationEngine -- Stage 1 of the genome subsystem rebuild.
"""
from __future__ import annotations

import pytest

from backend.oss.core.chronos_ledger import ChronosLedger
from backend.oss.core.speciation import SpeciationEngine
from backend.oss.core.species_memory import SpeciesMemory
from backend.oss.dna.genome_plane import Genome
from backend.oss.dna.selection_strategies import LowestLatencySelection


def test_chronos_ledger_returns_only_most_recent_result_per_genome():
    ledger = ChronosLedger()
    ledger.record_result("g1", {"score": 0.5})
    ledger.record_result("g1", {"score": 0.8})
    ledger.record_result("g2", {"score": 0.3})

    stats = ledger.get_recent_stats(["g1", "g2", "g3-no-history"])

    assert stats["g1"] == {"score": 0.8}
    assert stats["g2"] == {"score": 0.3}
    # A genome with no recorded results must be omitted, not padded with
    # a fabricated zero score.
    assert "g3-no-history" not in stats


def test_chronos_ledger_history_for_returns_full_ordered_history():
    ledger = ChronosLedger()
    ledger.record_result("g1", {"score": 0.1})
    ledger.record_result("g1", {"score": 0.2})

    assert ledger.history_for("g1") == [{"score": 0.1}, {"score": 0.2}]
    assert ledger.history_for("never-recorded") == []


def test_species_memory_default_strategy_picks_highest_score():
    memory = SpeciesMemory()
    memory.update("a", {"score": 0.3})
    memory.update("b", {"score": 0.7})

    assert memory.select_best_genome(task={}) == "b"


def test_species_memory_accepts_a_different_strategy():
    memory = SpeciesMemory(strategy=LowestLatencySelection(min_score=0.5))
    memory.update("a", {"score": 0.9, "latency": 1.0})
    memory.update("b", {"score": 0.6, "latency": 0.1})

    assert memory.select_best_genome(task={}) == "b"


def test_speciation_groups_genomes_by_shared_architectural_traits():
    engine = SpeciationEngine()
    g1 = Genome(id="g1", traits={"num_layers": 12, "dim": 1024, "stabilizer": "mgc"})
    g2 = Genome(id="g2", traits={"num_layers": 12, "dim": 1024, "stabilizer": "mgc"})
    g3 = Genome(id="g3", traits={"num_layers": 8, "dim": 512, "stabilizer": "damg"})

    species = engine.assign_species([g1, g2, g3])

    # g1 and g2 share every species trait -- must land in the same species.
    assert species["g1"] == species["g2"]
    # g3 differs on all three -- must be a different species.
    assert species["g3"] != species["g1"]


def test_speciation_signature_is_real_not_a_constant():
    """Regression guard against the original spec's SpeciationEngine,
    which returned the literal string "default_species" for every genome
    regardless of content."""
    engine = SpeciationEngine()
    sig_a = engine.species_signature({"num_layers": 4, "dim": 256, "stabilizer": "mgc"})
    sig_b = engine.species_signature({"num_layers": 8, "dim": 512, "stabilizer": "damg"})

    assert sig_a != sig_b
    assert sig_a != "default_species"
