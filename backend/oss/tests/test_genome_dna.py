"""Tests for the genome DNA layer: encoding round-trip, GenomePlane, and
the real mutation/selection operators -- Stage 1 of the genome subsystem
rebuild (see backend/oss/core/ for the stage-1 bookkeeping layer).
"""
from __future__ import annotations

import random

import pytest

from backend.oss.dna.genome_encoding import decode_genome, encode_genome
from backend.oss.dna.genome_plane import Genome, GenomePlane
from backend.oss.dna.mutation_operators import BinaryRatioJitterMutation, QuantModeMutation, StabilizerSwitchMutation
from backend.oss.dna.selection_strategies import HighestScoreSelection, LowestLatencySelection


def test_encode_decode_round_trip_preserves_full_identity():
    data = encode_genome("g1", {"binary_ratio": 0.95, "stabilizer": "mgc"}, {"parent": None})
    genome_id, traits, metadata = decode_genome(data)

    assert genome_id == "g1"
    assert traits == {"binary_ratio": 0.95, "stabilizer": "mgc"}
    assert metadata == {"parent": None}


def test_decode_rejects_malformed_data():
    with pytest.raises(ValueError):
        decode_genome(b"not valid json at all")

    with pytest.raises(ValueError):
        decode_genome(b'{"traits": {}}')  # missing required "id" field


def test_genome_plane_encode_decode_round_trip_via_the_plane():
    """The actual correctness gate: encode a real registered genome,
    decode it back through the SAME plane, and confirm it reconstructs
    to an equal Genome -- not just that the two free functions round-trip
    in isolation."""
    plane = GenomePlane()
    plane.add_genome("g1", traits={"binary_ratio": 0.9, "stabilizer": "damg"}, metadata={"note": "baseline"})

    encoded = plane.encode("g1")

    fresh_plane = GenomePlane()
    restored = fresh_plane.decode(encoded)

    assert restored == Genome(id="g1", traits={"binary_ratio": 0.9, "stabilizer": "damg"}, metadata={"note": "baseline"})
    assert fresh_plane.get_genome("g1") == restored


def test_apply_mutation_creates_a_new_genome_without_touching_the_base():
    plane = GenomePlane()
    base = plane.add_genome("base", traits={"binary_ratio": 0.9, "stabilizer": "mgc"})

    op = BinaryRatioJitterMutation(step=0.05, rng=random.Random(42))
    mutation = op.create_mutation(base, perf={})
    mutated = plane.apply_mutation(mutation)

    assert mutated.id != base.id
    assert mutated.traits["binary_ratio"] != base.traits["binary_ratio"]
    # Base genome must be untouched -- mutation creates a new genome, it
    # doesn't mutate in place.
    assert plane.get_genome("base").traits["binary_ratio"] == 0.9
    assert mutated.metadata["parent"] == "base"


def test_binary_ratio_jitter_stays_within_bounds_and_is_not_applicable_without_the_trait():
    op = BinaryRatioJitterMutation(step=0.5, min_ratio=0.5, max_ratio=1.0, rng=random.Random(1))
    genome_at_ceiling = Genome(id="g", traits={"binary_ratio": 1.0})

    assert op.is_applicable(genome_at_ceiling, {})
    mutation = op.create_mutation(genome_at_ceiling, {})
    new_traits = mutation.apply(genome_at_ceiling.traits)
    assert new_traits["binary_ratio"] <= 1.0  # large step must still clamp, not overshoot

    genome_without_trait = Genome(id="g2", traits={"stabilizer": "mgc"})
    assert not op.is_applicable(genome_without_trait, {})


def test_stabilizer_switch_flips_between_known_values_only():
    op = StabilizerSwitchMutation()
    genome = Genome(id="g", traits={"stabilizer": "mgc"})

    assert op.is_applicable(genome, {})
    mutation = op.create_mutation(genome, {})
    assert mutation.apply(genome.traits)["stabilizer"] == "damg"

    genome_unknown = Genome(id="g2", traits={"stabilizer": "something_fake"})
    assert not op.is_applicable(genome_unknown, {})


def test_quant_mode_mutation_flips_between_known_values_only():
    op = QuantModeMutation()
    genome = Genome(id="g", traits={"out_proj_quant_mode": "ternary"})

    assert op.is_applicable(genome, {})
    mutation = op.create_mutation(genome, {})
    assert mutation.apply(genome.traits)["out_proj_quant_mode"] == "binary"

    genome_reverse = Genome(id="g2", traits={"out_proj_quant_mode": "binary"})
    assert op.create_mutation(genome_reverse, {}).apply(genome_reverse.traits)["out_proj_quant_mode"] == "ternary"

    genome_unknown = Genome(id="g3", traits={"out_proj_quant_mode": "something_fake"})
    assert not op.is_applicable(genome_unknown, {})

    genome_missing = Genome(id="g4", traits={})
    assert not op.is_applicable(genome_missing, {})


def test_highest_score_selection_picks_the_real_max():
    strategy = HighestScoreSelection()
    scores = {"a": {"score": 0.5}, "b": {"score": 0.9}, "c": {"score": 0.1}}
    assert strategy.select(scores) == "b"


def test_highest_score_selection_rejects_empty_scores():
    with pytest.raises(ValueError):
        HighestScoreSelection().select({})


def test_lowest_latency_selection_respects_min_score_threshold():
    strategy = LowestLatencySelection(min_score=0.5)
    scores = {
        "fast_but_bad": {"score": 0.2, "latency": 0.001},
        "slow_but_good": {"score": 0.9, "latency": 0.5},
        "fast_and_good": {"score": 0.6, "latency": 0.05},
    }
    # fast_but_bad is excluded by the score floor even though it's fastest.
    assert strategy.select(scores) == "fast_and_good"


def test_lowest_latency_selection_raises_when_nothing_meets_threshold():
    strategy = LowestLatencySelection(min_score=0.99)
    with pytest.raises(ValueError):
        strategy.select({"a": {"score": 0.5, "latency": 0.01}})
