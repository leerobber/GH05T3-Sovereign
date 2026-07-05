"""Tests for BMEBridge: proves apply_genome_to_engine actually constructs
a real, differently-shaped GH05T3BinaryOSS per genome (not a config
struct that nothing downstream reads), that re-evaluating the SAME
genome_id is deterministic and cached, and that DIFFERENT genome_ids
stay independent even when their traits happen to be identical.
"""
from __future__ import annotations

from gh05t3_binary.core.binary_layers import BinaryLinear, TernaryLinear

from backend.oss.core.bme_bridge import BMEBridge


def test_apply_genome_to_engine_builds_real_model_with_requested_shape():
    bridge = BMEBridge()
    model = bridge.apply_genome_to_engine(
        "g1", {"num_layers": 3, "dim": 64, "num_heads": 4, "vocab_size": 50, "stabilizer": "mgc"}
    )

    assert len(model.transformer.layers) == 3
    assert model.transformer.dim == 64
    assert model.transformer.embedding.num_embeddings == 50


def test_different_genomes_produce_differently_shaped_real_models():
    bridge = BMEBridge()
    small = bridge.apply_genome_to_engine("small", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20})
    big = bridge.apply_genome_to_engine("big", {"num_layers": 4, "dim": 64, "num_heads": 4, "vocab_size": 20})

    assert len(small.transformer.layers) != len(big.transformer.layers)
    assert small.transformer.dim != big.transformer.dim


def test_stabilizer_trait_actually_selects_the_real_stabilizer():
    bridge = BMEBridge()
    mgc_model = bridge.apply_genome_to_engine(
        "g-mgc", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "stabilizer": "mgc"}
    )
    damg_model = bridge.apply_genome_to_engine(
        "g-damg", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "stabilizer": "damg"}
    )

    assert mgc_model.transformer.stabilizer == "mgc"
    assert damg_model.transformer.stabilizer == "damg"


def test_out_proj_quant_mode_trait_actually_selects_the_real_layer_type():
    """The whole point of QuantModeMutation: the trait must genuinely
    change which quantized layer type backs out_proj, not just be a
    dict key nothing downstream reads."""
    bridge = BMEBridge()
    ternary_model = bridge.apply_genome_to_engine(
        "g-ternary", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "out_proj_quant_mode": "ternary"}
    )
    binary_model = bridge.apply_genome_to_engine(
        "g-binary", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "out_proj_quant_mode": "binary"}
    )

    assert isinstance(ternary_model.transformer.layers[0].attention.out_proj, TernaryLinear)
    assert isinstance(binary_model.transformer.layers[0].attention.out_proj, BinaryLinear)
    assert not isinstance(binary_model.transformer.layers[0].attention.out_proj, TernaryLinear)


def test_mainbl_threshold_trait_actually_configures_the_real_gate():
    """The whole point of MainblThresholdMutation: the trait must
    genuinely reach MagnitudeAwareINBL's mag_threshold, not sit unused
    in a dict."""
    bridge = BMEBridge()
    model = bridge.apply_genome_to_engine(
        "g-threshold", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "mainbl_threshold": 0.42}
    )
    attn = model.transformer.layers[0].attention
    assert attn.q_proj.mag_threshold == 0.42
    assert attn.k_proj.mag_threshold == 0.42
    assert attn.v_proj.mag_threshold == 0.42


def test_missing_traits_fall_back_to_documented_defaults():
    bridge = BMEBridge()
    model = bridge.apply_genome_to_engine("g-defaults", {})

    assert len(model.transformer.layers) == 4
    assert model.transformer.dim == 256
    assert model.transformer.mainbl_threshold == 0.0  # gating disabled by default
    assert model.transformer.embedding.num_embeddings == 4096  # _DEFAULT_VOCAB_SIZE


def test_repeat_calls_for_the_same_genome_id_return_the_identical_cached_object():
    bridge = BMEBridge()
    traits = {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 10}

    first = bridge.apply_genome_to_engine("g1", traits)
    second = bridge.apply_genome_to_engine("g1", traits)

    assert first is second


def test_different_genome_ids_with_identical_traits_get_independent_random_weights():
    """The caching is keyed by genome_id, not by a hash of traits --
    two DISTINCT genomes that happen to share identical trait values
    must still be independent evolutionary samples, not silently
    collapsed onto the same weights."""
    bridge = BMEBridge()
    traits = {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 10}

    genome_a = bridge.apply_genome_to_engine("genome-a", traits)
    genome_b = bridge.apply_genome_to_engine("genome-b", traits)

    assert genome_a is not genome_b
    weight_a = genome_a.transformer.embedding.weight
    weight_b = genome_b.transformer.embedding.weight
    assert not (weight_a == weight_b).all()


def test_same_genome_id_gives_deterministic_weights_across_separate_bridge_instances():
    """Reproducibility must survive a fresh BMEBridge (e.g. a process
    restart), not just an in-memory cache hit -- confirms the seeding is
    derived from genome_id itself, not incidental object identity."""
    traits = {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 10}

    model_first_process = BMEBridge().apply_genome_to_engine("stable-id", traits)
    model_second_process = BMEBridge().apply_genome_to_engine("stable-id", traits)

    weight_a = model_first_process.transformer.embedding.weight
    weight_b = model_second_process.transformer.embedding.weight
    assert (weight_a == weight_b).all()
