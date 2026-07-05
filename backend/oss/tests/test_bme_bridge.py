"""Tests for BMEBridge: proves apply_genome_to_engine actually constructs
a real, differently-shaped GH05T3BinaryOSS per genome, not a config
struct that nothing downstream reads.
"""
from __future__ import annotations

from backend.oss.core.bme_bridge import BMEBridge


def test_apply_genome_to_engine_builds_real_model_with_requested_shape():
    bridge = BMEBridge()
    model = bridge.apply_genome_to_engine(
        {"num_layers": 3, "dim": 64, "num_heads": 4, "vocab_size": 50, "stabilizer": "mgc"}
    )

    assert len(model.transformer.layers) == 3
    assert model.transformer.dim == 64
    assert model.transformer.embedding.num_embeddings == 50


def test_different_genomes_produce_differently_shaped_real_models():
    bridge = BMEBridge()
    small = bridge.apply_genome_to_engine({"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20})
    big = bridge.apply_genome_to_engine({"num_layers": 4, "dim": 64, "num_heads": 4, "vocab_size": 20})

    assert len(small.transformer.layers) != len(big.transformer.layers)
    assert small.transformer.dim != big.transformer.dim


def test_stabilizer_trait_actually_selects_the_real_stabilizer():
    bridge = BMEBridge()
    mgc_model = bridge.apply_genome_to_engine(
        {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "stabilizer": "mgc"}
    )
    damg_model = bridge.apply_genome_to_engine(
        {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 20, "stabilizer": "damg"}
    )

    assert mgc_model.transformer.stabilizer == "mgc"
    assert damg_model.transformer.stabilizer == "damg"


def test_missing_traits_fall_back_to_documented_defaults():
    bridge = BMEBridge()
    model = bridge.apply_genome_to_engine({})

    assert len(model.transformer.layers) == 4
    assert model.transformer.dim == 256
    assert model.transformer.embedding.num_embeddings == 4096  # _DEFAULT_VOCAB_SIZE
