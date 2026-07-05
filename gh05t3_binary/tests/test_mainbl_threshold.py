"""Tests for MagnitudeAwareINBL's mag_threshold gate and its threading
through HybridBinaryAttention/BinaryTransformerBlock/
GH05T3BinaryTransformer/GH05T3BinaryOSS.
"""
from __future__ import annotations

import torch

from gh05t3_binary.core.attention import HybridBinaryAttention
from gh05t3_binary.core.binary_layers import MagnitudeAwareINBL
from gh05t3_binary.core.transformer import GH05T3BinaryTransformer


def test_default_threshold_is_disabled_and_matches_original_behavior():
    """mag_threshold=0.0 (the default) must reproduce the exact original
    forward computation -- no gate applied at all -- so the already
    trained checkpoint's behavior is unaffected unless a genome opts in."""
    torch.manual_seed(0)
    layer = MagnitudeAwareINBL(in_features=16, out_features=8, mag_threshold=0.0)
    x = torch.randn(4, 16) * torch.tensor([[0.001], [1.0], [10.0], [0.0001]])

    with torch.no_grad():
        mag = x.norm(dim=-1, keepdim=True)
        direction = x / (mag + 1e-8)
        y_dir = layer.binary_linear(direction)
        mag_q = layer.mag_quantizer(mag)
        expected = y_dir * mag_q * layer.denorm

        actual = layer(x)

    assert torch.equal(actual, expected)


def test_positive_threshold_suppresses_small_magnitude_and_passes_large_magnitude():
    torch.manual_seed(1)
    layer = MagnitudeAwareINBL(in_features=16, out_features=8, mag_threshold=1.0)

    small_mag_x = torch.randn(1, 16)
    small_mag_x = small_mag_x / small_mag_x.norm() * 0.1  # norm = 0.1, below threshold

    large_mag_x = torch.randn(1, 16)
    large_mag_x = large_mag_x / large_mag_x.norm() * 5.0  # norm = 5.0, above threshold

    with torch.no_grad():
        small_out = layer(small_mag_x)
        large_out = layer(large_mag_x)

    assert torch.equal(small_out, torch.zeros_like(small_out))
    assert not torch.equal(large_out, torch.zeros_like(large_out))


def test_gate_does_not_block_gradient_flow_to_learnable_parameters():
    layer = MagnitudeAwareINBL(in_features=16, out_features=8, mag_threshold=0.5)
    x = torch.randn(4, 16)
    # Ensure at least one row is above threshold so gradient has something
    # real to flow through.
    x[0] = x[0] / x[0].norm() * 5.0

    out = layer(x)
    out.sum().backward()

    assert layer.binary_linear.weight.grad is not None
    assert torch.any(layer.binary_linear.weight.grad != 0)
    assert layer.denorm.grad is not None
    assert torch.any(layer.denorm.grad != 0)


def test_mainbl_threshold_threads_through_attention_to_all_three_projections():
    attn = HybridBinaryAttention(dim=32, num_heads=4, mainbl_threshold=0.7)
    assert attn.q_proj.mag_threshold == 0.7
    assert attn.k_proj.mag_threshold == 0.7
    assert attn.v_proj.mag_threshold == 0.7


def test_mainbl_threshold_threads_through_full_model():
    model = GH05T3BinaryTransformer(
        num_layers=2, dim=32, num_heads=4, vocab_size=20, mainbl_threshold=0.3,
    )
    assert model.mainbl_threshold == 0.3
    for layer in model.layers:
        assert layer.attention.q_proj.mag_threshold == 0.3
        assert layer.attention.k_proj.mag_threshold == 0.3
        assert layer.attention.v_proj.mag_threshold == 0.3

    # Real forward/backward pass with gating enabled must still work.
    input_ids = torch.randint(0, 20, (1, 8))
    logits = model(input_ids)
    logits.sum().backward()
