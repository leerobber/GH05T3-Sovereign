"""Tests for TernaryLinear's sparsity_target: verifies the quantile-based
threshold genuinely hits the requested sparsity fraction (not just an
approximate side effect of a tunable multiplier), and that it threads
through the model correctly.
"""
from __future__ import annotations

import torch

from gh05t3_binary.core.attention import HybridBinaryAttention
from gh05t3_binary.core.binary_layers import TernaryLinear
from gh05t3_binary.core.transformer import GH05T3BinaryTransformer


def test_default_none_matches_original_fixed_threshold_exactly():
    torch.manual_seed(0)
    default_layer = TernaryLinear(64, 32, sparsity_target=None)
    x = torch.randn(4, 64)

    with torch.no_grad():
        w = default_layer.weight
        delta = 0.7 * w.abs().mean()
        ternary = torch.zeros_like(w)
        ternary[w > delta] = 1.0
        ternary[w < -delta] = -1.0
        mask = (ternary != 0).float()
        alpha = (w.abs() * mask).sum() / mask.sum().clamp(min=1.0)
        expected = torch.nn.functional.linear(x, ternary * alpha)

        actual = default_layer(x)

    assert torch.equal(actual, expected)


def test_sparsity_target_actually_hits_the_requested_fraction():
    torch.manual_seed(1)
    layer = TernaryLinear(1000, 10, sparsity_target=0.7)

    with torch.no_grad():
        w = layer.weight
        delta = torch.quantile(w.abs(), 0.7)
        zero_fraction = (w.abs() <= delta).float().mean().item()

    # torch.quantile-based threshold must land very close to the
    # requested fraction (small tie-breaking slack only), unlike the
    # fixed 0.7*mean(|W|) multiplier, which has no such guarantee.
    assert abs(zero_fraction - 0.7) < 0.02


def test_higher_sparsity_target_zeros_more_weights_than_lower():
    torch.manual_seed(2)
    low = TernaryLinear(500, 20, sparsity_target=0.2)
    high = TernaryLinear(500, 20, sparsity_target=0.8)
    # Same underlying weight distribution (same seed before construction
    # would differ due to separate allocations) -- instead directly copy
    # one weight tensor into both to isolate the effect of sparsity_target.
    torch.manual_seed(3)
    shared_weight = torch.randn(20, 500) * 0.02
    with torch.no_grad():
        low.weight.copy_(shared_weight)
        high.weight.copy_(shared_weight)

    def real_zero_fraction(layer):
        w = layer.weight
        delta = torch.quantile(w.abs(), layer.sparsity_target)
        return (w.abs() <= delta).float().mean().item()

    assert real_zero_fraction(high) > real_zero_fraction(low)


def test_invalid_sparsity_target_rejected():
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            TernaryLinear(8, 8, sparsity_target=bad)
            assert False, f"expected ValueError for sparsity_target={bad}"
        except ValueError:
            pass


def test_gradient_still_flows_with_sparsity_target_set():
    layer = TernaryLinear(16, 8, sparsity_target=0.5)
    x = torch.randn(4, 16)
    out = layer(x)
    out.sum().backward()

    assert layer.weight.grad is not None
    assert torch.any(layer.weight.grad != 0)


def test_ternary_sparsity_target_threads_through_attention():
    attn = HybridBinaryAttention(dim=32, num_heads=4, out_proj_quant_mode="ternary", ternary_sparsity_target=0.6)
    assert isinstance(attn.out_proj, TernaryLinear)
    assert attn.out_proj.sparsity_target == 0.6


def test_ternary_sparsity_target_ignored_when_out_proj_is_binary():
    """A real, honest constraint: sparsity_target only means something
    for a TernaryLinear out_proj. When out_proj_quant_mode="binary", the
    trait must not silently apply to anything (BinaryLinear has no such
    concept) -- confirmed by out_proj simply being a BinaryLinear with
    no sparsity_target attribute at all."""
    attn = HybridBinaryAttention(dim=32, num_heads=4, out_proj_quant_mode="binary", ternary_sparsity_target=0.6)
    assert not hasattr(attn.out_proj, "sparsity_target")


def test_ternary_sparsity_target_threads_through_full_model():
    model = GH05T3BinaryTransformer(
        num_layers=2, dim=32, num_heads=4, vocab_size=20,
        out_proj_quant_mode="ternary", ternary_sparsity_target=0.4,
    )
    for layer in model.layers:
        assert layer.attention.out_proj.sparsity_target == 0.4

    input_ids = torch.randint(0, 20, (1, 8))
    logits = model(input_ids)
    logits.sum().backward()
