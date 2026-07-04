"""Round-trip and cross-check tests for gh05t3_binary/inference/pack_ternary.py."""
from __future__ import annotations

import torch

from gh05t3_binary.core.binary_layers import TernaryLinear
from gh05t3_binary.inference.pack_ternary import (
    pack_ternary_row_major,
    ternarize,
    unpack_ternary_row_major,
)


def test_pack_unpack_round_trip_no_padding():
    torch.manual_seed(3)
    ternary = torch.randint(-1, 2, (8, 128)).to(torch.float32)  # in_features=128, multiple of 64

    nonzero_packed, sign_packed = pack_ternary_row_major(ternary)
    assert nonzero_packed.shape == (8, 2)
    assert sign_packed.shape == (8, 2)

    recovered = unpack_ternary_row_major(nonzero_packed, sign_packed, in_features=128)
    assert torch.equal(recovered, ternary)


def test_pack_unpack_round_trip_with_padding():
    torch.manual_seed(4)
    ternary = torch.randint(-1, 2, (5, 100)).to(torch.float32)  # not a multiple of 64

    nonzero_packed, sign_packed = pack_ternary_row_major(ternary)
    assert nonzero_packed.shape == (5, 2)  # ceil(100/64) = 2

    recovered = unpack_ternary_row_major(nonzero_packed, sign_packed, in_features=100)
    assert torch.equal(recovered, ternary)


def test_ternarize_matches_ternary_linear_forward_exactly():
    """ternarize() must reproduce EXACTLY what TernaryLinear.forward()
    computes internally -- cross-checked via an identity-matrix input,
    which reveals ternary_weight (= ternary * alpha) column by column."""
    torch.manual_seed(5)
    layer = TernaryLinear(in_features=64, out_features=16)

    with torch.no_grad():
        identity = torch.eye(64)
        forward_output = layer(identity)  # (64, 16): row i = ternary_weight[:, i]
        ternary_weight_from_forward = forward_output.t()  # (16, 64)

    ternary, alpha = ternarize(layer.weight)
    ternary_weight_from_pack = ternary * alpha

    assert torch.allclose(ternary_weight_from_pack, ternary_weight_from_forward, atol=1e-5)


def test_packed_ternary_weight_reconstructs_same_alpha_scaled_output():
    """End-to-end: pack a real TernaryLinear layer's weight, unpack it,
    scale by alpha, and confirm F.linear using the reconstructed weight
    matches the layer's own forward pass on real random input."""
    torch.manual_seed(6)
    layer = TernaryLinear(in_features=96, out_features=12)
    x = torch.randn(7, 96)

    with torch.no_grad():
        expected = layer(x)

    ternary, alpha = ternarize(layer.weight)
    nonzero_packed, sign_packed = pack_ternary_row_major(ternary)
    recovered_ternary = unpack_ternary_row_major(nonzero_packed, sign_packed, in_features=96)
    recovered_weight = recovered_ternary * alpha

    actual = torch.nn.functional.linear(x, recovered_weight)
    assert torch.allclose(actual, expected, atol=1e-5)
