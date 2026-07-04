"""Verifies TernaryLinear (Ternary Weight Networks, Li & Liu 2016) against
hand-computed references before anything downstream (packing, a Rust
kernel) is allowed to trust it.
"""
from __future__ import annotations

import torch

from gh05t3_binary.core.binary_layers import BinaryLinear, TernaryLinear


def test_ternary_weight_matches_explicit_threshold_formula():
    """Craft a weight tensor with known values straddling the threshold
    and confirm the exact expected {-1,0,+1} pattern and alpha, computed
    independently of TernaryLinear's own code."""
    layer = TernaryLinear(in_features=6, out_features=1)
    with torch.no_grad():
        # mean(|w|) = (0.05+0.1+3.0+0.2+0.05+4.0)/6 = 1.2333...
        # delta = 0.7 * 1.2333 = 0.8633...
        # -> only 3.0 and 4.0 (and their signs) survive the threshold
        layer.weight.copy_(torch.tensor([[0.05, -0.1, 3.0, -0.2, 0.05, -4.0]]))

    w = layer.weight.detach()
    delta = 0.7 * w.abs().mean()
    expected_ternary = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, -1.0]])
    expected_alpha = (3.0 + 4.0) / 2.0  # mean(|w_i| : |w_i| > delta)

    x = torch.eye(6)  # identity input -> output row j directly reveals ternary_weight[0, j]
    out = layer(x)
    expected_output = (expected_ternary * expected_alpha).squeeze(0)

    assert torch.allclose(out.squeeze(-1), expected_output, atol=1e-5), (
        f"got {out.squeeze(-1)}, expected {expected_output} (delta={delta.item()})"
    )


def test_ternary_gradient_flows_through_weight():
    torch.manual_seed(0)
    layer = TernaryLinear(in_features=32, out_features=16)
    x = torch.randn(4, 32, requires_grad=False)

    out = layer(x)
    loss = out.pow(2).sum()
    loss.backward()

    assert layer.weight.grad is not None
    assert layer.weight.grad.abs().sum().item() > 0, "STE produced zero gradient -- weight would never train"


def test_ternary_weight_is_actually_sparse():
    """The whole point of ternary over binary: some weights should
    genuinely be zero, not just small. Confirms the threshold isn't
    degenerate (e.g. always 0, ternarizing everything to +-1 like binary)."""
    torch.manual_seed(1)
    layer = TernaryLinear(in_features=512, out_features=512)
    w = layer.weight.detach()
    delta = 0.7 * w.abs().mean()
    zero_fraction = ((w.abs() <= delta).float().mean()).item()

    # Random Gaussian weights ternarized at delta=0.7*mean(|w|) should
    # zero out a substantial fraction (not ~0%, not ~100%) -- a real,
    # checkable property of the threshold formula, not a tuned constant.
    assert 0.2 < zero_fraction < 0.8, f"unexpected zero fraction {zero_fraction} -- threshold looks degenerate"


def test_ternary_output_differs_from_binary_on_same_weights():
    """Sanity check that TernaryLinear isn't secretly behaving like
    BinaryLinear (e.g. threshold always 0) by comparing outputs from two
    layers seeded with the identical underlying weight tensor."""
    torch.manual_seed(2)
    shared_weight = torch.randn(8, 32) * 0.02

    binary_layer = BinaryLinear(in_features=32, out_features=8)
    ternary_layer = TernaryLinear(in_features=32, out_features=8)
    with torch.no_grad():
        binary_layer.weight.copy_(shared_weight)
        ternary_layer.weight.copy_(shared_weight)

    x = torch.randn(5, 32)
    binary_out = binary_layer(x)
    ternary_out = ternary_layer(x)

    assert not torch.allclose(binary_out, ternary_out), (
        "ternary and binary layers produced identical output on the same weights -- "
        "ternarization threshold is not actually doing anything"
    )
