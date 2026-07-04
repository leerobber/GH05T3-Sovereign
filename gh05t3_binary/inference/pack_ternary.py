"""Packs TernaryLinear weights ({-1,0,+1}, see
gh05t3_binary/core/binary_layers.py) into two u64 bitplanes per row -- a
nonzero mask and a sign mask -- plus a per-tensor alpha scale, for the
Rust ternary SIMD kernel (gml_kernel/src/binary_inference/mod.rs).

Bit layout matches pack_weights.py's binary convention (row-major,
k_packed u64 words per row) but doubles it: bit b of word w in the
nonzero-mask plane is 1 iff that weight is nonzero (+1 or -1); bit b of
word w in the sign-mask plane is 1 iff that weight is +1 (written as 0,
not left undefined, wherever the nonzero bit is 0 -- deterministic output
even though the kernel never reads a sign bit whose nonzero bit is 0).
alpha is a single fp32 scalar per layer (TernaryLinear uses one
per-tensor scale, not per-row).
"""
from __future__ import annotations

import numpy as np
import torch


def ternarize(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Reproduces TernaryLinear.forward()'s quantization exactly (same
    delta=0.7*mean(|W|) threshold, same least-squares alpha), without
    needing to run an actual forward pass through the module -- so
    packing can operate directly on a saved state_dict's weight tensor.
    """
    w = weight.detach()
    delta = 0.7 * w.abs().mean()

    ternary = torch.zeros_like(w)
    ternary[w > delta] = 1.0
    ternary[w < -delta] = -1.0

    mask = (ternary != 0).to(w.dtype)
    nonzero_count = mask.sum().clamp(min=1.0)
    alpha = ((w.abs() * mask).sum() / nonzero_count).item()

    return ternary, alpha


def pack_ternary_row_major(ternary: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """ternary: (out_features, in_features) tensor of {-1,0,+1}.
    Returns (nonzero_packed, sign_packed), each (out_features, k_packed)
    uint64, same row-major/padding convention as pack_weights.py."""
    out_features, in_features = ternary.shape
    k_packed = (in_features + 63) // 64
    padded_len = k_packed * 64

    nonzero_bits = (ternary != 0).to(torch.uint8).numpy()
    sign_bits = (ternary > 0).to(torch.uint8).numpy()

    if padded_len != in_features:
        pad = np.zeros((out_features, padded_len - in_features), dtype=np.uint8)
        nonzero_bits = np.concatenate([nonzero_bits, pad], axis=1)
        sign_bits = np.concatenate([sign_bits, pad], axis=1)

    weights_lut = (1 << np.arange(64, dtype=np.uint64))

    def _pack(bits: np.ndarray) -> np.ndarray:
        bits = bits.reshape(out_features, k_packed, 64)
        return (bits.astype(np.uint64) * weights_lut).sum(axis=2, dtype=np.uint64)

    return _pack(nonzero_bits), _pack(sign_bits)


def unpack_ternary_row_major(
    nonzero_packed: np.ndarray, sign_packed: np.ndarray, in_features: int
) -> torch.Tensor:
    """Inverse of pack_ternary_row_major, for round-trip verification.
    Returns an (out_features, in_features) tensor of {-1.0, 0.0, +1.0}."""
    out_features, k_packed = nonzero_packed.shape

    def _unpack(packed: np.ndarray) -> np.ndarray:
        bits = np.zeros((out_features, k_packed * 64), dtype=np.uint8)
        for b in range(64):
            bits[:, b::64] = ((packed >> np.uint64(b)) & np.uint64(1)).astype(np.uint8)
        return bits[:, :in_features]

    nonzero_bits = _unpack(nonzero_packed)
    sign_bits = _unpack(sign_packed)

    ternary = np.zeros_like(nonzero_bits, dtype=np.float32)
    ternary[(nonzero_bits == 1) & (sign_bits == 1)] = 1.0
    ternary[(nonzero_bits == 1) & (sign_bits == 0)] = -1.0
    return torch.from_numpy(ternary)
