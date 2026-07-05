from typing import Optional

import torch
import torch.nn as nn

from .binary_layers import InputNormalizedBinaryLinear, MultiBitLinear4
from .attention import HybridBinaryAttention
from .stabilizers import (
    DepthAwareMagnitudeGate,
    MagnitudeGrowthClamper,
    OrthogonalResidualDecomposer,
)

_STABILIZERS = {"mgc", "damg"}


def _build_stabilizer(kind: str, max_depth: int):
    if kind == "mgc":
        return MagnitudeGrowthClamper(max_growth=0.10)
    if kind == "damg":
        return DepthAwareMagnitudeGate(max_depth=max_depth)
    raise ValueError(f"Unknown stabilizer {kind!r}, expected one of {_STABILIZERS}")


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """1 = attend, 0 = block (matches HybridBinaryAttention's masked_fill(mask==0, -inf))."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)


class BinaryTransformerBlock(nn.Module):
    """Binary transformer block with stabilized residuals.

    Each sublayer (attention, MLP) contributes an output O that is:
      1. diversified via OrthogonalResidualDecomposer (prevents O from
         aligning with x's existing direction across depth), then
      2. added back to x exactly once, with magnitude growth clamped via
         MagnitudeGrowthClamper.

    (The previous version applied MGC, ORS, and ORD as three independent
    additions of the same O — silently adding each sublayer's output to
    the residual stream three times instead of once. It also computed a
    block-level q/k/v via separate MA-INBL modules that were never used —
    HybridBinaryAttention already does its own Q/K/V projection internally
    with its own parameters. Both are fixed here.)

    stabilizer: "mgc" (default, MagnitudeGrowthClamper -- hard-clamps
    growth by a fixed ratio) or "damg" (DepthAwareMagnitudeGate -- learns
    a per-token gate as a function of |x|, |O|, and depth, rather than a
    fixed clamp). Both perform the sublayer's one residual addition;
    picking one doesn't change anything else about the block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_depth: int = 48,
        binary_ratio: float = 0.95,
        stabilizer: str = "mgc",
        out_proj_quant_mode: str = "ternary",
        mainbl_threshold: float = 0.0,
        ternary_sparsity_target: float | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.binary_ratio = binary_ratio
        self.stabilizer = stabilizer
        self._depth_aware = stabilizer == "damg"

        self.attention = HybridBinaryAttention(
            dim, num_heads, binary_ratio, out_proj_quant_mode=out_proj_quant_mode,
            mainbl_threshold=mainbl_threshold, ternary_sparsity_target=ternary_sparsity_target,
        )
        self.out_proj = MultiBitLinear4(dim, dim)

        self.attn_diversify = OrthogonalResidualDecomposer(dim)
        self.attn_clamp = _build_stabilizer(stabilizer, max_depth)

        self.mlp_diversify = OrthogonalResidualDecomposer(dim)
        self.mlp_clamp = _build_stabilizer(stabilizer, max_depth)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            InputNormalizedBinaryLinear(dim, 4 * dim),
            nn.GELU(),
            InputNormalizedBinaryLinear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, depth: int = 0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.out_proj(self.attention(self.norm1(x), mask=mask))
        attn_diverse = self.attn_diversify(attn_output)
        x = self.attn_clamp(x, attn_diverse, depth) if self._depth_aware else self.attn_clamp(x, attn_diverse)

        mlp_output = self.mlp(self.norm2(x))
        mlp_diverse = self.mlp_diversify(mlp_output)
        x = self.mlp_clamp(x, mlp_diverse, depth) if self._depth_aware else self.mlp_clamp(x, mlp_diverse)

        return x


class GH05T3BinaryTransformer(nn.Module):
    """GH05T3 binary transformer model.

    stabilizer: "mgc" (default) or "damg" -- see BinaryTransformerBlock.
    """

    def __init__(
        self,
        num_layers: int = 48,
        dim: int = 1024,
        num_heads: int = 16,
        vocab_size: int = 50257,
        binary_ratio: float = 0.95,
        stabilizer: str = "mgc",
        out_proj_quant_mode: str = "ternary",
        mainbl_threshold: float = 0.0,
        ternary_sparsity_target: float | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.binary_ratio = binary_ratio
        self.stabilizer = stabilizer
        self.out_proj_quant_mode = out_proj_quant_mode
        self.mainbl_threshold = mainbl_threshold
        self.ternary_sparsity_target = ternary_sparsity_target

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                BinaryTransformerBlock(
                    dim, num_heads, max_depth=num_layers, binary_ratio=binary_ratio, stabilizer=stabilizer,
                    out_proj_quant_mode=out_proj_quant_mode, mainbl_threshold=mainbl_threshold,
                    ternary_sparsity_target=ternary_sparsity_target,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = MultiBitLinear4(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        mask = _causal_mask(input_ids.size(1), input_ids.device)
        for depth, layer in enumerate(self.layers):
            x = layer(x, depth=depth, mask=mask)
        x = self.final_norm(x)
        return self.head(x)

    def set_binary_ratio(self, ratio: float) -> None:
        self.binary_ratio = ratio
        for layer in self.layers:
            layer.binary_ratio = ratio
            layer.attention.binary_ratio = ratio
