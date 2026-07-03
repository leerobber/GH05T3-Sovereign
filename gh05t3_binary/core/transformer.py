import torch
import torch.nn as nn

from .binary_layers import InputNormalizedBinaryLinear, MultiBitLinear4
from .attention import HybridBinaryAttention
from .stabilizers import MagnitudeGrowthClamper, OrthogonalResidualDecomposer


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
    """

    def __init__(self, dim: int, num_heads: int, max_depth: int = 48, binary_ratio: float = 0.95):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_depth = max_depth
        self.binary_ratio = binary_ratio

        self.attention = HybridBinaryAttention(dim, num_heads, binary_ratio)
        self.out_proj = MultiBitLinear4(dim, dim)

        self.attn_diversify = OrthogonalResidualDecomposer(dim)
        self.attn_clamp = MagnitudeGrowthClamper(max_growth=0.10)

        self.mlp_diversify = OrthogonalResidualDecomposer(dim)
        self.mlp_clamp = MagnitudeGrowthClamper(max_growth=0.10)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            InputNormalizedBinaryLinear(dim, 4 * dim),
            nn.GELU(),
            InputNormalizedBinaryLinear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor, depth: int = 0) -> torch.Tensor:
        attn_output = self.out_proj(self.attention(self.norm1(x)))
        x = self.attn_clamp(x, self.attn_diversify(attn_output))

        mlp_output = self.mlp(self.norm2(x))
        x = self.mlp_clamp(x, self.mlp_diversify(mlp_output))

        return x


class GH05T3BinaryTransformer(nn.Module):
    """GH05T3 binary transformer model."""

    def __init__(
        self,
        num_layers: int = 48,
        dim: int = 1024,
        num_heads: int = 16,
        vocab_size: int = 50257,
        binary_ratio: float = 0.95,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.binary_ratio = binary_ratio

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                BinaryTransformerBlock(dim, num_heads, max_depth=num_layers, binary_ratio=binary_ratio)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.head = MultiBitLinear4(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for depth, layer in enumerate(self.layers):
            x = layer(x, depth=depth)
        x = self.final_norm(x)
        return self.head(x)

    def set_binary_ratio(self, ratio: float) -> None:
        self.binary_ratio = ratio
        for layer in self.layers:
            layer.binary_ratio = ratio
            layer.attention.binary_ratio = ratio
