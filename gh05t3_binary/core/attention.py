import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_layers import BinaryLinear, MagnitudeAwareINBL, TernaryLinear

_OUT_PROJ_QUANT_MODES = {"ternary", "binary"}


class HybridBinaryAttention(nn.Module):
    """Hybrid attention:
    - MA-INBL for Q/K/V (binary directions, quantized magnitudes)
    - Float attention scores
    - Dual-scale compensation (r^2 for scores, r for values) during the
      transition to fully-binary inference (r -> 1.0)

    out_proj_quant_mode: "ternary" (default) or "binary" -- which
    quantized layer type backs out_proj. Was a plain full-precision
    nn.Linear originally -- the only unquantized component in this
    module -- then fixed to TernaryLinear specifically, since a zero
    state right before the residual add lets a head's contribution
    actually be dropped for a given output feature instead of always
    adding +-magnitude noise. Made selectable (rather than hardcoded)
    so a genome can evolve this choice per KernelGenome.quant_mode's
    real historical precedent (see backend/oss/dna/mutation_operators.py's
    QuantModeMutation) -- "binary" trades away the zero-state benefit
    for 1 bit/weight instead of 2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        binary_ratio: float = 0.95,
        out_proj_quant_mode: str = "ternary",
        mainbl_threshold: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.binary_ratio = binary_ratio

        if out_proj_quant_mode not in _OUT_PROJ_QUANT_MODES:
            raise ValueError(
                f"Unknown out_proj_quant_mode {out_proj_quant_mode!r}, expected one of {_OUT_PROJ_QUANT_MODES}"
            )
        self.out_proj_quant_mode = out_proj_quant_mode
        self.mainbl_threshold = mainbl_threshold

        # Same mainbl_threshold applied to all three -- q/k/v_proj are all
        # MagnitudeAwareINBL instances receiving the same real input x, so
        # a single genome-level trait gates all three uniformly (matching
        # how binary_ratio is one value applied model-wide, not per-branch).
        self.q_proj = MagnitudeAwareINBL(dim, dim, mag_threshold=mainbl_threshold)
        self.k_proj = MagnitudeAwareINBL(dim, dim, mag_threshold=mainbl_threshold)
        self.v_proj = MagnitudeAwareINBL(dim, dim, mag_threshold=mainbl_threshold)

        if out_proj_quant_mode == "ternary":
            self.out_proj = TernaryLinear(dim, dim, bias=True)
        else:
            self.out_proj = BinaryLinear(dim, dim, bias=True)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape
        r = self.binary_ratio

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if r >= 1.0:
            scores = scores * (self.binary_ratio ** 2)
            v = v * self.binary_ratio

        # Divide by temperature (a trainable parameter) before masking, not
        # after: masked_fill's -inf must be the last op before softmax.
        # Dividing an already-masked -inf score by temperature is fine in
        # the forward pass (-inf stays -inf), but its backward computes a
        # gradient contribution from those masked positions that multiplies
        # out to 0 * inf = NaN once combined with softmax's zero gradient
        # there -- a classic masking + division ordering bug.
        temperature = torch.exp(self.log_temperature)
        scores = scores / temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, S, self.dim)
        return self.out_proj(out)
