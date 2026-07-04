import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_layers import MagnitudeAwareINBL, TernaryLinear


class HybridBinaryAttention(nn.Module):
    """Hybrid attention:
    - MA-INBL for Q/K/V (binary directions, quantized magnitudes)
    - Float attention scores
    - Dual-scale compensation (r^2 for scores, r for values) during the
      transition to fully-binary inference (r -> 1.0)
    """

    def __init__(self, dim: int, num_heads: int, binary_ratio: float = 0.95):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.binary_ratio = binary_ratio

        self.q_proj = MagnitudeAwareINBL(dim, dim)
        self.k_proj = MagnitudeAwareINBL(dim, dim)
        self.v_proj = MagnitudeAwareINBL(dim, dim)

        # Was a plain full-precision nn.Linear -- the only unquantized
        # component in this module. Ternary (not binary) specifically
        # because this sits right before the residual add: a zero state
        # lets a head's contribution actually be dropped for a given
        # output feature instead of always adding +-magnitude noise.
        self.out_proj = TernaryLinear(dim, dim, bias=True)
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
