import torch
import torch.nn as nn

from .binary_layers import BinaryLinear


class MagnitudeGrowthClamper(nn.Module):
    """Performs the single residual addition x + O, clamping how much the
    combined magnitude is allowed to grow relative to x alone. This is the
    only place a sublayer's output actually gets added back into the
    residual stream — ORS/ORD below only transform O beforehand, they
    don't add it themselves (previously all three added O independently,
    silently triple-counting the same sublayer output)."""

    def __init__(self, max_growth: float = 0.10):
        super().__init__()
        self.max_growth = max_growth

    def forward(self, x: torch.Tensor, O: torch.Tensor) -> torch.Tensor:
        mag_x = x.norm(dim=-1, keepdim=True)
        new_mag = (x + O).norm(dim=-1, keepdim=True)
        max_allowed = mag_x * (1 + self.max_growth)
        scale = torch.clamp(max_allowed / (new_mag + 1e-8), max=1.0)
        return x + O * scale


class DepthAwareMagnitudeGate(nn.Module):
    """Alternative to MagnitudeGrowthClamper: gates O by a learned function
    of (|x|, |O|, depth) instead of a hard clamp. Also performs the
    residual addition itself — use one of MagnitudeGrowthClamper or
    DepthAwareMagnitudeGate per sublayer, not both."""

    def __init__(self, max_depth: int = 48):
        super().__init__()
        self.gate = BinaryLinear(3, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.max_depth = max_depth

    def forward(self, x: torch.Tensor, O: torch.Tensor, depth: int) -> torch.Tensor:
        mag_x = x.norm(dim=-1, keepdim=True)
        mag_O = O.norm(dim=-1, keepdim=True)
        depth_norm = torch.full_like(mag_x, depth / self.max_depth)
        features = torch.cat([mag_x, mag_O, depth_norm], dim=-1)
        gate = torch.sigmoid(self.gate(features) + self.bias)
        return x + O * gate


class OrthogonalResidualStream(nn.Module):
    """Rotates O through a fixed orthogonal transform before it's added to
    the residual stream, so it can't align/compound with x's existing
    direction over many layers. Returns the transformed O only — pair
    with MagnitudeGrowthClamper (or DepthAwareMagnitudeGate) for the
    actual addition."""

    def __init__(self, dim: int):
        super().__init__()
        raw = torch.randn(dim, dim)
        with torch.no_grad():
            q, _ = torch.linalg.qr(raw)
        self.ortho_matrix = nn.Parameter(q)

    def forward(self, O: torch.Tensor) -> torch.Tensor:
        return O @ self.ortho_matrix.t()


class OrthogonalResidualDecomposer(nn.Module):
    """Decomposes O into a learned mixture of orthogonal basis streams,
    guaranteeing directional diversity across streams. Returns the
    transformed O only — pair with MagnitudeGrowthClamper (or
    DepthAwareMagnitudeGate) for the actual addition."""

    def __init__(self, dim: int, num_streams: int = 4):
        super().__init__()
        self.num_streams = num_streams
        # QR needs a tall matrix (dim >= num_streams) to yield num_streams
        # orthonormal columns of length dim; transpose to get num_streams
        # orthonormal rows (the basis vectors we actually want).
        raw = torch.randn(dim, num_streams)
        with torch.no_grad():
            q, _ = torch.linalg.qr(raw)
        self.basis = nn.Parameter(q.t())
        self.stream_attn = BinaryLinear(dim, num_streams)

    def forward(self, O: torch.Tensor) -> torch.Tensor:
        stream_weights = torch.softmax(self.stream_attn(O), dim=-1)
        return torch.einsum("...n,nd->...d", stream_weights, self.basis)
