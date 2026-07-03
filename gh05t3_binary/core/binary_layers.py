import torch
import torch.nn as nn
import torch.nn.functional as F


class _RoundSTE(torch.autograd.Function):
    """Straight-through estimator for round()/sign(): forward quantizes,
    backward passes the gradient through unchanged (treats the quantizer
    as identity for gradient purposes). Without this, torch.sign()/
    torch.round() block all gradient flow and the "trainable" float
    weights behind them never receive an update.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class _SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class UniformQuantizer(nn.Module):
    """4-bit uniform quantizer for magnitudes, gradient-transparent via STE."""

    def __init__(self, bits: int = 4):
        super().__init__()
        self.bits = bits
        self.scale = 2 ** (bits - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = torch.clamp(x * self.scale, -self.scale, self.scale - 1)
        return _RoundSTE.apply(x_scaled) / self.scale


class BinaryLinear(nn.Module):
    """Binary linear layer with a real Straight-Through Estimator.

    The fp32 `weight` is used directly in the forward computation graph
    (via _SignSTE), so gradients flow back to it during backprop — the
    previous version cached torch.sign(weight) into a detached buffer,
    which silently made `weight` untrainable (zero gradient, ever).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        binary_weight = _SignSTE.apply(self.weight)
        return F.linear(x, binary_weight)


class MagnitudeAwareINBL(nn.Module):
    """MA-INBL: splits input into direction/magnitude, binarizes direction,
    reintegrates quantized magnitude."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.binary_linear = BinaryLinear(in_features, out_features)
        self.denorm = nn.Parameter(torch.ones(out_features))
        self.mag_quantizer = UniformQuantizer(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.norm(dim=-1, keepdim=True)
        direction = x / (mag + 1e-8)
        y_dir = self.binary_linear(direction)
        mag_q = self.mag_quantizer(mag)
        return y_dir * mag_q * self.denorm


class InputNormalizedBinaryLinear(nn.Module):
    """INBL: normalizes inputs to [-1,1], applies binary weights, denormalizes outputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.binary_linear = BinaryLinear(in_features, out_features)
        self.denorm = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
        y = self.binary_linear(x_norm)
        return y * self.denorm


class MultiBitLinear4(nn.Module):
    """4-bit linear layer for terminal projections. Keeps an fp32 shadow
    weight, quantized via STE on every forward pass (both input and
    weight), so the optimizer can actually move it."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randint(-8, 8, (out_features, in_features)).float()
        )
        # Summing `in_features` terms of a ~[-8,7]-valued weight against a
        # ~[-1,1]-valued quantized input gives a pre-scale output stddev of
        # roughly sqrt(in_features) * O(1) -- e.g. ~37 at in_features=256.
        # scale=1 left logits wildly overscaled at init (measured initial
        # training loss ~100-160 nats vs. the ~5.5 a random 256-way
        # classifier should start at). Fan-in scaling (1/sqrt(in_features),
        # the standard Kaiming/Xavier-style correction) brings the output
        # back to a well-calibrated O(1-3) range at initialization.
        self.scale = nn.Parameter(torch.full((out_features,), 1.0 / (in_features ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = _RoundSTE.apply(torch.clamp(x * 8, -8, 7)) / 8.0
        w_q = _RoundSTE.apply(torch.clamp(self.weight, -8, 7))
        out = torch.matmul(x_q, w_q.t()) * self.scale.unsqueeze(0)
        return out
