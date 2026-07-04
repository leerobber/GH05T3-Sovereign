import torch
import torch.nn as nn

from gh05t3_binary.core.transformer import GH05T3BinaryTransformer


class GH05T3BinaryOSS(nn.Module):
    """Integration shim: wraps GH05T3BinaryTransformer for use as a local
    backend tier, called from backend/integration/binary_backend.py."""

    def __init__(
        self,
        num_layers: int = 48,
        dim: int = 1024,
        num_heads: int = 16,
        vocab_size: int = 50257,
        binary_ratio: float = 0.95,
        stabilizer: str = "mgc",
    ):
        super().__init__()
        self.transformer = GH05T3BinaryTransformer(
            num_layers=num_layers,
            dim=dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            binary_ratio=binary_ratio,
            stabilizer=stabilizer,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer(input_ids)
