import torch

from gh05t3_binary.core.transformer import GH05T3BinaryTransformer
from gh05t3_binary.hardware.detector import HardwareDetector
from gh05t3_binary.hardware.dispatcher import HardwareAwareBinaryDispatcher


def _small_model():
    return GH05T3BinaryTransformer(
        num_layers=2, dim=128, num_heads=4, vocab_size=100, binary_ratio=0.95,
    )


def test_binary_transformer_smoke():
    model = _small_model()
    input_ids = torch.randint(0, 100, (1, 16))
    logits = model(input_ids)
    assert logits.shape == (1, 16, 100)


def test_binary_transformer_gradients_flow():
    """Confirms the STE fix actually works: without it, BinaryLinear's
    sign()-quantized weight never receives a gradient at all."""
    model = _small_model()
    input_ids = torch.randint(0, 100, (1, 16))
    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    binary_weight = model.layers[0].attention.q_proj.binary_linear.weight
    assert binary_weight.grad is not None
    assert torch.any(binary_weight.grad != 0)


def test_hardware_dispatcher_matmul():
    detector = HardwareDetector()
    dispatcher = HardwareAwareBinaryDispatcher(detector)
    a = torch.randn(4, 8)
    b = torch.randn(8, 4)
    out = dispatcher.dispatch("matmul", a, b)
    assert out.shape == (4, 4)
