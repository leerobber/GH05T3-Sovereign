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


def test_causal_mask_blocks_future_tokens():
    """Changing a later token must not change an earlier position's logits
    -- otherwise the model isn't actually causal, and next-token training
    on it would be predicting positions that can already see themselves."""
    model = _small_model()
    model.eval()

    ids_a = torch.randint(0, 100, (1, 16))
    ids_b = ids_a.clone()
    ids_b[0, 10] = (ids_b[0, 10] + 1) % 100  # change a later token

    with torch.no_grad():
        logits_a = model(ids_a)
        logits_b = model(ids_b)

    # Positions before the changed index must be identical; the changed
    # position itself (and anything after) may legitimately differ.
    assert torch.allclose(logits_a[:, :10], logits_b[:, :10], atol=1e-5)


def test_hardware_dispatcher_matmul():
    detector = HardwareDetector()
    dispatcher = HardwareAwareBinaryDispatcher(detector)
    a = torch.randn(4, 8)
    b = torch.randn(8, 4)
    out = dispatcher.dispatch("matmul", a, b)
    assert out.shape == (4, 4)
