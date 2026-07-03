"""GH05T3 -> gh05t3_binary local backend.

Wraps the (untrained, research-stage) GH05T3BinaryTransformer as one more
selectable MODEL_CALL backend, alongside the Ollama-backed ones in
gml_kernel_bridge.py. There's no training loop, checkpoint, or real
tokenizer wired up here -- this runs a genuine forward pass through real,
STE-trainable binary layers, but with random-initialized weights it can't
produce coherent text. The output is diagnostic (logits shape / argmax
token ids), not a chat response -- deliberately not dressed up as one.

torch and gh05t3_binary are imported lazily so callers that never touch
this tier don't need torch installed (same pattern as ghost_llm's deferred
import in gml_kernel_bridge.py).
"""
from __future__ import annotations

_MODEL_CACHE: dict = {}
_VOCAB_SIZE = 256  # matches the naive byte-level tokenizer below


def _naive_tokenize(prompt: str, max_len: int = 64):
    import torch

    ids = [min(ord(c), _VOCAB_SIZE - 1) for c in prompt[:max_len]] or [0]
    return torch.tensor([ids], dtype=torch.long)


def _get_model():
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]

    from gh05t3_binary.oss.integration import GH05T3BinaryOSS

    model = GH05T3BinaryOSS(
        num_layers=4, dim=256, num_heads=4, vocab_size=_VOCAB_SIZE, binary_ratio=1.0,
    )
    model.eval()
    _MODEL_CACHE["model"] = model
    return model


def run_binary_transformer_tier(prompt: str) -> str:
    """Runs one forward pass through the binary transformer. Never raises —
    reports [MODEL_ERROR] on any failure (missing torch, bad shapes, etc.)."""
    try:
        import torch

        model = _get_model()
        input_ids = _naive_tokenize(prompt)

        with torch.no_grad():
            logits = model(input_ids)

        top_ids = logits[0].argmax(dim=-1).tolist()
        return (
            f"[binary_kernel] untrained forward pass ok — "
            f"logits shape={tuple(logits.shape)}, argmax_ids={top_ids[:16]}"
        )
    except Exception as e:
        return f"[MODEL_ERROR] binary_kernel forward failed: {e!r}"
