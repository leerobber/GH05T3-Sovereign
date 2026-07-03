"""GH05T3 -> gh05t3_binary local backend.

Wraps GH05T3BinaryTransformer as one more selectable MODEL_CALL backend,
alongside the Ollama-backed ones in gml_kernel_bridge.py. Loads the
checkpoint produced by gh05t3_binary/train/train_binary.py if one exists
at CHECKPOINT_PATH; falls back to an untrained (random-weight) model
otherwise, so this tier works before training has ever been run. Either
way the output is a diagnostic report (logits shape / argmax token ids),
not dressed up as a chat response -- there's still no real tokenizer,
just a naive byte-level mapping, so even a trained checkpoint won't
produce coherent text at this scale.

torch and gh05t3_binary are imported lazily so callers that never touch
this tier don't need torch installed (same pattern as ghost_llm's deferred
import in gml_kernel_bridge.py).
"""
from __future__ import annotations

import os

_MODEL_CACHE: dict = {}
_VOCAB_SIZE = 256  # matches the naive byte-level tokenizer below
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "gh05t3_binary", "train", "checkpoints", "binary_v2.pt"
)


def _naive_tokenize(prompt: str, max_len: int = 64):
    import torch

    ids = [min(ord(c), _VOCAB_SIZE - 1) for c in prompt[:max_len]] or [0]
    return torch.tensor([ids], dtype=torch.long)


def _get_model():
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"]

    import torch

    from gh05t3_binary.oss.integration import GH05T3BinaryOSS

    if os.path.isfile(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model = GH05T3BinaryOSS(
            num_layers=ckpt.get("num_layers", 4),
            dim=ckpt.get("dim", 256),
            num_heads=ckpt.get("num_heads", 4),
            vocab_size=ckpt.get("vocab_size", _VOCAB_SIZE),
            binary_ratio=1.0,
        )
        model.load_state_dict(ckpt["model_state"])
        _MODEL_CACHE["trained"] = True
    else:
        model = GH05T3BinaryOSS(
            num_layers=4, dim=256, num_heads=4, vocab_size=_VOCAB_SIZE, binary_ratio=1.0,
        )
        _MODEL_CACHE["trained"] = False

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
        label = "trained checkpoint" if _MODEL_CACHE.get("trained") else "untrained"
        return (
            f"[binary_kernel] {label} forward pass ok — "
            f"logits shape={tuple(logits.shape)}, argmax_ids={top_ids[:16]}"
        )
    except Exception as e:
        return f"[MODEL_ERROR] binary_kernel forward failed: {e!r}"
