"""GH05T3 -> gh05t3_binary local backend.

Wraps GH05T3BinaryTransformer as one more selectable MODEL_CALL backend,
alongside the Ollama-backed ones in gml_kernel_bridge.py. Loads the
checkpoint produced by gh05t3_binary/train/train_binary.py if one exists
at CHECKPOINT_PATH; falls back to an untrained (random-weight) model
otherwise, so this tier works before training has ever been run.

If the checkpoint records tokenizer_type="bpe" (gh05t3_binary/train/
bpe_tokenizer.py, trained on a real corpus), the matching tokenizer is
loaded from tokenizer_path and used for real encode/decode -- the output
includes an actual decoded text preview, not just token ids. Older
checkpoints (or no checkpoint at all) fall back to the naive byte-level
mapping this always used. Either way the label ("trained checkpoint" vs
"untrained") and the model's tiny scale (a few layers, dim in the
hundreds) mean this stays a diagnostic report, not a claim of coherent
chat output -- a real decoded preview from an undertrained toy-scale
model is still mostly noise, just noise you can read as text now instead
of only as token ids.

torch and gh05t3_binary are imported lazily so callers that never touch
this tier don't need torch installed (same pattern as ghost_llm's deferred
import in gml_kernel_bridge.py).
"""
from __future__ import annotations

import os

_MODEL_CACHE: dict = {}
_VOCAB_SIZE = 256  # fallback vocab size for the naive byte-level tokenizer
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "gh05t3_binary", "train", "checkpoints", "binary_v2.pt"
)


def _naive_tokenize(prompt: str, max_len: int = 64):
    import torch

    ids = [min(ord(c), _VOCAB_SIZE - 1) for c in prompt[:max_len]] or [0]
    return torch.tensor([ids], dtype=torch.long)


def _naive_decode(ids) -> str:
    return "".join(chr(i) for i in ids)


def _get_model_and_tokenizer():
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]

    import torch

    from gh05t3_binary.oss.integration import GH05T3BinaryOSS

    tokenizer = None

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

        tokenizer_path = ckpt.get("tokenizer_path")
        if ckpt.get("tokenizer_type") == "bpe" and tokenizer_path and os.path.isfile(tokenizer_path):
            from gh05t3_binary.train.bpe_tokenizer import BPETokenizer

            tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        model = GH05T3BinaryOSS(
            num_layers=4, dim=256, num_heads=4, vocab_size=_VOCAB_SIZE, binary_ratio=1.0,
        )
        _MODEL_CACHE["trained"] = False

    model.eval()
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["tokenizer"] = tokenizer
    return model, tokenizer


def run_binary_transformer_tier(prompt: str) -> str:
    """Runs one forward pass through the binary transformer. Never raises —
    reports [MODEL_ERROR] on any failure (missing torch, bad shapes, etc.)."""
    try:
        import torch

        model, tokenizer = _get_model_and_tokenizer()

        if tokenizer is not None:
            ids = tokenizer.encode(prompt)[:64] or [0]
            input_ids = torch.tensor([ids], dtype=torch.long)
        else:
            input_ids = _naive_tokenize(prompt)

        with torch.no_grad():
            logits = model(input_ids)

        top_ids = logits[0].argmax(dim=-1).tolist()
        label = "trained checkpoint" if _MODEL_CACHE.get("trained") else "untrained"

        if tokenizer is not None:
            decoded = tokenizer.decode(top_ids)
            return (
                f"[binary_kernel] {label} forward pass ok (bpe tokenizer) — "
                f"logits shape={tuple(logits.shape)}, decoded_preview={decoded[:200]!r}"
            )

        return (
            f"[binary_kernel] {label} forward pass ok — "
            f"logits shape={tuple(logits.shape)}, argmax_ids={top_ids[:16]}"
        )
    except Exception as e:
        return f"[MODEL_ERROR] binary_kernel forward failed: {e!r}"
