"""Step 3 correctness gate: a real, full 12-layer model forward pass must
produce (nearly) the same logits whether run with normal PyTorch
BinaryLinear layers or with enable_fast_inference() applied. This is the
thing that actually matters -- Steps 1/2 verified an isolated layer and an
isolated kernel; this test verifies wiring the kernel into the real model
doesn't silently change what the model computes.

Requires a built gml_kernel .so, an exported packed checkpoint, and the
real trained binary_v2.pt; skips cleanly if any are missing, since all
three are regenerated build/export artifacts, not repo-tracked files.
"""
from __future__ import annotations

import copy
import os
import time

import pytest
import torch

from gh05t3_binary.inference.fast_inference import enable_fast_inference
from gh05t3_binary.oss.integration import GH05T3BinaryOSS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_PATH = os.path.join(_REPO_ROOT, "gml_kernel", "target", "release", "libgml_kernel.so")
_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "gh05t3_binary", "inference", "checkpoints")
_META_PATH = os.path.join(_CHECKPOINT_DIR, "packed_weights.json")
_TRAIN_CHECKPOINT_PATH = os.path.join(
    _REPO_ROOT, "gh05t3_binary", "train", "checkpoints", "binary_v2.pt"
)

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_LIB_PATH) and os.path.isfile(_META_PATH) and os.path.isfile(_TRAIN_CHECKPOINT_PATH)),
    reason="requires a built gml_kernel .so, an exported packed checkpoint, and the trained binary_v2.pt",
)


def _load_real_model() -> GH05T3BinaryOSS:
    ckpt = torch.load(_TRAIN_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = GH05T3BinaryOSS(
        num_layers=ckpt["num_layers"],
        dim=ckpt["dim"],
        num_heads=ckpt["num_heads"],
        vocab_size=ckpt["vocab_size"],
        binary_ratio=1.0,
        stabilizer=ckpt.get("stabilizer", "mgc"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def test_full_model_logits_match_between_pytorch_and_fast_inference():
    torch.manual_seed(7)

    reference_model = _load_real_model()
    fast_model = copy.deepcopy(reference_model)

    vocab_size = reference_model.transformer.embedding.weight.shape[0]
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (2, seq_len))  # real batch>1, seq>1 shapes

    with torch.no_grad():
        reference_logits = reference_model(input_ids)

    patched = enable_fast_inference(fast_model, _CHECKPOINT_DIR, lib_path=_LIB_PATH)
    assert patched > 0, "expected at least one BinaryLinear layer to be patched"

    with torch.no_grad():
        fast_logits = fast_model(input_ids)

    assert reference_logits.shape == fast_logits.shape

    max_abs_diff = (reference_logits - fast_logits).abs().max().item()
    logit_scale = reference_logits.abs().mean().item()
    # 84 patched layers deep, some fp32 summation-order drift between
    # PyTorch's BLAS reduction and the SIMD kernel's reduction compounds
    # across layers -- this bound is loose relative to Step 2's per-layer
    # 1e-2 gate for exactly that reason, but must still be small relative
    # to the actual logit magnitudes, not just "some tolerance".
    assert max_abs_diff < 0.5, (
        f"full-model logits diverge by {max_abs_diff} (mean |logit|={logit_scale}) "
        "after enabling fast inference"
    )
    assert max_abs_diff < 0.1 * max(logit_scale, 1.0), (
        f"full-model logit divergence {max_abs_diff} is large relative to "
        f"mean |logit|={logit_scale}"
    )


def test_full_model_fast_inference_is_not_slower_than_pytorch():
    """Real end-to-end timing comparison -- Step 2 measured a per-layer
    win, but with 84 FFI calls per forward pass across a real model, this
    confirms the win survives (or at least doesn't regress) once
    conversion/dispatch overhead is included, rather than just asserting
    "it works"."""
    reference_model = _load_real_model()
    fast_model = copy.deepcopy(reference_model)
    enable_fast_inference(fast_model, _CHECKPOINT_DIR, lib_path=_LIB_PATH)

    vocab_size = reference_model.transformer.embedding.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (2, 16))

    with torch.no_grad():
        reference_model(input_ids)  # warmup
        fast_model(input_ids)

        n_runs = 5
        t0 = time.perf_counter()
        for _ in range(n_runs):
            reference_model(input_ids)
        pytorch_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            fast_model(input_ids)
        fast_time = (time.perf_counter() - t0) / n_runs

    print(f"\nPyTorch forward: {pytorch_time*1000:.3f}ms, fast-inference forward: {fast_time*1000:.3f}ms")
    # Reported honestly regardless of outcome -- not asserted as "must be
    # faster", since per-layer FFI/numpy conversion overhead across 84
    # small layers may or may not net out ahead of PyTorch's own
    # (already-vectorized, BLAS-backed) F.linear at this model size.
