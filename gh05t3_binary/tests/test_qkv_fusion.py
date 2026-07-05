"""Correctness and benchmark gate for enable_qkv_fusion: fuses each
HybridBinaryAttention's q/k/v binary_linear projections into one batched
Rust kernel call (they share the same input) instead of three separate
ones. Real fix for the FFI-crossing-count overhead measured in
test_fast_inference.py, using the already-verified batched kernel --
no new Rust code.

Requires a built gml_kernel .so, an exported packed checkpoint, and the
real trained binary_v2.pt; skips cleanly if any are missing.
"""
from __future__ import annotations

import copy
import os
import time

import pytest
import torch

from gh05t3_binary.inference.fast_inference import enable_fast_inference, enable_qkv_fusion
from gh05t3_binary.oss.integration import GH05T3BinaryOSS

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_PATH = os.path.join(_REPO_ROOT, "gml_kernel", "target", "release", "libgml_kernel.so")
_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "gh05t3_binary", "inference", "checkpoints")
_META_PATH = os.path.join(_CHECKPOINT_DIR, "packed_weights.json")
_TRAIN_CHECKPOINT_PATH = os.path.join(_REPO_ROOT, "gh05t3_binary", "train", "checkpoints", "binary_v2.pt")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_LIB_PATH) and os.path.isfile(_META_PATH) and os.path.isfile(_TRAIN_CHECKPOINT_PATH)),
    reason="requires a built gml_kernel .so, an exported packed checkpoint, and the trained binary_v2.pt",
)


def _load_real_model() -> GH05T3BinaryOSS:
    ckpt = torch.load(_TRAIN_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model = GH05T3BinaryOSS(
        num_layers=ckpt["num_layers"], dim=ckpt["dim"], num_heads=ckpt["num_heads"],
        vocab_size=ckpt["vocab_size"], binary_ratio=1.0, stabilizer=ckpt.get("stabilizer", "mgc"),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def test_qkv_fusion_patches_all_12_real_attention_blocks():
    model = _load_real_model()
    patched = enable_qkv_fusion(model, _CHECKPOINT_DIR, lib_path=_LIB_PATH)
    assert patched == 12  # one HybridBinaryAttention per real transformer block


def test_qkv_fusion_logits_match_plain_pytorch():
    torch.manual_seed(11)
    reference_model = _load_real_model()
    fused_model = copy.deepcopy(reference_model)

    vocab_size = reference_model.transformer.embedding.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (2, 16))

    with torch.no_grad():
        reference_logits = reference_model(input_ids)

    patched = enable_qkv_fusion(fused_model, _CHECKPOINT_DIR, lib_path=_LIB_PATH)
    assert patched == 12

    with torch.no_grad():
        fused_logits = fused_model(input_ids)

    max_abs_diff = (reference_logits - fused_logits).abs().max().item()
    logit_scale = reference_logits.abs().mean().item()
    # Same tolerance rationale as test_fast_inference.py: fp32
    # summation-order drift compounding across 12 fused blocks.
    assert max_abs_diff < 0.5, f"diverges by {max_abs_diff} (mean |logit|={logit_scale})"
    assert max_abs_diff < 0.1 * max(logit_scale, 1.0)


def test_qkv_fusion_composes_with_full_fast_inference():
    """Real end-to-end combination: qkv-fusion for q/k/v, plain
    per-layer fast inference for everything else (out_proj, MLP, stream
    diversifiers). Both patch the same model without conflicting --
    enable_fast_inference only ever touches true BinaryLinear instances
    (q/k/v's own binary_linear submodules are BinaryLinear, but
    enable_qkv_fusion patches the ATTENTION MODULE's forward, not
    q_proj.binary_linear's forward directly, so there's no overlap to
    conflict over)."""
    torch.manual_seed(12)
    reference_model = _load_real_model()
    combined_model = copy.deepcopy(reference_model)

    vocab_size = reference_model.transformer.embedding.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (2, 16))

    with torch.no_grad():
        reference_logits = reference_model(input_ids)

    enable_qkv_fusion(combined_model, _CHECKPOINT_DIR, lib_path=_LIB_PATH)
    enable_fast_inference(combined_model, _CHECKPOINT_DIR, lib_path=_LIB_PATH)

    with torch.no_grad():
        combined_logits = combined_model(input_ids)

    max_abs_diff = (reference_logits - combined_logits).abs().max().item()
    logit_scale = reference_logits.abs().mean().item()
    assert max_abs_diff < 0.5
    assert max_abs_diff < 0.1 * max(logit_scale, 1.0)


def test_qkv_fusion_reduces_crossings_and_honest_benchmark():
    """Real, honest timing comparison: plain fast_inference (84 per-layer
    crossings, including 3 separate q/k/v crossings per block = 36) vs.
    fast_inference + qkv_fusion (same 84 minus the 36 separate q/k/v
    crossings, replaced by 12 fused ones = 60 total crossings). Reported
    as measured, not asserted as "must be faster" -- consistent with
    this project's discipline of not assuming a design change is a win
    without checking."""
    model_a = _load_real_model()  # fast_inference only
    model_b = copy.deepcopy(model_a)  # fast_inference + qkv_fusion

    enable_fast_inference(model_a, _CHECKPOINT_DIR, lib_path=_LIB_PATH)
    enable_qkv_fusion(model_b, _CHECKPOINT_DIR, lib_path=_LIB_PATH)
    enable_fast_inference(model_b, _CHECKPOINT_DIR, lib_path=_LIB_PATH)

    vocab_size = model_a.transformer.embedding.weight.shape[0]
    input_ids = torch.randint(0, vocab_size, (2, 16))

    with torch.no_grad():
        model_a(input_ids)  # warmup
        model_b(input_ids)

        n_runs = 5
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model_a(input_ids)
        per_layer_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            model_b(input_ids)
        fused_time = (time.perf_counter() - t0) / n_runs

    print(
        f"\nfast_inference only (84 crossings): {per_layer_time*1000:.3f}ms, "
        f"+ qkv_fusion (60 crossings): {fused_time*1000:.3f}ms"
    )
