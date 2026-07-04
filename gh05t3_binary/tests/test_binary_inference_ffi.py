"""Verifies gml_kernel's binary_inference AVX-512 kernel against real
PyTorch computation on a real trained layer -- not just Rust-internal
self-consistency (that's covered by gml_kernel's own `cargo test`).

Requires the compiled gml_kernel .so and an exported packed_weights.bin/
.json (gh05t3_binary/inference/pack_weights.py) to already exist; skips
cleanly if either is missing rather than failing, since both are
regenerated build/export artifacts, not repo-tracked files.
"""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np
import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_PATH = os.path.join(_REPO_ROOT, "gml_kernel", "target", "release", "libgml_kernel.so")
_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "gh05t3_binary", "inference", "checkpoints")
_META_PATH = os.path.join(_CHECKPOINT_DIR, "packed_weights.json")
_BIN_PATH = os.path.join(_CHECKPOINT_DIR, "packed_weights.bin")
_TRAIN_CHECKPOINT_PATH = os.path.join(
    _REPO_ROOT, "gh05t3_binary", "train", "checkpoints", "binary_v2.pt"
)

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(_LIB_PATH) and os.path.isfile(_META_PATH) and os.path.isfile(_TRAIN_CHECKPOINT_PATH)),
    reason="requires a built gml_kernel .so, an exported packed checkpoint, and the trained binary_v2.pt",
)


def _load_binary_forward_fn():
    lib = ctypes.CDLL(_LIB_PATH)
    lib.gh05t3_binary_forward_layer.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    lib.gh05t3_binary_forward_layer.restype = ctypes.c_int32
    return lib.gh05t3_binary_forward_layer


def _load_real_layer(layer_name: str):
    with open(_META_PATH) as f:
        meta = json.load(f)
    layer = next(l for l in meta["layers"] if l["name"] == layer_name)

    with open(_BIN_PATH, "rb") as f:
        f.seek(layer["byte_offset"])
        raw = f.read(layer["byte_length"])
    packed = np.frombuffer(raw, dtype=np.uint64).copy()

    ckpt = torch.load(_TRAIN_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    real_weight = ckpt["model_state"][f"{layer_name}.weight"]
    real_signs = torch.sign(real_weight)
    real_signs = torch.where(real_signs == 0, torch.ones_like(real_signs), real_signs)

    return layer, packed, real_signs


def test_rust_kernel_matches_pytorch_on_real_trained_weights():
    """The actual correctness gate for step 2: given the SAME real trained
    weights and the SAME input, the Rust FFI kernel's output must match
    BinaryLinear.forward()'s real computation (F.linear(x, sign(weight))),
    not just be internally self-consistent."""
    forward_fn = _load_binary_forward_fn()
    layer, packed, real_signs = _load_real_layer("transformer.layers.0.attention.q_proj.binary_linear")

    torch.manual_seed(42)
    x = torch.randn(layer["in_features"])
    torch_output = torch.nn.functional.linear(x, real_signs).numpy()

    x_np = x.numpy().astype(np.float32).copy()
    out_np = np.zeros(layer["out_features"], dtype=np.float32)

    ret = forward_fn(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), packed.size,
        layer["out_features"], layer["k_packed"],
        out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_np.size,
    )

    assert ret == 0
    max_abs_diff = float(np.abs(torch_output - out_np).max())
    # Real trained weights, real random input, 1024-element sums in f32 --
    # some floating-point summation-order noise between PyTorch's BLAS
    # reduction and this kernel's SIMD reduction is expected; it should be
    # many orders of magnitude smaller than the actual signal.
    assert max_abs_diff < 1e-2, f"kernel output diverges from PyTorch reference by {max_abs_diff}"


def test_rust_kernel_rejects_mismatched_buffer_sizes():
    """The FFI boundary validates lengths rather than reading out of bounds."""
    forward_fn = _load_binary_forward_fn()
    layer, packed, _ = _load_real_layer("transformer.layers.0.attention.q_proj.binary_linear")

    x_np = np.zeros(layer["in_features"], dtype=np.float32)
    out_np = np.zeros(layer["out_features"] + 1, dtype=np.float32)  # deliberately wrong size

    ret = forward_fn(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), packed.size,
        layer["out_features"], layer["k_packed"],
        out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_np.size,
    )
    assert ret < 0
