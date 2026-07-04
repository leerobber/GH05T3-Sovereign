"""Verifies gml_kernel's ternary AVX-512 kernel against real PyTorch
computation on a real TernaryLinear layer -- the same evidentiary bar
used for the binary kernel in test_binary_inference_ffi.py. Unlike the
binary path, this doesn't need a trained checkpoint (TernaryLinear isn't
used by any existing checkpoint yet): the layer's own random init is
enough to prove the kernel matches TernaryLinear.forward() exactly, since
alpha/threshold are geometric properties of whatever weight is loaded,
not something that needs training to be meaningful for a correctness
check.

Requires only the compiled gml_kernel .so; skips cleanly if missing.
"""
from __future__ import annotations

import ctypes
import os
import time

import numpy as np
import pytest
import torch

from gh05t3_binary.core.binary_layers import BinaryLinear, TernaryLinear
from gh05t3_binary.inference.pack_ternary import pack_ternary_row_major, ternarize

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LIB_PATH = os.path.join(_REPO_ROOT, "gml_kernel", "target", "release", "libgml_kernel.so")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_LIB_PATH), reason="requires a built gml_kernel .so"
)


def _load_ternary_forward_fn():
    lib = ctypes.CDLL(_LIB_PATH)
    fn = lib.gh05t3_ternary_forward_layer_batched
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int32
    return fn


def _load_binary_batched_forward_fn():
    lib = ctypes.CDLL(_LIB_PATH)
    fn = lib.gh05t3_binary_forward_layer_batched
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int32
    return fn


def _pack_and_call(forward_fn, x: torch.Tensor, layer: TernaryLinear, out_features: int, k_packed: int):
    ternary, alpha = ternarize(layer.weight)
    nonzero_packed, sign_packed = pack_ternary_row_major(ternary)
    nonzero_flat = np.ascontiguousarray(nonzero_packed.reshape(-1))
    sign_flat = np.ascontiguousarray(sign_packed.reshape(-1))

    x_np = np.ascontiguousarray(x.numpy().astype(np.float32))
    num_rows = x_np.shape[0]
    out_np = np.zeros((num_rows, out_features), dtype=np.float32)

    ret = forward_fn(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
        nonzero_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), nonzero_flat.size,
        sign_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), sign_flat.size,
        ctypes.c_float(alpha),
        out_features, k_packed, num_rows,
        out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_np.size,
    )
    return ret, out_np


def test_rust_ternary_kernel_matches_pytorch_on_real_layer():
    torch.manual_seed(11)
    in_features, out_features = 1024, 64
    layer = TernaryLinear(in_features=in_features, out_features=out_features)
    layer.eval()

    x = torch.randn(8, in_features)  # real batch of rows, not a single vector
    with torch.no_grad():
        torch_output = layer(x).numpy()

    forward_fn = _load_ternary_forward_fn()
    k_packed = (in_features + 63) // 64
    ret, out_np = _pack_and_call(forward_fn, x, layer, out_features, k_packed)

    assert ret == 0
    max_abs_diff = float(np.abs(torch_output - out_np).max())
    assert max_abs_diff < 1e-2, f"ternary kernel diverges from PyTorch reference by {max_abs_diff}"


def test_rust_ternary_kernel_rejects_mismatched_buffer_sizes():
    forward_fn = _load_ternary_forward_fn()
    in_features, out_features = 128, 8
    k_packed = (in_features + 63) // 64

    x_np = np.zeros((2, in_features), dtype=np.float32)
    nonzero = np.zeros(out_features * k_packed, dtype=np.uint64)
    sign = np.zeros(out_features * k_packed, dtype=np.uint64)
    out_np = np.zeros((2, out_features + 1), dtype=np.float32)  # deliberately wrong

    ret = forward_fn(
        x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
        nonzero.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), nonzero.size,
        sign.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), sign.size,
        ctypes.c_float(1.0),
        out_features, k_packed, 2,
        out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_np.size,
    )
    assert ret < 0


def test_ternary_kernel_honest_benchmark_vs_binary_and_pytorch():
    """Reports real, measured numbers -- not asserted as "ternary must be
    faster" in either direction, matching this project's established
    discipline of not assuming a perf win just because sparsity exists.
    Ternary does strictly more work per weight than binary (an extra mask
    check/AND), so the real question is whether the zero-skip shortcut
    (see ternary_accumulate_row_avx512's `if nz_mask16 == 0 { continue }`)
    earns back the difference at realistic sparsity (~40-60%, per
    test_ternary_layer.py's threshold sanity check)."""
    torch.manual_seed(12)
    in_features, out_features = 1024, 1024
    num_rows = 16
    k_packed = (in_features + 63) // 64

    ternary_layer = TernaryLinear(in_features=in_features, out_features=out_features)
    ternary_layer.eval()
    binary_layer = BinaryLinear(in_features=in_features, out_features=out_features)
    binary_layer.eval()

    x = torch.randn(num_rows, in_features)

    ternary_fn = _load_ternary_forward_fn()
    binary_fn = _load_binary_batched_forward_fn()

    ternary, alpha = ternarize(ternary_layer.weight)
    nz_packed, sign_packed = pack_ternary_row_major(ternary)
    nz_flat = np.ascontiguousarray(nz_packed.reshape(-1))
    sign_flat = np.ascontiguousarray(sign_packed.reshape(-1))

    from gh05t3_binary.inference.pack_weights import pack_signs_row_major
    binary_signs = torch.sign(binary_layer.weight)
    binary_signs = torch.where(binary_signs == 0, torch.ones_like(binary_signs), binary_signs)
    binary_packed = pack_signs_row_major(binary_signs)
    binary_flat = np.ascontiguousarray(binary_packed.reshape(-1))

    x_np = np.ascontiguousarray(x.numpy().astype(np.float32))
    out_ternary = np.zeros((num_rows, out_features), dtype=np.float32)
    out_binary = np.zeros((num_rows, out_features), dtype=np.float32)

    def run_ternary():
        ternary_fn(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
            nz_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), nz_flat.size,
            sign_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), sign_flat.size,
            ctypes.c_float(alpha), out_features, k_packed, num_rows,
            out_ternary.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_ternary.size,
        )

    def run_binary():
        binary_fn(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
            binary_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), binary_flat.size,
            out_features, k_packed, num_rows,
            out_binary.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_binary.size,
        )

    run_ternary()
    run_binary()
    with torch.no_grad():
        ternary_layer(x)
        binary_layer(x)

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_ternary()
    ternary_time = (time.perf_counter() - t0) / n_runs

    t0 = time.perf_counter()
    for _ in range(n_runs):
        run_binary()
    binary_time = (time.perf_counter() - t0) / n_runs

    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(n_runs):
            ternary_layer(x)
        pytorch_ternary_time = (time.perf_counter() - t0) / n_runs

        t0 = time.perf_counter()
        for _ in range(n_runs):
            binary_layer(x)
        pytorch_binary_time = (time.perf_counter() - t0) / n_runs

    zero_fraction = float((ternary == 0).float().mean())
    print(
        f"\n[{num_rows}x{in_features}->{out_features}, zero_fraction={zero_fraction:.2f}] "
        f"Rust ternary: {ternary_time*1e6:.1f}us, Rust binary: {binary_time*1e6:.1f}us, "
        f"PyTorch ternary: {pytorch_ternary_time*1e6:.1f}us, PyTorch binary: {pytorch_binary_time*1e6:.1f}us"
    )
