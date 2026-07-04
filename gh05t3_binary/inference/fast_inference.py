"""Inference-only fast path for gh05t3_binary: patches individual
BinaryLinear INSTANCES (never the class) to call the Rust AVX-512 batched
signed-accumulation kernel (gml_kernel::binary_inference) instead of
running their normal PyTorch forward.

Only instances explicitly passed to enable_fast_inference() are touched,
via an instance-level `.forward` assignment -- a standard, safe PyTorch
pattern since nn.Module.__call__ looks up `self.forward` through normal
attribute resolution, which checks the instance's __dict__ before the
class. BinaryLinear's class-level forward (used during training, and by
any model this function is never called on) is completely unaffected.

This path has no autograd support (inputs are detached, weights come from
a static packed export) -- callers must run model.eval() and must not
call .backward() through a model that has been patched.
"""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np
import torch

from gh05t3_binary.inference.pack_weights import find_binary_linear_layers

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_LIB_PATH = os.path.join(_REPO_ROOT, "gml_kernel", "target", "release", "libgml_kernel.so")


def _load_batched_forward_fn(lib_path: str):
    lib = ctypes.CDLL(lib_path)
    fn = lib.gh05t3_binary_forward_layer_batched
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float), ctypes.c_size_t,
    ]
    fn.restype = ctypes.c_int32
    return fn


class _FastBinaryLinearForward:
    """Bound to one packed layer's weights; replaces a single BinaryLinear
    instance's `.forward`. Flattens arbitrary leading (batch/seq) dims to
    rows, zero-pads to k_packed*64 (matching pack_weights.py's padding
    convention), calls the batched Rust kernel once for all rows, then
    restores the original leading shape."""

    def __init__(self, forward_fn, packed: np.ndarray, out_features: int, in_features: int, k_packed: int):
        self._forward_fn = forward_fn
        self._packed_flat = np.ascontiguousarray(packed.reshape(-1))
        self._out_features = out_features
        self._in_features = in_features
        self._k_packed = k_packed

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        in_features = orig_shape[-1]
        if in_features != self._in_features:
            raise ValueError(
                f"fast-inference layer expects in_features={self._in_features}, got {in_features}"
            )

        flat = x.reshape(-1, in_features).detach().to(torch.float32)
        num_rows = flat.shape[0]
        padded_len = self._k_packed * 64
        if padded_len != in_features:
            pad = torch.zeros(num_rows, padded_len - in_features, dtype=torch.float32)
            flat = torch.cat([flat, pad], dim=1)

        x_np = np.ascontiguousarray(flat.numpy())
        out_np = np.zeros((num_rows, self._out_features), dtype=np.float32)

        ret = self._forward_fn(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
            self._packed_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), self._packed_flat.size,
            self._out_features, self._k_packed, num_rows,
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_np.size,
        )
        if ret != 0:
            raise RuntimeError(f"gh05t3_binary_forward_layer_batched failed with code {ret}")

        out = torch.from_numpy(out_np)
        return out.reshape(*orig_shape[:-1], self._out_features)


def enable_fast_inference(model: torch.nn.Module, packed_weights_dir: str, lib_path: str | None = None) -> int:
    """Loads a packed_weights.bin/.json export (see pack_weights.py) and
    overrides every matching real BinaryLinear instance's `.forward` in
    `model` with a closure calling the Rust batched kernel. Layers present
    in the export but not found in `model` (e.g. exported from a
    differently-shaped checkpoint) are skipped, not guessed at. Returns
    the number of layers actually patched.

    Caller must call model.eval() first; the patched layers have no
    autograd support.
    """
    lib_path = lib_path or _DEFAULT_LIB_PATH
    meta_path = os.path.join(packed_weights_dir, "packed_weights.json")
    bin_path = os.path.join(packed_weights_dir, "packed_weights.bin")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(bin_path, "rb") as f:
        raw = f.read()

    forward_fn = _load_batched_forward_fn(lib_path)
    by_name = dict(find_binary_linear_layers(model))

    patched = 0
    for layer in meta["layers"]:
        mod = by_name.get(layer["name"])
        if mod is None:
            continue
        chunk = raw[layer["byte_offset"]: layer["byte_offset"] + layer["byte_length"]]
        packed = np.frombuffer(chunk, dtype=np.uint64).reshape(
            layer["out_features"], layer["k_packed"]
        ).copy()

        mod.forward = _FastBinaryLinearForward(
            forward_fn, packed, layer["out_features"], layer["in_features"], layer["k_packed"],
        )
        patched += 1

    return patched
