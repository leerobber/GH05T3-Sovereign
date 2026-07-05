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
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from gh05t3_binary.core.attention import HybridBinaryAttention
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


def _find_hybrid_attention_layers(model: torch.nn.Module) -> list[tuple[str, HybridBinaryAttention]]:
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, HybridBinaryAttention)]


class _FusedQKVForward:
    """Bound to one HybridBinaryAttention instance; replaces its whole
    `.forward` with a version that computes Q/K/V's binary_linear
    projections in ONE batched Rust kernel call instead of three separate
    ones -- q_proj/k_proj/v_proj all take the IDENTICAL input x (see
    HybridBinaryAttention.forward: `q = self.q_proj(x); k = self.k_proj(x);
    v = self.v_proj(x)`), so their packed weight rows are concatenated
    ahead of time and the fused output is split back apart after one call.

    Everything past Q/K/V (reshape to heads, scores, binary_ratio
    scaling, temperature, masking, softmax, out_proj) is copied verbatim
    from HybridBinaryAttention.forward -- not reimplemented differently,
    to avoid silently diverging from the real computation.
    """

    def __init__(self, attn: HybridBinaryAttention, forward_fn, fused_packed: np.ndarray, dim: int, k_packed: int):
        self._attn = attn
        self._forward_fn = forward_fn
        self._fused_packed_flat = np.ascontiguousarray(fused_packed.reshape(-1))
        self._dim = dim
        self._k_packed = k_packed
        self._out_features = 3 * dim

    def _fused_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn = self._attn
        # Same direction/magnitude split MagnitudeAwareINBL.forward does,
        # computed ONCE since q_proj/k_proj/v_proj all receive the same x.
        mag = x.norm(dim=-1, keepdim=True)
        direction = x / (mag + 1e-8)

        orig_shape = direction.shape
        in_features = orig_shape[-1]
        flat = direction.reshape(-1, in_features).detach().to(torch.float32)
        num_rows = flat.shape[0]
        padded_len = self._k_packed * 64
        if padded_len != in_features:
            pad = torch.zeros(num_rows, padded_len - in_features, dtype=torch.float32)
            flat = torch.cat([flat, pad], dim=1)

        x_np = np.ascontiguousarray(flat.numpy())
        out_np = np.zeros((num_rows, self._out_features), dtype=np.float32)

        ret = self._forward_fn(
            x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x_np.size,
            self._fused_packed_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)), self._fused_packed_flat.size,
            self._out_features, self._k_packed, num_rows,
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out_np.size,
        )
        if ret != 0:
            raise RuntimeError(f"fused Q/K/V kernel call failed with code {ret}")

        fused = torch.from_numpy(out_np).reshape(*orig_shape[:-1], self._out_features)
        q_dir, k_dir, v_dir = fused.split(self._dim, dim=-1)

        # UniformQuantizer(4) has no learned parameters (pure function of
        # `bits`), and all three MA-INBL instances construct it with the
        # same default -- computing it once via q_proj's instance and
        # reusing it for k/v is mathematically identical to calling each
        # instance's own mag_quantizer separately, not an approximation.
        mag_q = attn.q_proj.mag_quantizer(mag)
        q = q_dir * mag_q * attn.q_proj.denorm
        k = k_dir * mag_q * attn.k_proj.denorm
        v = v_dir * mag_q * attn.v_proj.denorm
        return q, k, v

    def __call__(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        attn = self._attn
        B, S, _ = x.shape
        r = attn.binary_ratio

        q, k, v = self._fused_qkv(x)

        q = q.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = k.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = v.view(B, S, attn.num_heads, attn.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)

        if r >= 1.0:
            scores = scores * (attn.binary_ratio ** 2)
            v = v * attn.binary_ratio

        temperature = torch.exp(attn.log_temperature)
        scores = scores / temperature

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).reshape(B, S, attn.dim)
        return attn.out_proj(out)


def enable_qkv_fusion(model: torch.nn.Module, packed_weights_dir: str, lib_path: str | None = None) -> int:
    """Fuses each HybridBinaryAttention's q_proj/k_proj/v_proj binary_linear
    projections into ONE batched Rust kernel call instead of three
    separate ones. Reduces per-block FFI crossings for this specific step
    from 3 to 1 -- 36 fewer crossings across a real 12-block model. This
    is the concrete, scoped fix for the FFI-crossing overhead measured in
    test_fast_inference.py's full-model benchmark (316.8ms vs PyTorch's
    233.5ms) -- reducing crossing COUNT, not reimplementing attention's
    real softmax/reshape/masking logic in Rust (see the module docstring
    of the rejected sovereign_graph.rs monolith for why that's the wrong
    shape of fix: those ops are already fast in PyTorch, and the real
    topology -- parallel Q/K/V branches, real attention, GELU-gated MLP,
    two different stabilizer schemes -- doesn't fit a flat sequential
    chain at all).

    Patches the attention MODULE's whole forward (not the individual
    q/k/v sub-layers), since their outputs must be split back apart
    before the rest of attention's real logic runs. Independent of
    enable_fast_inference(): out_proj (ternary) and any other
    BinaryLinear layers are untouched here -- call both functions
    together for full coverage.
    """
    lib_path = lib_path or _DEFAULT_LIB_PATH
    meta_path = os.path.join(packed_weights_dir, "packed_weights.json")
    bin_path = os.path.join(packed_weights_dir, "packed_weights.bin")

    with open(meta_path) as f:
        meta = json.load(f)
    with open(bin_path, "rb") as f:
        raw = f.read()

    layers_by_name = {layer["name"]: layer for layer in meta["layers"]}
    forward_fn = _load_batched_forward_fn(lib_path)

    def _load_packed(layer_meta: dict) -> np.ndarray:
        chunk = raw[layer_meta["byte_offset"]: layer_meta["byte_offset"] + layer_meta["byte_length"]]
        return np.frombuffer(chunk, dtype=np.uint64).reshape(layer_meta["out_features"], layer_meta["k_packed"])

    patched = 0
    for attn_name, attn in _find_hybrid_attention_layers(model):
        q_name = f"{attn_name}.q_proj.binary_linear"
        k_name = f"{attn_name}.k_proj.binary_linear"
        v_name = f"{attn_name}.v_proj.binary_linear"
        if q_name not in layers_by_name or k_name not in layers_by_name or v_name not in layers_by_name:
            continue

        q_meta, k_meta, v_meta = layers_by_name[q_name], layers_by_name[k_name], layers_by_name[v_name]
        k_packed = q_meta["k_packed"]
        if k_meta["k_packed"] != k_packed or v_meta["k_packed"] != k_packed:
            continue  # shapes must match to fuse; skip rather than guess

        fused_packed = np.concatenate(
            [_load_packed(q_meta), _load_packed(k_meta), _load_packed(v_meta)], axis=0
        ).copy()

        attn.forward = _FusedQKVForward(
            attn, forward_fn, fused_packed, dim=q_meta["out_features"], k_packed=k_packed,
        )
        patched += 1

    return patched
