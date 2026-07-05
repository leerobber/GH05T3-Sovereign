# 0002: Rust AVX-512 kernels via ctypes FFI, not pyo3

**Status:** Accepted, live in `gml_kernel/`

## Context

The quantized model's binary/ternary matmuls are the hot path. A native
kernel is worth having, but it needs to be callable from the existing
PyTorch model without restructuring the whole inference path around a
different binding mechanism.

## Decision

- `gml_kernel::binary_inference` implements real AVX-512 **signed-accumulation**
  kernels for both binary and ternary weights — addition-only, no
  XNOR+popcount, because this model's activations are never binarized (see
  [0001](0001-quantization-and-attention.md)). Verified against PyTorch
  references and cross-checked against each other (ternary-with-all-nonzero-weights
  == binary, verified directly).
- Bound via **ctypes**, not pyo3. A C-ABI shared library callable from
  plain Python needs no build-time coupling between the Rust toolchain and
  the exact PyTorch/Python ABI version — pyo3 would tie the kernel's build
  to a specific Python version and complicate the "just call a `.so`" story
  this repo already uses successfully.
- `enable_fast_inference()` wires the real kernel into the model's forward
  pass directly (not a reimplementation of the kernel-calling code).
- `enable_qkv_fusion()`: each attention block's q/k/v projections share the
  identical input, so they're batched into **one** kernel call instead of
  three, using the already-verified kernel — zero new Rust code. Cuts
  84→60 total FFI crossings across the real 12-block model. Measured
  result: 345.7ms → 338.4ms, a real but modest ~2% win — other overhead
  (numpy conversions, Python orchestration) still dominates.

## Consequences / open items

- A genuine whole-model-forward FFI call (batching LayerNorm/softmax/GELU-adjacent
  linear/quantized sub-steps into fewer, larger Rust calls, while leaving
  LayerNorm/softmax/GELU themselves in PyTorch where they're already fast)
  is real and worth doing — not done yet.
- **Dead end, don't retry:** naive `rayon::par_iter_mut()` parallelization
  of `forward_layer`/`ternary_forward_layer`'s `out_features` loop measured
  **32x slower** (703.8us → 22488.2us isolated; 316.8ms → 1716ms full-model).
  Reverted immediately. Cause: rayon's per-call work-stealing dispatch
  overhead vastly exceeds the actual work per row (a few hundred
  nanoseconds) at `out_features=1024` — this granularity is far too fine
  for a fresh-per-call thread pool. If threading is revisited, it needs
  either a persistent/warm thread pool or a coarser granularity (across
  layers/blocks, not across one layer's output rows), documented directly
  in `binary_inference/mod.rs`'s docstrings.
- See [0007](0007-rejected-proposals.md) for kernel-design proposals that
  were rejected before being merged (the "sovereign_graph.rs" monolith, a
  fused-QKV kernel draft with a nonexistent `.bias` attribute, and the
  C-ABI `EngineConfig`/`ffi_config.rs` design).
