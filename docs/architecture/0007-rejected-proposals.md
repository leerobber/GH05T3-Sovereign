# 0007: Rejected proposals

**Status:** Reference only — none of these were merged. Recorded so they
aren't re-proposed and re-investigated from scratch.

## C-ABI `EngineConfig`/`ffi_config.rs` config-push design

Proposed: a Rust `GMLContext` holding an opaque config struct, updated/read
via ctypes from `BMEBridge`/`SwarmRuntime`, as a "shadow engine" mirroring
the PyTorch model's configuration on the Rust side.

Rejected because:
- `binary_inference::config`, `ternary_inference`, `orthogonal_residual`
  modules referenced by the proposal don't exist in `gml_kernel` (only
  `bin`, `binary_inference`, `gh05t3`, `kernel`, `naming`, `runtime` do).
- `AgentRuntime::new()` and `GlyphBlock::for_task()`, also referenced,
  don't exist (zero grep matches).
- `static mut ENGINE_STATE` is unsound, and contrasts directly with the
  already-correct `Mutex<AgentRuntime>`/`Mutex<KernelState>` pattern this
  crate already uses (see [0003](0003-async-runtime.md)).
- No Rust-side "shadow engine" mirroring the PyTorch components exists or
  is architecturally sensible — the real model is the Python object,
  already correctly configured via `BMEBridge` ([0004](0004-genome-evolution-subsystem.md)).
- The proposed `SwarmRuntime.evaluate_genome` replacement returned
  `{"score": 0.0, "status": status}` — a fabricated-score placeholder, the
  same anti-pattern as an even earlier rejected rebuild spec that used
  `{"score": 0.75, ...}`.
- Two rounds of real compile-testing on the actual proposed Rust each
  produced fresh compile errors (first: 8 duplicate-symbol errors from an
  `extern "C" {}` block colliding with its own `#[no_mangle]` definitions;
  "corrected" version: 4 new pointer-cast/deref errors).

## "INR" quantization mode

Proposed as an additional `out_proj_quant_mode` option ("infinite
functional substrate"). Rejected: already benchmarked and rejected earlier
in this project — 17x slower than a real matmul, non-learnable.

## "sovereign_graph.rs" 84-layer monolith

Proposed 3 times, verbatim/near-verbatim, across this project. Rejected
every time: a complete no-op (never populates weights) and topologically
wrong (the real block has 4 distinct layer shapes, parallel q/k/v
branches, real softmax attention, GELU-gated MLP — not a flat chain),
confirmed by reading the real `BinaryTransformerBlock`/`HybridBinaryAttention`
code directly each time.

## Fused-QKV Rust kernel draft

Referenced a `.bias` attribute that doesn't exist on `BinaryLinear`/
`MagnitudeAwareINBL` (confirmed by grep), processed one token per call
(would need a call per token — worse than what exists), and used unpacked
i8 weights with no AVX-512 (a double regression vs. the real kernel). The
real fix for this exact problem ended up needing zero new Rust code — see
`enable_qkv_fusion()` in [0002](0002-rust-kernel-strategy.md), which just
batches calls into the existing verified kernel.

**Pattern across all of the above:** before writing new Rust or accepting
a pasted design, check whether the referenced modules/functions actually
exist (`grep`), and whether the fix is "write new code" vs. "call the
existing, already-verified code smarter."
