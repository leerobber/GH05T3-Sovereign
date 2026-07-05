# 0001: Binary/ternary weight quantization + MA-INBL attention

**Status:** Accepted, live in `gh05t3_binary/`

## Context

The engine needs a transformer that runs entirely on local hardware without
depending on full-precision weights. The two established approaches are
binary weights (BitNet-style, 1 bit/weight, always ±magnitude) and ternary
weights (Ternary Weight Networks, Li & Liu 2016, 2 bits/weight, {-1,0,+1}).

## Decision

- `BinaryLinear`: weights quantized via `_SignSTE` (straight-through
  estimator — forward uses `torch.sign(weight)`, backward passes the
  gradient through unchanged). The live fp32 `weight` is used directly in
  the graph so gradients actually reach it — an earlier version cached a
  detached `torch.sign(weight)` buffer, which silently made the weight
  untrainable.
- `TernaryLinear`: weights quantized via `_TernarySTE` with threshold
  `delta = 0.7 * mean(|W|)` (the paper's fixed constant) by default, or
  `torch.quantile(|W|, sparsity_target)` when a genome sets
  `sparsity_target` (see [0004](0004-genome-evolution-subsystem.md)) —
  the quantile form genuinely targets that fraction of weights going to
  zero, unlike exposing `0.7` as a tunable multiplier, which would still
  be data-dependent and only approximately related to the resulting
  sparsity.
- `MagnitudeAwareINBL` (MA-INBL): splits input into direction (binarized)
  and magnitude (4-bit uniform-quantized via `UniformQuantizer`), reintegrates
  both. Used for Q/K/V projection in `HybridBinaryAttention`.
- `HybridBinaryAttention.out_proj`: originally a full-precision `nn.Linear`
  — the only unquantized component in the model — then fixed to
  `TernaryLinear` specifically, since a real zero state right before the
  residual add lets a head's contribution actually be dropped for a given
  output feature, instead of always adding ±magnitude noise. Retrained from
  scratch after the swap; final loss ~6.48 train / 6.62 val, matching
  pre-swap numbers — no regression, no proven gain either, just one honest
  run.
- `out_proj_quant_mode` ("ternary" | "binary") and `mainbl_threshold`
  (real magnitude gate on MA-INBL, default `0.0` = disabled, byte-identical
  to prior behavior) are both real, evolvable genome traits — not just
  static config.

## Consequences

- Model size: ternary is 16x smaller than fp32 (2 bits/weight), binary is
  32x smaller (1 bit/weight) — `out_proj_quant_mode` lets a genome trade
  between them.
- Activations are **not** quantized — only weights. The original GH05T3
  `oss/quantization/binary_quant.py` (recovered via `git show
  6d133ac:oss/quantization/binary_quant.py` in the original repo, doesn't
  survive to that repo's `HEAD`) also did per-token absmax INT8 activation
  quantization, true BitNet b1.58 style. Matching that here would need a
  different computational graph and a full retrain — deliberately deferred
  as its own larger task, not silently skipped.
- `BinaryTransformerBlock`'s residual handling was fixed alongside this:
  each sublayer's output is diversified once via
  `OrthogonalResidualDecomposer` and added back exactly once with
  `MagnitudeGrowthClamper`/`DepthAwareMagnitudeGate` — a previous version
  applied three stabilizers as three independent additions of the same
  output, and separately computed a block-level q/k/v that was never used.
