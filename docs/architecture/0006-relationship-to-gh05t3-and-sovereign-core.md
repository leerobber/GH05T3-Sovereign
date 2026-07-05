# 0006: Relationship to GH05T3 and sovereign-core

**Status:** Accepted (federation over merging), with an open reconciliation item

## Context

This repo is one of at least three places the same underlying ideas
(quantized inference, genetic-algorithm-style architecture/hyperparameter
evolution, an "economy" of credits between agents) have been built:

1. **GH05T3** (`leerobber/GH05T3`) — this repo's origin, described on
   GitHub as "built from the GH05T3 architecture." Critically, this repo
   was forked from GH05T3's `claude/fix-multi-gpu-training-2WHKH` feature
   branch, which diverged from GH05T3's own `main` on 2026-05-22. As of
   this writing `main` has 133 commits that branch never received,
   including a materially more complete genome/evolution/economy system
   (`OmniDNA`, `oss/ecosystem` species-FSM, a real SQLite economy ledger,
   real Stripe-settled revenue) than the 5-trait system built here in
   [0004](0004-genome-evolution-subsystem.md).
2. **sovereign-core** (`leerobber/sovereign-core`) — a separate,
   parallel ecosystem/brand (SovereignNation, AGPL, paid tiers) with its
   own KAIROS/SAGE self-improvement loop and its own economy modules,
   built independently on the same physical hardware
   (RTX 5050 / Radeon 780M / Ryzen 7).
3. This repo.

sovereign-core already has a real, working (if partial) answer to
"how should these relate": `src/orchestration/registry.py` +
`config/registry.yaml`, an HTTP health/role registry that probes GH05T3's
real gateway (dual-runtime WSL/Windows aware) as an external "silo,"
rather than importing its code.

## Decision

**Federate over HTTP, don't merge codebases.** This repo's
`/oss/genome/*` API is registered in sovereign-core's `registry.yaml`
as `gh05t3_sovereign`, alongside the `gh05t3` entry — both probed the same
way, both documented as sharing the same `GATEWAY_PORT` default (8002),
requiring the env var override when running both on one host.

This mirrors a change already made independently inside sovereign-core
itself: `nightly_full_evolution.py` was switched to drive its mesh via
HTTP instead of importing local KAIROS modules directly — i.e., "federate,
don't import" is an already-validated pattern there, not a new one
invented for this repo.

## Consequences / open item

- **Not yet resolved:** whether this repo's 5-trait genome subsystem
  should be folded into GH05T3 `main`'s more complete `OmniDNA`
  substrate, or deliberately kept as a separate "clean room" rebuild. Right
  now there are four independent evolution/genome-adjacent systems in this
  whole ecosystem (GH05T3 `main`'s `OmniDNA`+`GenomicSubstrate`, GH05T3
  `main`'s separate `oss/ecosystem` species-FSM+economy, this repo's
  5-trait system, and sovereign-core's KAIROS/DGM) and none of them talk
  to each other beyond the health-probe level above. Full inventory,
  updated after `oss/ecosystem` was wired to real telemetry (it no longer
  runs on a synthetic sandbox — real KAIROS/ledger/Stripe signals now):
  [GH05T3's `docs/architecture/evolution-systems-inventory.md`](https://github.com/leerobber/GH05T3/blob/main/docs/architecture/evolution-systems-inventory.md).
- The Rust kernel work in this repo ([0002](0002-rust-kernel-strategy.md))
  and sovereign-core's own `sovereign-core-rs`+`sovereign-gpu` (a general
  WGSL/CUDA/Vulkan host runtime with WASM agents, started the same week)
  are a second instance of parallel, non-communicating development —
  lower-priority to reconcile since they target different problems
  (bit-packed AVX-512 kernels for one specific model vs. a general
  portable-GPU-compute runtime), but worth a deliberate check-in before
  more time is invested in either.
- A prior proposal to solve this via a C-ABI config-push mechanism
  (`ffi_config.rs`) instead of HTTP federation was rejected — see
  [0007](0007-rejected-proposals.md).
