# 0004: Real trait-driven genome/evolution subsystem

**Status:** Accepted, live in `backend/oss/`, exposed at `/oss/genome/*`

## Context

The original GH05T3 repo had a genome/evolution subsystem
(`genomic_substrate.py`, `elite/`, `hyper_elite/`, `speciation.py`, etc.)
that was deleted wholesale in commit `98b9ee8` on that repo's history —
only `.pyc` bytecode remnants survive there, no source. This repo needed
a real, working replacement, not a reconstruction from bytecode.

**Correction / open item (added after later investigation):** GH05T3's
own `main` branch (not the branch this repo was forked from) turned out to
have a *much* more complete real genome/evolution/economy system —
`OmniDNA`, `GenomicSubstrate`, `OmniMind`, `OmniEconomy`,
`SpeciesMemory`, `SpeciationEngine`, plus a separate `oss/ecosystem`
species-FSM. This repo's genome subsystem was built without knowing that
existed, from a feature branch that had diverged from `main` before it was
added. See [0006](0006-relationship-to-gh05t3-and-sovereign-core.md) —
this is unresolved duplication, not a settled decision.

## Decision

Built from scratch against real, already-verified components, not
bytecode: `GenomePlane` (real encode/decode round trip), `MutationOperator`
protocol (`is_applicable`/`create_mutation`) with concrete operators,
`OmniEvolutionEngine`, `ChronosLedger`/`SpeciesMemory`, `BMEBridge`
(genome traits → a real `GH05T3BinaryOSS` PyTorch model), and
`SwarmRuntime.evaluate_genome` (a real forward pass + real cross-entropy
loss — never a fabricated/hardcoded score).

**`BMEBridge`** caches one model per `genome_id` (not per trait-hash,
since two genomes with identical traits are still independent
evolutionary samples), seeded deterministically via
`sha256(genome_id)[:8]` (not Python's randomized `hash()`, which changes
per process) so the same genome gives the same real score across restarts.

**Five evolvable traits**, added one at a time, each following the same
pattern (real constructor param → threaded through all 4 model levels →
wired into `BMEBridge` → a `MutationOperator` → registered in
`OmniEvolutionEngine` → live-verified over real HTTP, not just unit
tests):

| Trait | Mutation style | Why |
|---|---|---|
| `binary_ratio` | additive ±0.02, clamped [0.5, 1.0] | bounded fraction |
| `stabilizer` | flip mgc↔damg | binary choice |
| `out_proj_quant_mode` | flip ternary↔binary | binary choice |
| `mainbl_threshold` | **log-space**, factor `exp(±0.4)`, floor `1e-3` from 0.0 | wide dynamic range (0.001 and 1.0 are both real settings) |
| `ternary_sparsity_target` | additive ±0.05, clamped [0.1, 0.9] | already a bounded fraction, and only meaningful when `out_proj_quant_mode == "ternary"` |

## Consequences

- Live-verified via the real HTTP API (not just pytest): e.g. registering
  ternary-sparsity genomes at 0.2 and 0.8, evaluating both, getting
  distinct real losses (8.577 vs 7.285), then `/oss/genome/evolve`
  correctly firing `TernarySparsityMutation` for both and producing real
  children with their own distinct scores.
- `GenomePlane`/`ChronosLedger` are in-memory only — genomes, scores, and
  history are lost on restart. Documented limitation, not a silent gap;
  real persistence is future work if this needs to survive restarts.
- Rejected during this build: a C-ABI `GMLContext`/`EngineConfig` design
  that would have replaced this Python-side bridge with a Rust-side
  "shadow engine" — see [0007](0007-rejected-proposals.md).
