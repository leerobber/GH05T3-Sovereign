# Architecture Decisions

This folder is the durable record of what was actually built in this repo,
why, and what was tried and rejected. Each entry is a lightweight ADR
(Architecture Decision Record): Status / Context / Decision / Consequences.

Written from real code and real test runs, not aspiration — if a decision
below turns out to be stale, trust the code over this document and update
the doc.

| # | Decision | Status |
|---|---|---|
| [0001](0001-quantization-and-attention.md) | Binary/ternary weight quantization + MA-INBL attention | Accepted, live |
| [0002](0002-rust-kernel-strategy.md) | Rust AVX-512 kernels via ctypes FFI, not pyo3 | Accepted, live |
| [0003](0003-async-runtime.md) | Async episode-loop runtime wraps the sync core loop | Accepted, live |
| [0004](0004-genome-evolution-subsystem.md) | Real trait-driven genome/evolution subsystem | Accepted, live |
| [0005](0005-binary-ledger-and-swarm.md) | mmap-backed BinaryLedger for agent-swarm state | Accepted, live |
| [0006](0006-relationship-to-gh05t3-and-sovereign-core.md) | Relationship to GH05T3 and sovereign-core | Accepted, ongoing tension noted |
| [0007](0007-rejected-proposals.md) | Rejected designs (why, and what was wrong with each) | Reference only |
| [0008](0008-remove-hardcoded-slack-oauth-values.md) | Remove hardcoded Slack OAuth values | Resolved |
| [0009](0009-ci-and-torch-dependency-fix.md) | Add CI, declare the missing torch dependency | Resolved |

## How to add a new one

Copy the format of any existing entry, number it sequentially, and add a
row to the table above. Prefer documenting a real decision after it's been
implemented and tested over speculating about one in advance.
