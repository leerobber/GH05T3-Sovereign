# 0005: mmap-backed BinaryLedger for agent-swarm state

**Status:** Accepted, live in `backend/oss/core/binary_ledger.py`, `backend/oss/swarm/`

## Context

A 32-byte-slot binary mmap format for agent-swarm state (7 float16
"desire" dimensions, maturity, fitness, parent-slot lineage, heartbeat,
bit-flagged scratchpad) was documented in a Claude skill
(`binary-ledger`, installed in the *original* GH05T3 repo, not this one)
and independently corroborated by a real data file,
`backend/data/aethyro_swarm.bin`. `.pyc` bytecode remnants in both repos
suggested `SentinelPrime`/`LivingLoop` were once real Python classes
implementing something like this — their source doesn't survive anywhere
found.

## Decision

Built `BinaryLedger` (deliberately not named `ChronosLedger` — that name
is already used by the unrelated architecture-genome ledger from
[0004](0004-genome-evolution-subsystem.md)) implementing the documented
API: read/write/vacancy-scan/lineage/scratchpad-bits/vectorized-numpy-views/stats.
Verified byte-identical (MD5) against the real data file before and after
the test run — this is a different concept from the architecture-genome
system: it tracks agent SWARM/PERSONA state (`num_layers`/`dim`/etc.
architecture genomes are a different "genome" that happens to share
vocabulary), deliberately not merged with it.

`backend/oss/swarm/`: `swarm_agent.py`'s `load_active_agents(ledger)` reads
real personas out of a `BinaryLedger`. `agent_swarm_runtime.py`'s
`AgentSwarmRuntime` runs one real batched forward pass for a group of
agents through one shared `GH05T3BinaryOSS` model, with
`enable_fast_inference()` wired in directly (the real function from
[0002](0002-rust-kernel-strategy.md), not a reimplementation).

## Consequences

- This repo has its own untracked, gitignored `aethyro_swarm.bin`,
  distinct (different MD5, newer, more real activity) from the original
  GH05T3 repo's copy — not overwritten with the older file.
  `.gitignore` had a real bug (blanket-excluding `backend/data/` twice
  over, via both `*.bin` and the directory pattern) that was fixed so this
  data is actually committed instead of floating untracked.
- Real, undesigned gaps, not invented around: every agent in a swarm group
  currently shares one model (the 32-byte format has no spare field for a
  per-agent config reference), and persona doesn't yet influence
  inference.
- A second real file, `genome_plane.bin` (262144 bytes), sits in the same
  data directory in the original GH05T3 repo — not yet investigated or
  used from this repo.
