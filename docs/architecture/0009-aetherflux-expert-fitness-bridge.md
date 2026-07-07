# 0009: Aetherflux expert fitness bridge

**Status:** Accepted, live in `backend/oss/swarm/aetherflux_bridge.py`

## Context

A separate repo, `leerobber/aetherflux-zero` (a from-scratch Rust
ternary-weight/MatMul-free neural stack, unrelated to this repo's own
`gh05t3_binary`/`gml_kernel` binary transformer work), trains one real
ternary char-level language model per real, titled knowledge domain found
in its own corpus, plus a router (`TrainableMoBERouter`) that learns to
pick the right specialist for a given held-out text window. Its
`examples/specialist_pilot.rs` measures, per domain, real held-out routing
recall/precision and whether the specialist's own held-out loss actually
beat its cross-domain loss (`specialized: bool`) — real numbers from a real
run, not asserted.

That measurement had nowhere to live except a terminal printout. This repo
already has the right substrate for treating it as a real fitness signal:
[0005](0005-binary-ledger-and-swarm.md)'s `BinaryLedger` — a real,
mmap-backed, 32-byte-per-agent format with a dedicated `fitness` field
enforced to `[0,1]`, plus a `SCRATCH_NEEDS_REVIEW` bit already documented
for exactly this kind of "flag this one" signal.

## Decision

Built `backend/oss/swarm/aetherflux_bridge.py`, co-located with
[0005](0005-binary-ledger-and-swarm.md)'s `swarm_agent.py` (which already
reads personas out of a `BinaryLedger` the same way). It reads
aetherflux-zero's JSON export (produced separately, by running that repo's
own `cargo run --example specialist_pilot` — this repo has no Rust
toolchain dependency of its own) and, per domain:

- Uses **recall** (not an invented credit/reward unit) as the `fitness`
  value written via `update_fitness`/`write_at_next_available_slot` —
  recall is already bounded in `[0,1]` by construction (fraction of a
  domain's own held-out text the router actually recognized as belonging to
  it), so no conversion factor is needed the way a credit/penalty score
  would need one. `precision` is exported alongside it for transparency but
  not used as the ledger value.
- Sets `SCRATCH_NEEDS_REVIEW` when the domain's specialist did *not* beat
  cross-domain loss (`specialized: false`) — a direct, real use of an
  existing documented field, not a new one. On the 15-domain run measured
  this session, that's 2 of 15 domains (Memetics, Physarum Polycephalum).
- Is idempotent across repeated runs via a small JSON side-file,
  `backend/data/aetherflux_slots.json` (domain name → slot index), added to
  `.gitignore`'s existing `!backend/data/aethyro_swarm.bin` exception for
  the same reason: slot indices are only meaningful relative to one
  specific ledger file, so this side-file needs to travel with it, not be
  silently excluded by the blanket `backend/data/*` rule.
- Defaults to **dry-run** (prints the computed slot/action/fitness/flag
  table, opens no ledger handle at all) — a real, hard-to-undo mmap write
  into a live, shared ledger file is exactly the kind of action that should
  require an explicit `--live` flag, not fire by default the way this
  repo's other reward-style hooks do.

## Consequences

- This wires the *measurement* into the *ledger*. It does **not** yet feed
  `genesis-ops`' spawn/prune logic or `bme`'s speciation/role-tier
  promotion — real, separate, unbuilt future work, not a silent gap papered
  over.
- Every aetherflux domain currently gets neutral `desires` (`0.5` for all
  seven dimensions) — how a routing-recall signal should shape a swarm
  agent's psychological desire vector isn't specified anywhere real;
  guessing a mapping would be fabricating behavior, the same restraint
  [0005](0005-binary-ledger-and-swarm.md)'s `swarm_agent.py` already
  applies to persona-shaping inference.
- This repo's own genome/economy duplication with GH05T3 main
  ([0006](0006-relationship-to-gh05t3-and-sovereign-core.md), still
  unresolved) means there may eventually be a *second*, GH05T3-main-side
  place this same signal could plug in (`OmniEconomy`/`SpeciesMemory`) —
  deliberately not built here; this ADR only resolves where aetherflux's
  signal lands in *this* repo.
