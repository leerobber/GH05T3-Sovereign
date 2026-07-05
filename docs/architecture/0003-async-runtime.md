# 0003: Async episode-loop runtime wraps the sync core loop

**Status:** Accepted, live in `gml_kernel::runtime`

## Context

The existing `kernel::executor`/`gh05t3::core_loop` engine is synchronous.
An agent/episode loop (policy → tool calls → episode bookkeeping) benefits
from async I/O (concurrent tool calls, streaming), but the core glyph
execution logic itself doesn't need to change to get that.

## Decision

Add `gml_kernel::runtime`, a tokio-based async layer (`ModelPolicy`/`Agent`/
`ToolBus`/`EpisodeManager`) that **wraps** the existing synchronous engine
rather than reimplementing or replacing it. `Gh05t3Policy::infer` runs the
real glyph core loop each tick and reports real executed glyphs — not a
stub standing in for it. A new binary, `gh05t3_runtime`, runs this
alongside the existing `gml_kernel_main` binary; neither replaces the
other.

## Consequences

- The question "keep the async runtime or rewrite it synchronous" was
  raised and closed: keeping the async version, no rewrite planned.
- Shared mutable engine state uses `Mutex<AgentRuntime>`/`Mutex<KernelState>`
  (in `runtime::model_policy::Gh05t3Policy`) — this is the correct,
  already-established pattern in this crate. A later proposal
  (`ffi_config.rs`) used `static mut ENGINE_STATE` instead, which is
  unsound and was rejected partly on that basis — see
  [0007](0007-rejected-proposals.md).
