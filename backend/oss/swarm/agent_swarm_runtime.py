"""Orchestrates real forward passes for a swarm of SwarmAgents.

Named AgentSwarmRuntime, not SwarmRuntime -- that name is already the
real, tested, live-API-wired class in swarm/swarm_runtime.py, which
evaluates a single architecture-genome's fitness. This is a different
job: running many agents through a real forward pass together.

Uses PyTorch's native batch dimension for "batching" -- genuinely real
and correct, and doesn't need any custom Rust-side grouping logic yet,
since every agent in this first pass shares one model (see
swarm_agent.py's docstring for why per-agent configs aren't wired up
yet). Real AVX-512 kernel acceleration is wired in via
enable_fast_inference() below, which calls the same, already-verified
gh05t3_binary.inference.fast_inference.enable_fast_inference() used by
the genome-evaluation path -- not a reimplementation. Known limitation,
inherited from that function: it only patches true BinaryLinear
instances (found via isinstance), not TernaryLinear (e.g.
attention.out_proj) -- so even with fast inference enabled, out_proj
still runs the plain PyTorch ternary forward.
"""
from __future__ import annotations

import torch

from backend.oss.swarm.swarm_agent import SwarmAgent
from gh05t3_binary.oss.integration import GH05T3BinaryOSS


class AgentSwarmRuntime:
    def __init__(self, model: GH05T3BinaryOSS, packed_weights_dir: str | None = None):
        self.model = model
        self.model.eval()
        self.fast_inference_enabled = False
        self.patched_layer_count = 0
        if packed_weights_dir is not None:
            self.enable_fast_inference(packed_weights_dir)

    def enable_fast_inference(self, packed_weights_dir: str, lib_path: str | None = None) -> int:
        """Patches this runtime's model's real BinaryLinear instances to
        use the verified Rust AVX-512 kernel (see
        gh05t3_binary.inference.fast_inference.enable_fast_inference --
        this calls that function directly, not a reimplementation).

        Requires packed weights already exported (pack_weights.py) from
        a checkpoint with the SAME architecture as this runtime's model
        -- a shape mismatch surfaces as a real error from the
        underlying kernel, not something silently swallowed here.

        Entirely optional: run_batch works against the plain PyTorch
        model if this is never called, or if it's called but patches
        zero layers (e.g. an untrained/empty checkpoint)."""
        from gh05t3_binary.inference.fast_inference import enable_fast_inference as _enable_fast_inference

        patched = _enable_fast_inference(self.model, packed_weights_dir, lib_path=lib_path)
        self.fast_inference_enabled = patched > 0
        self.patched_layer_count = patched
        return patched

    def run_batch(self, agents: list[SwarmAgent], input_ids: torch.Tensor) -> dict[int, torch.Tensor]:
        """Runs ONE real forward pass over `input_ids` (shape
        [len(agents), seq_len]) through the shared model, returning each
        agent's own logits row keyed by slot_index. Row i of input_ids
        is assumed to belong to agents[i] -- caller's responsibility to
        keep these aligned (real, minimal contract; no hidden per-agent
        state lives inside this call)."""
        if input_ids.shape[0] != len(agents):
            raise ValueError(
                f"input_ids has {input_ids.shape[0]} rows but {len(agents)} agents were given -- "
                "each row must correspond to exactly one agent"
            )

        with torch.no_grad():
            logits = self.model(input_ids)  # [num_agents, seq_len, vocab_size]

        return {agent.slot_index: logits[i] for i, agent in enumerate(agents)}
