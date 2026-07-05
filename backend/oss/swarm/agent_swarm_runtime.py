"""Orchestrates real forward passes for a swarm of SwarmAgents.

Named AgentSwarmRuntime, not SwarmRuntime -- that name is already the
real, tested, live-API-wired class in swarm/swarm_runtime.py, which
evaluates a single architecture-genome's fitness. This is a different
job: running many agents through a real forward pass together.

Uses PyTorch's native batch dimension for "batching" -- genuinely real
and correct, and doesn't need any custom Rust-side grouping logic yet,
since every agent in this first pass shares one model (see
swarm_agent.py's docstring for why per-agent configs aren't wired up
yet). Real AVX-512 kernel acceleration (gh05t3_binary.inference.
fast_inference.enable_fast_inference) can be layered onto the shared
model exactly like it already is for the genome-evaluation path -- not
duplicated here.
"""
from __future__ import annotations

import torch

from backend.oss.swarm.swarm_agent import SwarmAgent
from gh05t3_binary.oss.integration import GH05T3BinaryOSS


class AgentSwarmRuntime:
    def __init__(self, model: GH05T3BinaryOSS):
        self.model = model
        self.model.eval()

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
