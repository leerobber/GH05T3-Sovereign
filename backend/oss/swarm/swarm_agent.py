"""Real SwarmAgent: ties together two real subsystems built this session
-- BinaryLedger (agent persona/lineage/fitness, backend/oss/core/
binary_ledger.py) and the real GH05T3BinaryOSS model
(gh05t3_binary.oss.integration) -- into one coherent agent concept.

Scoped narrowly for this first pass, deliberately:
- Every agent currently shares ONE model instance (see
  AgentSwarmRuntime), since the real BinaryLedger's 32-byte slot format
  has no spare field for a per-agent model/config reference -- all 32
  bytes are already accounted for by the documented format (desires,
  maturity, fitness, lineage, generation, heartbeat, scratchpad). Giving
  each agent its own architecture needs either extending that format (a
  real, consequential change to a format that's supposed to match an
  external spec) or a separate agent_id -> genome_id mapping -- real,
  undesigned work, not invented here.
- `persona` is read and carried per agent but NOT used to influence
  inference yet. How a 7-float desire vector should shape a
  transformer's forward pass isn't specified anywhere real; guessing
  (e.g. concatenating it into the input embedding) would be fabricating
  behavior, not building it.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.oss.core.binary_ledger import DESIRE_NAMES, BinaryLedger


@dataclass
class SwarmAgent:
    slot_index: int
    persona: dict[str, float]  # keys are DESIRE_NAMES
    fitness: float
    generation: int
    parent_offset: int


def load_active_agents(ledger: BinaryLedger) -> list[SwarmAgent]:
    """Reads every currently-active slot (fitness != 0.0, the documented
    vacancy sentinel) from a real BinaryLedger into SwarmAgent objects.
    Read-only -- never mutates the ledger."""
    agents = []
    for i in range(ledger.active_count):
        record = ledger.read_agent(i)
        if record["fitness"] == 0.0:
            continue
        agents.append(
            SwarmAgent(
                slot_index=i,
                persona=dict(record["desires"]),
                fitness=record["fitness"],
                generation=record["generation"],
                parent_offset=record["parent_offset"],
            )
        )
    return agents
