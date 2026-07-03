# backend/oss/swarm/theorist_swarm.py

from typing import Dict, Any, List
from backend.oss.mvs import OmniMind
from backend.oss.omni_economy import OmniEconomy
from backend.oss.genomic_substrate import AgentHandle


class TheoristSwarmEngine:
    """
    Swarm reasoning engine restricted to THEORIST_ELITE agents.
    """

    def __init__(self, mind: OmniMind, economy: OmniEconomy):
        self.mind = mind
        self.economy = economy

    def get_theorists(self) -> List[str]:
        return [
            gid for gid, rec in self.mind.agents.items()  # adjust if needed
            if getattr(rec, "role", getattr(rec.dna if hasattr(rec, 'dna') else None, 'role', '')) == "THEORIST_ELITE"
        ]

    def run_swarm(self, prompt: str, reward: float = 300.0) -> Dict[str, Any]:
        theorists = self.get_theorists()
        results = []

        for agent_id in theorists:
            # adapt to current
            # assume mind has way to get dna
            dna = getattr(self.mind, 'agents', {}).get(agent_id) or None
            if not dna:
                continue
            handle = AgentHandle(agent_id, "THEORIST_ELITE", dna, context={})
            action = handle.act({"prompt": prompt})
            results.append({"agent_id": agent_id, "output": action["raw_output"]})

        # simple consensus: pick longest / richest output
        best = max(results, key=lambda r: len(r["output"])) if results else {}
        share = reward / max(len(theorists), 1)
        for agent_id in theorists:
            self.economy.reward(agent_id, share, reason="theorist_swarm")

        return {"prompt": prompt, "results": results, "consensus": best}
