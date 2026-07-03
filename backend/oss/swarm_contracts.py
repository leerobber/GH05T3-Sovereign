"""
Swarm Contract Engine (Phase 2+)

OmniMind assigns tasks to swarms with shared rewards.
This creates collective intelligence.
"""

from __future__ import annotations
from typing import Dict, Any, List
import random
from backend.oss.omni_mind import OmniMind
from backend.oss.omni_economy import OmniEconomy
from backend.oss.genomic_substrate import GenomicSubstrate


class SwarmContractEngine:
    def __init__(self, mind: OmniMind, economy: OmniEconomy, substrate: GenomicSubstrate):
        self.mind = mind
        self.economy = economy
        self.substrate = substrate
        self.contracts: Dict[str, Dict[str, Any]] = {}

    def create_contract_from_goal(self, goal: Dict[str, Any], reward: float) -> str:
        cid = f"contract_{goal['goal_id']}"
        self.contracts[cid] = {
            "goal": goal,
            "reward": reward,
            "participants": [],
            "completed": False,
        }
        return cid

    def assign_swarm(self, contract_id: str, max_agents: int = 5) -> List[str]:
        contract = self.contracts[contract_id]
        required = contract["goal"].get("required_traits", [])
        selected = []

        candidates = list(self.substrate.genomes.items())
        # Prefer Theorist Elite for theory goals
        goal_desc = contract.get("goal", {}).get("description", "").upper()
        if "THEORY" in goal_desc or "FORMAL" in goal_desc or "META" in goal_desc:
            candidates.sort(key=lambda x: 0 if "THEORIST" in x[1].role.upper() else 1)

        for gid, rec in candidates:
            dna = rec.dna
            has_traits = all(dna.get_traits().get(t, 0.0) >= 0.5 for t in required)
            if has_traits and random.random() < 0.8:
                selected.append(gid)
            if len(selected) >= max_agents:
                break

        contract["participants"] = selected
        return selected

    def execute_contract(self, contract_id: str, task_prompt: str) -> Dict[str, Any]:
        contract = self.contracts[contract_id]
        results = []

        for gid in contract["participants"]:
            agent = self.substrate.spawn_agent(gid)
            action = agent.act({"prompt": task_prompt})
            results.append({"genome_id": gid, "output": action["raw_output"]})

        contract["completed"] = True
        return {"contract_id": contract_id, "results": results}

    def distribute_rewards(self, contract_id: str):
        contract = self.contracts[contract_id]
        if not contract["completed"]:
            return

        reward = contract["reward"]
        participants = contract["participants"]
        if not participants:
            return

        share = reward / len(participants)
        for gid in participants:
            self.economy.reward(gid, share, reason=f"swarm_contract:{contract_id}")
