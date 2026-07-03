"""
ContextWorld — sparse-context pressure challenges.

Agents are given tasks with intentionally thin context and rewarded for
extracting maximum signal from minimum input. Forces evolution of:
  - M_CONTEXT_COMPRESSION   (extract more from less)
  - M_CONTEXT_SYNTHESIS     (combine sparse signals)
  - M_CONTEXT_EFFICIENCY    (no wasted context budget)
  - M_CONTEXT_ANTICIPATION  (predict what's missing)

Flow for an agent:
  1. World strips context down to a sparse representation.
  2. Agent acts on the sparse prompt.
  3. World scores: how well did the agent recover the full meaning?
  4. Fitness bonus scales with compression ratio achieved.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ContextChallenge:
    challenge_id: str
    full_prompt:  str
    sparse_prompt: str          # stripped-down version given to agent
    keywords:     List[str]     # ground-truth signal the agent should recover
    compression_ratio: float    # how aggressively context was stripped (0-1)
    domain:       str


class ContextWorld:
    """
    Sparse-context challenge environment.
    Agents that recover more semantic meaning from thinner context
    receive higher fitness scores.
    """

    def __init__(self):
        self.challenges = self._build_challenges()

    # ── Challenge bank ─────────────────────────────────────────────────────────

    def _build_challenges(self) -> List[ContextChallenge]:
        return [
            ContextChallenge(
                challenge_id="cw_001",
                full_prompt=(
                    "Analyze the performance of a multi-agent system where agents "
                    "compete for compute resources during a market volatility spike. "
                    "Identify the three primary failure modes and propose mitigations."
                ),
                sparse_prompt="multi-agent, compute, market spike → failures?",
                keywords=["failure modes", "compute", "market", "mitigation", "multi-agent"],
                compression_ratio=0.85,
                domain="analysis",
            ),
            ContextChallenge(
                challenge_id="cw_002",
                full_prompt=(
                    "Design a genome mutation strategy that increases novelty discovery "
                    "rate by 20% without sacrificing task coherence. Reference the "
                    "M_LEARNING_RATE and M_ADAPTABILITY molecules."
                ),
                sparse_prompt="genome mutation → +20% novelty, keep coherence",
                keywords=["mutation", "novelty", "coherence", "M_LEARNING_RATE", "M_ADAPTABILITY"],
                compression_ratio=0.82,
                domain="cognitive",
            ),
            ContextChallenge(
                challenge_id="cw_003",
                full_prompt=(
                    "Explain the causal relationship between desire-based reward "
                    "multipliers and long-run swarm diversity. Include the risk of "
                    "desire monoculture if dominant desires go uncontested."
                ),
                sparse_prompt="desire rewards → swarm diversity? monoculture risk?",
                keywords=["desire", "reward", "diversity", "monoculture", "swarm"],
                compression_ratio=0.88,
                domain="psychology",
            ),
            ContextChallenge(
                challenge_id="cw_004",
                full_prompt=(
                    "A CPA client needs a one-page summary of how local AI reduces "
                    "their data privacy risk compared to cloud LLMs. Focus on GDPR, "
                    "SOC2, and zero data egress."
                ),
                sparse_prompt="CPA: local AI vs cloud → privacy? GDPR, SOC2",
                keywords=["privacy", "GDPR", "SOC2", "local", "egress", "CPA"],
                compression_ratio=0.80,
                domain="market",
            ),
            ContextChallenge(
                challenge_id="cw_005",
                full_prompt=(
                    "Synthesize the key differences between PDAC and OODA loops "
                    "for autonomous AI agents. When should a system prefer one over "
                    "the other, and why?"
                ),
                sparse_prompt="PDAC vs OODA for AI agents → when to use each?",
                keywords=["PDAC", "OODA", "autonomous", "planning", "decision"],
                compression_ratio=0.79,
                domain="cognitive",
            ),
            ContextChallenge(
                challenge_id="cw_006",
                full_prompt=(
                    "Propose a three-phase onboarding pipeline for new agents entering "
                    "the SovereignNation swarm: genome initialization, curriculum "
                    "bootstrap, and loyalty calibration."
                ),
                sparse_prompt="new agent onboarding: genome → curriculum → loyalty",
                keywords=["onboarding", "genome", "curriculum", "loyalty", "bootstrap"],
                compression_ratio=0.83,
                domain="cognitive",
            ),
            ContextChallenge(
                challenge_id="cw_007",
                full_prompt=(
                    "Given a breakthrough detection threshold of novelty≥0.85 and "
                    "impact≥0.75, explain why rarity≥0.75 is the hardest criterion "
                    "to satisfy and how the frontier research lab addresses this."
                ),
                sparse_prompt="breakthrough: novelty+impact easy, rarity hard → frontier lab fix?",
                keywords=["breakthrough", "rarity", "novelty", "impact", "frontier", "threshold"],
                compression_ratio=0.87,
                domain="research",
            ),
        ]

    # ── Challenge execution ───────────────────────────────────────────────────

    def get_challenge(self, challenge_id: Optional[str] = None) -> ContextChallenge:
        if challenge_id:
            for c in self.challenges:
                if c.challenge_id == challenge_id:
                    return c
        return random.choice(self.challenges)

    def run_challenge(self, agent: Any, challenge_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Run a sparse-context challenge against an agent.

        Returns:
            {challenge_id, score, keyword_recovery, compression_ratio, domain, output}
        """
        challenge = self.get_challenge(challenge_id)

        task = {
            "task_id":  challenge.challenge_id,
            "type":     challenge.domain,
            "prompt":   challenge.sparse_prompt,
            "domain":   challenge.domain,
            "context":  {"compression_mode": True, "compression_ratio": challenge.compression_ratio},
        }
        result = agent.act(task)
        output_text = str(result.get("output", {}).get("content", "")).lower()

        # Score how many keywords the agent recovered
        recovered = [kw for kw in challenge.keywords if kw.lower() in output_text]
        keyword_recovery = len(recovered) / max(1, len(challenge.keywords))

        # Final score: keyword recovery weighted by compression ratio (harder = more credit)
        score = round(keyword_recovery * (0.5 + challenge.compression_ratio * 0.5), 4)

        return {
            "challenge_id":      challenge.challenge_id,
            "domain":            challenge.domain,
            "compression_ratio": challenge.compression_ratio,
            "keywords_expected": challenge.keywords,
            "keywords_recovered": recovered,
            "keyword_recovery":  round(keyword_recovery, 4),
            "score":             score,
            "agent_fitness":     result.get("fitness", 0.5),
            "output":            result.get("output", {}),
        }

    def run_suite(self, agent: Any, n: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full challenge suite (or n random challenges) against an agent.
        """
        subset = self.challenges if n is None else random.sample(
            self.challenges, min(n, len(self.challenges))
        )
        results = [self.run_challenge(agent, c.challenge_id) for c in subset]
        avg_score = sum(r["score"] for r in results) / max(1, len(results))
        avg_recovery = sum(r["keyword_recovery"] for r in results) / max(1, len(results))
        return {
            "agent_id":         getattr(agent, "agent_id", "unknown"),
            "challenges_run":   len(results),
            "avg_score":        round(avg_score, 4),
            "avg_recovery":     round(avg_recovery, 4),
            "results":          results,
        }

    def challenge_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "challenge_id":      c.challenge_id,
                "sparse_prompt":     c.sparse_prompt,
                "compression_ratio": c.compression_ratio,
                "domain":            c.domain,
                "keyword_count":     len(c.keywords),
            }
            for c in self.challenges
        ]
