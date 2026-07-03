"""
Context Maturity Self-Score Gate — agents score their own context readiness (1-8)
before acting. Score < threshold triggers enrichment rather than hallucination.
Wraps the existing score_context_maturity() / context_efficiency_score() from
context_genome.py so no new genomic machinery is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.oss.genomic.context_genome import (
    score_context_maturity,
    context_efficiency_score,
)

LOG = logging.getLogger("ghost.context_maturity")

_DEFAULT_THRESHOLD = 5          # below this → request enrichment, not hallucination
_DOMAIN_MINIMUMS: Dict[str, int] = {
    "llm_architecture":  6,
    "slm_optimization":  6,
    "mlm_training":      5,
    "data_science":      4,
    "research":          5,
    "cognitive":         4,
    "psychology":        3,
    "market":            4,
}


@dataclass
class ContextMaturityScore:
    score: int              # 1-8
    efficiency: float       # 0-1
    needs_enrichment: bool
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score":            self.score,
            "efficiency":       round(self.efficiency, 4),
            "needs_enrichment": self.needs_enrichment,
            "recommendations":  self.recommendations,
        }


class ContextMaturityScorer:
    """
    Scores an agent's context maturity for a given task and determines
    whether context enrichment is required before acting.

    Usage:
        scorer = ContextMaturityScorer()
        result = scorer.score(agent.genome, task, context_packet)
        if result.needs_enrichment:
            # request enrichment via ContextBroker instead of acting
    """

    def __init__(self, default_threshold: int = _DEFAULT_THRESHOLD):
        self.default_threshold = default_threshold

    def score(
        self,
        genome: "Genome",
        task: Dict[str, Any],
        context_packet: Optional[Any] = None,
    ) -> ContextMaturityScore:
        """
        Score context maturity (1-8) for this genome + task pair.
        context_packet: optional ContextPacket from ContextBroker — its hit_rate
                        can push the score up by 1 level.
        """
        maturity   = score_context_maturity(genome)
        efficiency = context_efficiency_score(genome)

        # Boost: a high-quality assembled context_packet can lift score by 1
        if context_packet is not None:
            hit_rate = getattr(context_packet, "hit_rate", 0.0)
            if hit_rate >= 0.7 and maturity < 8:
                maturity = min(8, maturity + 1)

        domain    = task.get("domain", task.get("type", "general"))
        threshold = _DOMAIN_MINIMUMS.get(domain, self.default_threshold)
        needs     = maturity < threshold

        recs = self._recommend(maturity, efficiency, domain, context_packet)
        return ContextMaturityScore(
            score=maturity, efficiency=efficiency,
            needs_enrichment=needs, recommendations=recs,
        )

    def should_enrich(self, genome: "Genome", task: Optional[Dict[str, Any]] = None) -> bool:
        """Quick boolean gate — True means agent should enrich context first."""
        maturity  = score_context_maturity(genome)
        domain    = (task or {}).get("domain", "general")
        threshold = _DOMAIN_MINIMUMS.get(domain, self.default_threshold)
        return maturity < threshold

    # ── Internal ─────────────────────────────────────────────────────────────

    def _recommend(
        self,
        score: int,
        efficiency: float,
        domain: str,
        context_packet: Optional[Any],
    ) -> List[str]:
        recs: List[str] = []
        if score <= 2:
            recs.append("Context is SURFACE-level — evolve M_CONTEXT_DEPTH molecule")
        elif score <= 4:
            recs.append("Context is SHALLOW — boost M_CONTEXT_SYNTHESIS via ContextBroker")
        elif score <= 6:
            recs.append("Context is MODERATE — enable M_CONTEXT_ANTICIPATION for higher levels")
        if efficiency < 0.35:
            recs.append("Low efficiency — increase M_CONTEXT_COMPRESSION and M_CONTEXT_EFFICIENCY")
        if context_packet is not None:
            hit_rate = getattr(context_packet, "hit_rate", None)
            if hit_rate is not None and hit_rate < 0.5:
                recs.append(f"ContextBroker hit_rate={hit_rate:.2f} is low — widen query terms")
        min_level = _DOMAIN_MINIMUMS.get(domain, self.default_threshold)
        if score < min_level:
            recs.append(
                f"Domain {domain!r} requires Level {min_level}+ context "
                f"(current {score}) — query knowledge graph before acting"
            )
        return recs


# ── Singleton ─────────────────────────────────────────────────────────────────

_scorer: Optional[ContextMaturityScorer] = None

def get_context_maturity_scorer() -> ContextMaturityScorer:
    global _scorer
    if _scorer is None:
        _scorer = ContextMaturityScorer()
    return _scorer
