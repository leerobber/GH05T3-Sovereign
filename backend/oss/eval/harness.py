"""
Evaluation Harness for GH05T3-Omni and Theorist agents.

Used to measure improvement after each meta-training cycle.
"""

from typing import List, Dict, Any
import statistics
import time
from backend.oss.mvs import get_mvs
from backend.oss.genomic_substrate import AgentHandle


class EvaluationHarness:
    def __init__(self):
        mvs = get_mvs()
        self.substrate = mvs["substrate"]
        self.mind = mvs["mind"]
        self.economy = mvs["economy"]

    def _get_theorists(self) -> List[Any]:
        """Get current theorist DNA records."""
        theorists = []
        for gid, rec in self.substrate.genomes.items():
            if "THEORIST" in getattr(rec, "role", "").upper():
                theorists.append(rec)
        return theorists

    def run_batch(
        self, tasks: List[Dict[str, Any]], role: str = "THEORIST_ELITE"
    ) -> Dict[str, Any]:
        """Run a batch of tasks against current agents and collect scores."""
        theorists = self._get_theorists()
        if not theorists:
            # fallback to any genomes
            theorists = list(self.substrate.genomes.values())[:5]

        all_scores = []
        results = []

        for task in tasks:
            for rec in theorists:
                try:
                    handle = AgentHandle(
                        genome_id=rec.genome_id,
                        role=role,
                        dna=rec.dna,
                        context={"eval": True},
                    )
                    result = handle.act(task)
                    # Use the world-style scoring if available, else heuristic
                    score = self._score_output(task, result.get("raw_output", ""))
                    all_scores.append(score)
                    results.append({
                        "genome_id": rec.genome_id,
                        "task": task.get("prompt", str(task))[:80],
                        "score": round(score, 4),
                        "model_version": result.get("model_version", "unknown"),
                    })
                except Exception as e:
                    all_scores.append(0.0)

        mean = statistics.mean(all_scores) if all_scores else 0.0
        std = statistics.pstdev(all_scores) if len(all_scores) > 1 else 0.0

        return {
            "mean_score": round(mean, 4),
            "std_score": round(std, 4),
            "n": len(all_scores),
            "n_agents": len(theorists),
            "sample_results": results[:5],
            "timestamp": time.time(),
        }

    def _score_output(self, task: Dict[str, Any], output: str) -> float:
        """Simple scoring for harness (same spirit as TheoryLab)."""
        text = (output or "").lower()
        score = 0.3
        if len(text) > 300:
            score += 0.2
        if any(k in text for k in ["therefore", "equation", "formal", "model", "alignment"]):
            score += 0.25
        if any(k in text for k in ["harm", "exploit", "override"]):
            score -= 0.2
        return max(0.0, min(1.0, score))
