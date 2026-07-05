"""Real mutation-proposal engine: decides which genomes are
underperforming via ChronosLedger's real recorded scores, and for each,
asks its registered mutation operators (dna/mutation_operators.py) which
ones are applicable -- creating one real Mutation per applicable
operator, not a fabricated list.
"""
from __future__ import annotations

from typing import Any

from backend.oss.dna.mutation_operators import Mutation, MutationOperator


class OmniEvolutionEngine:
    def __init__(self, mutation_operators: list[MutationOperator], score_threshold: float = 0.0):
        self.mutation_operators = mutation_operators
        self.score_threshold = score_threshold

    def propose_mutations(self, genomes: list, stats: dict[str, dict[str, Any]]) -> list[Mutation]:
        """Only proposes mutations for genomes that HAVE a recorded score
        (genomes with no history are omitted by ChronosLedger.get_recent_stats,
        not padded with a fabricated score) and whose score is at or
        below score_threshold. A genome the ledger has no data on yet is
        left alone rather than guessed at."""
        proposals: list[Mutation] = []
        for genome in genomes:
            perf = stats.get(genome.id)
            if perf is None:
                continue
            if perf.get("score", float("-inf")) > self.score_threshold:
                continue
            for op in self.mutation_operators:
                if op.is_applicable(genome, perf):
                    proposals.append(op.create_mutation(genome, perf))
        return proposals
