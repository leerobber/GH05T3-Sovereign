"""Tracks each genome's latest performance and selects the best genome
for a task, delegating the actual selection logic to a pluggable
SelectionStrategy (see dna/selection_strategies.py) rather than
hardcoding "pick highest score" directly here.
"""
from __future__ import annotations

from typing import Any

from backend.oss.dna.selection_strategies import HighestScoreSelection, SelectionStrategy


class SpeciesMemory:
    def __init__(self, strategy: SelectionStrategy | None = None) -> None:
        self._scores: dict[str, dict[str, Any]] = {}
        self.strategy = strategy or HighestScoreSelection()

    def update(self, genome_id: str, score: dict[str, Any]) -> None:
        self._scores[genome_id] = dict(score)

    def select_best_genome(self, task: dict[str, Any]) -> str:
        """`task` is accepted for interface compatibility with the
        broader genome-routing flow (see genomic_substrate.py, stage 2)
        -- this stage's strategies don't yet condition on task content,
        only on recorded scores. Task-aware selection (matching a
        genome's declared specialization to task type) is real future
        work, not something faked here by pretending `task` is used."""
        return self.strategy.select(self._scores)

    def scores_snapshot(self) -> dict[str, dict[str, Any]]:
        return dict(self._scores)
