"""Real, pluggable genome selection strategies -- pick the best genome
given per-genome performance stats (see core/species_memory.py). Kept as
swappable strategy objects rather than a single hardcoded "pick highest
score" so a caller with a different real tradeoff (e.g. don't sacrifice
quality just for speed) doesn't have to fork SpeciesMemory to get it.
"""
from __future__ import annotations

from typing import Any, Protocol


class SelectionStrategy(Protocol):
    def select(self, scores: dict[str, dict[str, Any]]) -> str: ...


class HighestScoreSelection:
    """Picks the genome with the highest `score` field. Raises
    ValueError on an empty scores dict rather than returning an
    arbitrary/None key that would look like a valid selection to a
    caller that doesn't check for it."""

    def select(self, scores: dict[str, dict[str, Any]]) -> str:
        if not scores:
            raise ValueError("cannot select a genome from an empty scores dict")
        return max(scores.items(), key=lambda item: item[1].get("score", float("-inf")))[0]


class LowestLatencySelection:
    """Picks the lowest-latency genome among those meeting a minimum
    score threshold -- a real two-axis tradeoff (don't sacrifice quality
    just for speed), not a copy of HighestScoreSelection with a
    different field name."""

    def __init__(self, min_score: float = 0.0):
        self.min_score = min_score

    def select(self, scores: dict[str, dict[str, Any]]) -> str:
        eligible = {gid: s for gid, s in scores.items() if s.get("score", float("-inf")) >= self.min_score}
        if not eligible:
            raise ValueError(f"no genome meets min_score={self.min_score} among {scores!r}")
        return min(eligible.items(), key=lambda item: item[1].get("latency", float("inf")))[0]
