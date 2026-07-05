"""Real, in-memory time-series performance ledger for genomes. In-memory
by design at this stage (fast-access log for a live evolution loop) --
can be backed by a persistent store later without changing this
interface.
"""
from __future__ import annotations

from typing import Any


class ChronosLedger:
    def __init__(self) -> None:
        self._history: dict[str, list[dict[str, Any]]] = {}

    def record_result(self, genome_id: str, score: dict[str, Any]) -> None:
        self._history.setdefault(genome_id, []).append(dict(score))

    def get_recent_stats(self, genome_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Returns each genome's most recent recorded result. Genomes
        with no history yet are omitted entirely (not padded with a
        fabricated zero score) -- "no data yet" and "scored zero" are
        different things a caller (e.g. deciding whether a genome is a
        mutation candidate) needs to be able to tell apart."""
        stats = {}
        for gid in genome_ids:
            hist = self._history.get(gid)
            if hist:
                stats[gid] = hist[-1]
        return stats

    def history_for(self, genome_id: str) -> list[dict[str, Any]]:
        return list(self._history.get(genome_id, []))
