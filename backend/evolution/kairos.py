"""GH05T3 — KAIROS evolutionary cycle engine."""
from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

KAIROS_LOG       = Path("evolution/kairos_log.jsonl")
ELITE_THRESHOLD  = float(os.environ.get("SAGE_ELITE_THRESHOLD", "0.90"))


@dataclass
class KAIROSCycle:
    id:         int
    proposal:   str
    verdict:    str
    score:      float
    timestamp:  float = field(default_factory=time.time)
    is_elite:   bool  = False

    def to_dict(self) -> dict:
        return asdict(self)


class KAIROS:
    """Records evolutionary cycles, maintains elite archive, persists to JSONL."""

    def __init__(self, elite_threshold: float = ELITE_THRESHOLD):
        self.elite_threshold = elite_threshold
        self._cycles: list[KAIROSCycle] = []
        self._elite:  list[KAIROSCycle] = []
        KAIROS_LOG.parent.mkdir(parents=True, exist_ok=True)

    def record_cycle(self, proposal: str, verdict: str, score: float) -> KAIROSCycle:
        cycle = KAIROSCycle(
            id=len(self._cycles) + 1,
            proposal=proposal,
            verdict=verdict,
            score=score,
            is_elite=score >= self.elite_threshold,
        )
        self._cycles.append(cycle)
        if cycle.is_elite:
            self._elite.append(cycle)

        with open(KAIROS_LOG, "a") as f:
            f.write(json.dumps(cycle.to_dict()) + "\n")

        # W&B — best-effort
        try:
            from integrations.wandb_logger import log_kairos_cycle
            log_kairos_cycle(
                cycle_id=cycle.id,
                score=cycle.score,
                is_elite=cycle.is_elite,
                total_cycles=len(self._cycles),
                elite_cycles=len(self._elite),
            )
        except Exception:
            pass

        # Notify on elite — best-effort
        if cycle.is_elite:
            try:
                import asyncio
                from integrations.notifier import notify_elite_cycle
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(notify_elite_cycle(
                        cycle.id, cycle.score, cycle.proposal))
                except RuntimeError:
                    pass  # no event loop — skip notification
            except Exception:
                pass

        return cycle

    @property
    def elite_archive(self) -> list[KAIROSCycle]:
        return list(self._elite)

    @property
    def stats(self) -> dict:
        scores = [c.score for c in self._cycles]
        return {
            "total_cycles":     len(self._cycles),
            "elite_cycles":     len(self._elite),
            "elite_threshold":  self.elite_threshold,
            "avg_score":        sum(scores) / len(scores) if scores else 0.0,
        }

