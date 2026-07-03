"""
Persistent Species Memory + Divergence Metrics

Tracks the "genome" of the entire population over evolution phases:
- Model versions used
- Trait statistics (mean, std, divergence)
- Fitness / eval history
- Number of active "species" (distinct high-variance clusters)

Used to measure true evolutionary progress beyond single agent scores.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any
import statistics

PERSIST_PATH = Path("data/species_memory.json")


@dataclass
class PhaseRecord:
    timestamp: float
    model_version: str
    mean_fitness: float
    std_fitness: float
    trait_means: Dict[str, float]
    trait_stds: Dict[str, float]   # divergence metric per trait
    population_size: int
    notes: str = ""


class SpeciesMemory:
    def __init__(self):
        self.history: List[PhaseRecord] = []
        self._load()

    def _load(self):
        if PERSIST_PATH.exists():
            try:
                raw = json.loads(PERSIST_PATH.read_text())
                self.history = [PhaseRecord(**r) for r in raw]
            except Exception:
                self.history = []

    def _save(self):
        PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self.history]
        PERSIST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_phase(
        self,
        model_version: str,
        mean_fitness: float,
        std_fitness: float,
        trait_means: Dict[str, float],
        trait_stds: Dict[str, float],
        population_size: int,
        notes: str = "",
    ):
        rec = PhaseRecord(
            timestamp=time.time(),
            model_version=model_version,
            mean_fitness=round(mean_fitness, 4),
            std_fitness=round(std_fitness, 4),
            trait_means={k: round(v, 3) for k, v in trait_means.items()},
            trait_stds={k: round(v, 3) for k, v in trait_stds.items()},
            population_size=population_size,
            notes=notes,
        )
        self.history.append(rec)
        if len(self.history) > 100:
            self.history = self.history[-80:]
        self._save()

    def get_divergence_metrics(self) -> Dict[str, Any]:
        if not self.history:
            return {}
        last = self.history[-1]
        prev = self.history[-2] if len(self.history) > 1 else last
        trait_div = {k: round(last.trait_stds.get(k, 0) - prev.trait_stds.get(k, 0), 3)
                     for k in last.trait_stds}
        return {
            "current_model": last.model_version,
            "mean_fitness": last.mean_fitness,
            "fitness_std": last.std_fitness,
            "avg_trait_divergence": round(statistics.mean(last.trait_stds.values()), 3),
            "trait_divergence_delta": trait_div,
            "phases": len(self.history),
        }

    def summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"status": "no history"}
        return {
            "phases_recorded": len(self.history),
            "latest": asdict(self.history[-1]),
            "divergence": self.get_divergence_metrics(),
        }


_species_mem: SpeciesMemory | None = None

def get_species_memory() -> SpeciesMemory:
    global _species_mem
    if _species_mem is None:
        _species_mem = SpeciesMemory()
    return _species_mem
