"""MAP-Elites quality-diversity archive for GH05T3 SAGE cycle optimization.

Module-level API (use these directly):
    ask()           → list[dict]  — get next batch of parameter targets from emitter
    tell(obj, msr)  → None        — feed results back after a full ask() batch
    add(sol, obj, m)→ None        — direct single insertion (KAIROS path)
    archive_stats() → dict        — coverage / quality stats
    get_archive()   → GridArchive — raw pyribs archive
    get_scheduler() → Scheduler   — raw pyribs scheduler

Archive dimensions (4500 cells — 20×15×15):
    quality     0.0–1.0      (20 bins)
    latency_ms  0–30 000 ms  (15 bins)
    token_count 0–2 000 tok  (15 bins)

Solution vector (5D):
    [quality, token_budget, latency_ms, temperature, top_p]

Emitters:
    EvolutionStrategyEmitter (CMA-MA-ES, batch=4) — exploitation
    GaussianEmitter          (σ=0.2,     batch=4) — exploration
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    from ribs.archives import GridArchive
    from ribs.emitters import EvolutionStrategyEmitter, GaussianEmitter
    from ribs.schedulers import Scheduler
except ImportError as exc:
    raise ImportError(
        "Missing dependency. Install with: python -m pip install ribs numpy"
    ) from exc

LOG = logging.getLogger("ghost.evolution.map_elites")

_QUALITY_BINS = 20
_LATENCY_BINS = 15
_TOKEN_BINS   = 15
_BATCH_SIZE   = 8   # total per ask(); split evenly between emitters

_archive:              Optional[GridArchive] = None
_scheduler:            Optional[Scheduler]   = None
_last_ask_solutions:   Optional[np.ndarray]  = None


def _seed_solution() -> np.ndarray:
    return np.array([0.75, 400.0, 12000.0, 0.65, 0.90], dtype=np.float64)


def _build_archive() -> GridArchive:
    return GridArchive(
        solution_dim=5,
        dims=[_QUALITY_BINS, _LATENCY_BINS, _TOKEN_BINS],
        ranges=[
            (0.0, 1.0),
            (0.0, 30000.0),
            (0.0, 2000.0),
        ],
        seed=42,
    )


def _build_emitters(archive: GridArchive) -> list:
    half = max(1, _BATCH_SIZE // 2)
    return [
        EvolutionStrategyEmitter(
            archive,
            x0=_seed_solution(),
            sigma0=0.15,
            batch_size=half,
            seed=42,
        ),
        GaussianEmitter(
            archive,
            x0=_seed_solution(),
            sigma=0.2,
            batch_size=half,
            seed=43,
        ),
    ]


def get_archive() -> GridArchive:
    global _archive
    if _archive is None:
        _archive = _build_archive()
    return _archive


def get_scheduler() -> Scheduler:
    global _scheduler
    if _scheduler is None:
        archive = get_archive()
        _scheduler = Scheduler(archive, _build_emitters(archive))
    return _scheduler


def _target_from_solution(sol: np.ndarray, idx: int) -> dict:
    q_target    = float(np.clip(sol[0], 0.0, 1.0))
    t_budget    = int(np.clip(sol[1], 50, 2000))
    l_target    = float(np.clip(sol[2], 500, 30000))
    temperature = float(np.clip(sol[3], 0.1, 1.0))
    top_p       = float(np.clip(sol[4], 0.5, 1.0))
    return {
        "_cycle_num":     idx,
        "_batch_idx":     idx,
        "_solution":      [float(x) for x in sol],
        "quality_target": round(q_target, 3),
        "token_budget":   t_budget,
        "latency_target": round(l_target, 0),
        "temperature":    round(temperature, 2),
        "top_p":          round(top_p, 2),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask() -> list[dict]:
    """Return next batch of emitter-guided parameter targets.

    Each dict has: quality_target, token_budget, latency_target, temperature, top_p.
    You MUST call tell() with exactly len(ask()) results.
    """
    global _last_ask_solutions
    scheduler = get_scheduler()
    solutions = scheduler.ask()
    _last_ask_solutions = np.asarray(solutions, dtype=np.float64)
    return [_target_from_solution(sol, i) for i, sol in enumerate(_last_ask_solutions)]


def tell(objectives: list[float], measures: list[list[float]]) -> None:
    """Feed evaluation results back to the scheduler.

    Args:
        objectives: one float per solution (composite score)
        measures:   one [quality, latency_ms, token_count] list per solution
    """
    scheduler = get_scheduler()
    obj_arr = np.asarray(objectives, dtype=np.float64)
    msr_arr = np.asarray(measures,   dtype=np.float64)
    if msr_arr.ndim != 2 or msr_arr.shape[1] != 3:
        raise ValueError("measures must be shaped [[quality, latency_ms, token_count], ...]")
    if len(obj_arr) != len(msr_arr):
        raise ValueError("objectives and measures must have the same length")
    scheduler.tell(obj_arr, msr_arr)
    LOG.debug("[map-elites] tell() flushed %d results", len(obj_arr))


def add(solution: list[float], objective: float, measures: list[float]) -> None:
    """Direct single insertion — used by KAIROS for high-quality proposals."""
    archive = get_archive()
    archive.add_single(
        solution  = np.asarray(solution,  dtype=np.float64),
        objective = float(objective),
        measures  = np.asarray(measures,  dtype=np.float64),
    )


def archive_stats() -> dict:
    """Return coverage and quality statistics for the archive."""
    archive = get_archive()
    stats   = archive.stats
    obj_max  = stats.obj_max  if stats.obj_max  is not None else 0.0
    obj_mean = stats.obj_mean if stats.obj_mean is not None else 0.0
    return {
        "occupied_cells":  int(len(archive)),
        "total_cells":     int(archive.cells),
        "coverage_pct":    round(float(stats.coverage) * 100.0, 2),
        "best_objective":  round(float(obj_max),  3),
        "elite_count":     int(stats.num_elites),
        "objective_mean":  round(float(obj_mean), 3),
    }
