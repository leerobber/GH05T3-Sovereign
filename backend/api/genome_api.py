"""Live HTTP surface for the genome subsystem (backend/oss/) -- register
genomes, evaluate them with a real forward pass, run a real evolution
cycle, and route tasks to the current best genome. Mounted into
gateway_v3.py at /oss/genome/*.

A single process-wide GenomicSubstrate instance backs every request.
Genomes/scores/history are in-memory only (ChronosLedger and GenomePlane
are both plain in-memory dicts at this stage) and are lost on restart --
an accepted, documented limitation, not a silent gap; persistence is
real future work if this needs to survive restarts.

Standalone-runnable for direct testing (`python -m backend.api.genome_api`);
also mounted on the main gateway (gateway_v3.py).
"""
from __future__ import annotations

import os
import sys
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

# gateway_v3.py is normally launched with backend/ itself as the sys.path
# root (its own internal imports are unprefixed: `from swarm...`,
# `from api.binary_training import ...`), but `backend.oss.*` needs the
# REPO ROOT on sys.path to resolve as a dotted package path. Ensure it's
# there regardless of how this module ends up imported (mounted into
# gateway_v3 with cwd=backend/, run standalone, or collected by pytest
# from the repo root, where this is already a no-op).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.oss.core.chronos_ledger import ChronosLedger
from backend.oss.core.genomic_substrate import GenomicSubstrate
from backend.oss.core.omni_evolution import OmniEvolutionEngine
from backend.oss.core.species_memory import SpeciesMemory
from backend.oss.dna.genome_plane import GenomePlane
from backend.oss.dna.mutation_operators import BinaryRatioJitterMutation, QuantModeMutation, StabilizerSwitchMutation
from backend.oss.swarm.swarm_runtime import SwarmRuntime

router = APIRouter(prefix="/oss/genome", tags=["genome"])

_substrate = GenomicSubstrate(
    genome_plane=GenomePlane(),
    evolution_engine=OmniEvolutionEngine(
        [BinaryRatioJitterMutation(), StabilizerSwitchMutation(), QuantModeMutation()]
    ),
    species_memory=SpeciesMemory(),
    chronos_ledger=ChronosLedger(),
    swarm_runtime=SwarmRuntime(),
)


class RegisterGenomeRequest(BaseModel):
    genome_id: str
    traits: dict[str, Any]


class EvaluateGenomeRequest(BaseModel):
    genome_id: str


def _genome_summary(genome, latest_score: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "genome_id": genome.id,
        "traits": genome.traits,
        "metadata": genome.metadata,
        "latest_score": latest_score,
    }


@router.post("/register")
def register_genome(req: RegisterGenomeRequest) -> dict:
    if _substrate.genome_plane.has_genome(req.genome_id):
        raise HTTPException(status_code=409, detail=f"genome {req.genome_id!r} already registered")
    genome = _substrate.genome_plane.add_genome(req.genome_id, req.traits)
    return _genome_summary(genome)


@router.get("/list")
def list_genomes() -> dict:
    scores = _substrate.species_memory.scores_snapshot()
    return {
        "genomes": [
            _genome_summary(g, scores.get(g.id)) for g in _substrate.genome_plane.list_genomes()
        ]
    }


@router.post("/evaluate")
def evaluate_genome(req: EvaluateGenomeRequest) -> dict:
    """Runs one real forward pass through the genome's real model and
    returns real loss/score/latency -- see swarm/swarm_runtime.py.
    Re-evaluating the same genome_id gives the same real score every
    time (deterministic per-genome seeding + caching in BMEBridge)."""
    if not _substrate.genome_plane.has_genome(req.genome_id):
        raise HTTPException(status_code=404, detail=f"genome {req.genome_id!r} not found")

    genome = _substrate.genome_plane.get_genome(req.genome_id)
    result = _substrate.swarm_runtime.evaluate_genome(genome.id, genome.traits)
    _substrate.chronos_ledger.record_result(genome.id, result)
    _substrate.species_memory.update(genome.id, result)
    return {"genome_id": genome.id, **result}


@router.post("/evolve")
def evolve() -> dict:
    """Runs one real evolution cycle: proposes mutations for genomes
    with a recorded score at or below the engine's threshold, applies
    and evaluates each with a real forward pass. Genomes with no
    recorded score yet (never /evaluate'd) are left untouched -- call
    /evaluate at least once per genome before expecting /evolve to act
    on it."""
    new_genomes = _substrate.run_evolution_cycle()
    scores = _substrate.species_memory.scores_snapshot()
    return {"new_genomes": [_genome_summary(g, scores.get(g.id)) for g in new_genomes]}


@router.get("/best")
def best_genome(task: str = "") -> dict:
    """Returns the current best REGISTERED-AND-EVALUATED genome per
    SpeciesMemory's selection strategy. 404s (not a fabricated default)
    if no genome has a recorded score yet."""
    try:
        genome = _substrate.route_task(task={"task": task} if task else {})
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    scores = _substrate.species_memory.scores_snapshot()
    return _genome_summary(genome, scores.get(genome.id))


def _standalone_app() -> FastAPI:
    app = FastAPI(title="GH05T3 genome subsystem")
    app.include_router(router)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_standalone_app(), host="0.0.0.0", port=8021)
