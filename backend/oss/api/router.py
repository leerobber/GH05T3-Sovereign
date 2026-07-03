"""OSS public API router."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..core.bme_bridge import get_bme_bridge
from ..loop import load_oss, run_cycle
from ..mvs import get_mvs
from ..omni_net.routing import RoutingEngine
from ..observability.metrics import (
    metrics_payload,
)

router = APIRouter(tags=["oss"])

_ROUTER = RoutingEngine()


class CycleRequest(BaseModel):
    cycles: int = Field(default=1, ge=1, le=1000)
    dry_run: bool = True


class BMECycleRequest(BaseModel):
    cycles: int = Field(default=1, ge=1, le=1000)
    target_universe: Optional[int] = Field(default=None, ge=0, le=7)
    allow_migration: bool = True
    allow_promotion: bool = True
    allow_breakthrough: bool = True
    push_sovereign_core: bool = False


class RouteRequest(BaseModel):
    prompt: str
    traits: Dict[str, float] = Field(default_factory=dict)
    history: List[str] = Field(default_factory=list)
    agent_id: str = "API"


def _substrate_snapshot() -> Dict[str, Any]:
    mvs = get_mvs()
    substrate = mvs.get("substrate")
    if substrate is None:
        return {"total_genomes": 0, "roles": [], "avg_fitness": 0.0}
    try:
        return substrate.stats()
    except Exception:
        return {"total_genomes": len(getattr(substrate, "genomes", {})), "roles": [], "avg_fitness": 0.0}


def _loop_snapshot() -> Dict[str, Any]:
    state = load_oss()
    rewards = state.get("rewards", {})
    return {
        "species": state.get("species_state", "S0_perceive"),
        "aggregate": rewards.get("aggregate", 0.0),
        "omni_mind": rewards.get("omni_mind", 0.0),
    }


@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "layer": "oss",
        "version": "1.0.0",
    }


@router.get("/metrics")
def metrics() -> Response:
    body, media_type = metrics_payload()
    return Response(content=body, media_type=media_type)


@router.get("/mvs/status")
def mvs_status() -> Dict[str, Any]:
    mvs = get_mvs()
    substrate = _substrate_snapshot()
    loop_state = _loop_snapshot()
    available = not bool(mvs.get("partial") or mvs.get("error"))
    return {
        "available": available,
        "genomes": {
            "total_genomes": substrate.get("total_genomes", 0),
            "roles": substrate.get("roles", []),
        },
        "loop_state": loop_state,
        "source": "backend.oss.mvs + oss_ecosystem.json",
    }


@router.post("/mvs/cycle")
def mvs_cycle(payload: CycleRequest) -> Dict[str, Any]:
    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    for tick in range(payload.cycles):
        cl = run_cycle(tick, dry_run=payload.dry_run, verbose=False)
        results.append(
            {
                "tick": cl.tick,
                "global": cl.global_state,
                "agg": cl.rewards.get("aggregate", 0.0),
            }
        )
    return {
        "ran": payload.cycles,
        "dry_run": payload.dry_run,
        "results": results,
        "duration_sec": round(time.perf_counter() - started, 4),
    }


@router.post("/omni/route")
def omni_route(payload: RouteRequest) -> Dict[str, Any]:
    route = _ROUTER.route(
        payload.prompt,
        traits=payload.traits or None,
        history=payload.history or None,
    )
    return {
        "ok": True,
        "route": route,
        "adapter_bucket": route,
        "agent_id": payload.agent_id,
        "source": "oss.omni_net.routing",
    }


@router.get("/registry/status")
def registry_status() -> Dict[str, Any]:
    stats = _substrate_snapshot()
    return {
        "genome_count": stats.get("total_genomes", 0),
        "roles": stats.get("roles", []),
        "avg_fitness": stats.get("avg_fitness", 0.0),
        "source": "backend.oss.genomic_substrate",
    }


@router.get("/economy/unified")
def economy_unified() -> Dict[str, Any]:
    mvs = get_mvs()
    economy = mvs.get("economy")
    balance_count = len(getattr(economy, "balances", {}) or {})
    return {
        "currency": "NeuroCoin",
        "local_online": True,
        "sovereign_core_online": False,
        "balance_count": balance_count,
        "source": "backend.oss.omni_economy",
    }


@router.get("/bme/status")
def bme_status() -> Dict[str, Any]:
    bme = get_bme_bridge()
    return bme.collect_stats().as_dict()


@router.get("/bme/flagships")
def bme_flagships() -> Dict[str, Any]:
    bme = get_bme_bridge()
    return bme.flagship_profiles()


@router.get("/bme/flagships/{species_name}")
def bme_flagship(species_name: str) -> Dict[str, Any]:
    bme = get_bme_bridge()
    return bme.flagship_profile(species_name)


@router.post("/bme/cycle")
async def bme_cycle(payload: BMECycleRequest) -> Dict[str, Any]:
    bme = get_bme_bridge()
    started = time.perf_counter()
    last_result: Dict[str, Any] = {}
    for _ in range(payload.cycles):
        last_result = bme.universe_pass(
            target_universe=payload.target_universe,
            allow_migration=payload.allow_migration and payload.target_universe is None,
            allow_promotion=payload.allow_promotion,
            allow_breakthrough=payload.allow_breakthrough,
        )

    stats = bme.collect_stats().as_dict()
    sovereign_pushed = False
    if payload.push_sovereign_core:
        sovereign_pushed = await bme.push_to_sovereign_core(
            universe_counts=last_result.get("universe_counts", {}),
            migrations=last_result.get("migrations", 0),
            promotions=last_result.get("promotions", 0),
            tick=last_result.get("agents_processed", 0),
        )

    return {
        "cycle_completed": True,
        "ran": payload.cycles,
        "target_universe": payload.target_universe,
        "allow_migration": payload.allow_migration,
        "allow_promotion": payload.allow_promotion,
        "allow_breakthrough": payload.allow_breakthrough,
        "sovereign_core_pushed": sovereign_pushed,
        **last_result,
        **stats,
    }
