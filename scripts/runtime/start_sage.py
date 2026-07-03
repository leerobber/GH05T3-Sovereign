"""
start_sage.py — Aethyro SAGE/KAIROS Engine Service
====================================================
FastAPI service that exposes the full SAGE + KAIROS + Honcho API.

Port: 8098 (not conflicting with anything in the stack)

Endpoints:
  GET  /                         → system info
  GET  /health                   → health check
  POST /sage/run                 → trigger manual SAGE cycle
  GET  /sage/status              → SAGE loop status
  GET  /sage/history             → cycle history
  GET  /kairos/plans             → recent KAIROS plans
  GET  /memory/stats             → memory system stats
  GET  /meta/history             → Meta-Agent rewrite history
  GET  /gitops/recent            → recent mutations
  GET  /ibac/log                 → IBAC events
  GET  /hardware                 → CPU/GPU/RAM stats
  GET  /iron_dome/verify         → chain integrity check
  WS   /ws                       → real-time event stream
  POST /ibac/request             → IBAC capability request
  POST /ibac/verify              → IBAC token verification
  GET  /ibac/policy              → capability policy
  GET  /ibac/stats               → IBAC statistics

Run:
    python start_sage.py
    # or via supervisor:
    uvicorn start_sage:app --host 0.0.0.0 --port 8098
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "sage.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
LOG = logging.getLogger("aethyro.sage_service")

# Add GH05T3 root to path so all imports work
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan — start background tasks
# ─────────────────────────────────────────────────────────────────────────────

_bg_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bg_task
    LOG.info("=== Aethyro SAGE Engine starting ===")

    # Start triage governor background loop
    async def _triage_bg():
        try:
            from memory.triage_governor import run_triage_loop
            await run_triage_loop()
        except Exception as e:
            LOG.error("Triage loop error: %s", e)

    # Start nightly SAGE scheduler
    async def _sage_bg():
        try:
            from kairos.sage_loop import SAGELoop
            sage = SAGELoop.instance()
            await sage.nightly_scheduler()
        except Exception as e:
            LOG.error("SAGE scheduler error: %s", e)

    # Properly create both as independent tasks
    t1 = asyncio.create_task(_triage_bg())
    t2 = asyncio.create_task(_sage_bg())
    _bg_task = (t1, t2)

    LOG.info("Background tasks launched: SAGE + Triage Governor")
    yield
    LOG.info("=== Aethyro SAGE Engine shutting down ===")
    t1.cancel()
    t2.cancel()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Aethyro SAGE Engine",
    description="SAGE Self-Improvement Engine + KAIROS Framework + Honcho Dashboard API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount routers ─────────────────────────────────────────────────────────────

try:
    from kairos.honcho_api import honcho_router
    app.include_router(honcho_router, prefix="", tags=["Honcho"])
    LOG.info("[OK] Honcho router mounted")
except Exception as e:
    LOG.warning("Honcho router failed: %s", e)

try:
    from sovereignnation.ibac_daemon import ibac_router
    app.include_router(ibac_router, prefix="/ibac", tags=["IBAC"])
    LOG.info("[OK] IBAC router mounted")
except Exception as e:
    LOG.warning("IBAC router failed: %s", e)


# ── Core endpoints ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service":     "Aethyro SAGE Engine",
        "version":     "1.0.0",
        "port":        8098,
        "components": [
            "KAIROS 6-Phase State Machine",
            "SAGE Self-Improvement Loop (10 cycles/night @ 3AM)",
            "GhostRecall 7-Layer Memory",
            "Iron Dome SHA-256 Hash Chain",
            "Triage-and-Bid Memory Governance",
            "3-Gate Verification Pipeline (Ethics/Sim/CLARA)",
            "Meta-Agent (rewrites rules every 3 cycles)",
            "Headless Git-Ops Mutation Pipeline",
            "IBAC Daemon (Intent-Based Access Control)",
            "Honcho Dashboard API + WebSocket",
        ],
        "ts": time.time(),
    }


@app.get("/health")
async def health():
    checks = {}

    # Iron Dome
    try:
        from memory.iron_dome import dome_verify
        ok, errs = dome_verify()
        checks["iron_dome"] = {"ok": ok, "errors": len(errs)}
    except Exception as e:
        checks["iron_dome"] = {"ok": False, "error": str(e)}

    # GhostRecall
    try:
        from memory.ghostrecall import ghost
        checks["ghostrecall"] = ghost.stats()
    except Exception as e:
        checks["ghostrecall"] = {"ok": False, "error": str(e)}

    # SAGE
    try:
        from kairos.sage_loop import SAGELoop
        sage = SAGELoop.instance()
        checks["sage"] = {"cycle": sage._cycle_number, "running": sage._running}
    except Exception as e:
        checks["sage"] = {"ok": False, "error": str(e)}

    # Triage
    try:
        from memory.triage_governor import governor
        checks["triage"] = governor.stats()
    except Exception as e:
        checks["triage"] = {"ok": False, "error": str(e)}

    all_ok = all(
        v.get("ok", True) is not False
        for v in checks.values()
        if isinstance(v, dict)
    )
    return {"status": "ok" if all_ok else "degraded", "checks": checks, "ts": time.time()}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    LOG.info("Starting Aethyro SAGE Engine on port 8098")
    uvicorn.run(
        "start_sage:app",
        host="0.0.0.0",
        port=8098,
        log_level="info",
        reload=False,
    )
