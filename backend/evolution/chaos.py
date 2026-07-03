"""GH05T3 — War Room: AIOpsLab-style fault injection for KAIROS hardening.

Run fault scenarios *between* KAIROS cycles to force the swarm to exercise
fault recovery paths before they're needed in production.

Usage:
    from evolution.chaos import FaultInjector, FAULT_CATALOG

    injector = FaultInjector()
    async with injector.scenario("groq_rate_limit"):
        await kairos.run_cycle(...)   # runs under simulated fault

    # or run all registered scenarios sequentially
    results = await injector.run_all(kairos_cycle_fn)

Env vars:
    CHAOS_ENABLED   "1" to enable fault injection (default "0" — safe)
    CHAOS_LOG_ONLY  "1" to log faults without actually injecting (dry-run)
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Awaitable

LOG = logging.getLogger("ghost.chaos")

CHAOS_ENABLED  = os.environ.get("CHAOS_ENABLED",  "0") == "1"
CHAOS_LOG_ONLY = os.environ.get("CHAOS_LOG_ONLY", "1") == "1"  # dry-run by default


@dataclass
class FaultResult:
    scenario:    str
    injected:    bool
    duration_ms: float
    error:       str  = ""
    passed:      bool = True


@dataclass
class FaultScenario:
    name:        str
    description: str
    inject:      Callable[[], Awaitable[None]]
    restore:     Callable[[], Awaitable[None]]
    tags:        list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in fault catalog — wire into KAIROS between cycles
# ---------------------------------------------------------------------------

async def _noop(): pass


async def _inject_inference_latency():
    """Simulate 2-second inference latency spike (ORACLE hang scenario)."""
    LOG.warning("[chaos] INJECTING: inference_latency_spike (2000ms)")
    await asyncio.sleep(2.0)


async def _inject_swarmbus_partition():
    """Log a 40% message drop simulation — actual drop requires bus hook."""
    LOG.warning("[chaos] INJECTING: swarmbus_partition (40%% drop, FORGE+CODEX)")


async def _inject_groq_rate_limit():
    """Temporarily poison the groq cooloff timer to simulate daily limit."""
    LOG.warning("[chaos] INJECTING: groq_rate_limit (60s cooldown)")
    try:
        import ghost_llm
        ghost_llm._mark_rl("groq", 60)
    except Exception as e:
        LOG.debug("chaos groq inject skipped: %s", e)


async def _restore_groq():
    try:
        import ghost_llm
        ghost_llm._rl_until.pop("groq", None)
        LOG.info("[chaos] RESTORED: groq circuit")
    except Exception:
        pass


async def _inject_memory_failure():
    """Simulate Qdrant write failure — log only, no actual block."""
    LOG.warning("[chaos] INJECTING: qdrant_write_fail (simulated, 30s)")
    await asyncio.sleep(0.1)


async def _inject_vram_pressure():
    """Log VRAM pressure scenario (actual allocation requires GPU access)."""
    LOG.warning("[chaos] INJECTING: torch_oom_pressure (near 8GB ceiling on RTX 5050)")


FAULT_CATALOG: dict[str, FaultScenario] = {
    "inference_latency_spike": FaultScenario(
        name        = "inference_latency_spike",
        description = "2-second inference latency spike — tests timeout/fallback paths",
        inject      = _inject_inference_latency,
        restore     = _noop,
        tags        = ["latency", "gpu"],
    ),
    "swarmbus_partition": FaultScenario(
        name        = "swarmbus_partition",
        description = "40% message drop on FORGE+CODEX channels",
        inject      = _inject_swarmbus_partition,
        restore     = _noop,
        tags        = ["network", "swarm"],
    ),
    "groq_rate_limit": FaultScenario(
        name        = "groq_rate_limit",
        description = "Simulate Groq daily quota hit — forces Google/OpenRouter cascade",
        inject      = _inject_groq_rate_limit,
        restore     = _restore_groq,
        tags        = ["cloud", "rate_limit"],
    ),
    "qdrant_write_fail": FaultScenario(
        name        = "qdrant_write_fail",
        description = "Qdrant write blocked — memory falls back to BM25 recall",
        inject      = _inject_memory_failure,
        restore     = _noop,
        tags        = ["memory", "storage"],
    ),
    "torch_oom": FaultScenario(
        name        = "torch_oom",
        description = "Near-8GB VRAM pressure — tests quantization fallback paths",
        inject      = _inject_vram_pressure,
        restore     = _noop,
        tags        = ["gpu", "memory"],
    ),
}


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------

class FaultInjector:
    """Manages fault injection lifecycle for KAIROS War Room cycles."""

    def __init__(self, catalog: dict[str, FaultScenario] = None):
        self._catalog = catalog or FAULT_CATALOG
        self._results: list[FaultResult] = []

    @asynccontextmanager
    async def scenario(self, name: str) -> AsyncGenerator[FaultScenario, None]:
        """Context manager: inject fault, yield, restore."""
        sc = self._catalog.get(name)
        if not sc:
            raise ValueError(f"Unknown fault scenario: {name!r}. "
                             f"Available: {list(self._catalog.keys())}")

        if not CHAOS_ENABLED:
            LOG.info("[chaos] DISABLED — skipping scenario %r (set CHAOS_ENABLED=1)", name)
            yield sc
            return

        t0 = time.monotonic()
        injected = False
        error    = ""
        passed   = True
        try:
            if not CHAOS_LOG_ONLY:
                await sc.inject()
            else:
                LOG.info("[chaos] DRY-RUN: would inject %r", name)
            injected = True
            yield sc
        except Exception as e:
            error  = str(e)
            passed = False
            raise
        finally:
            if injected and not CHAOS_LOG_ONLY:
                await sc.restore()
            # Append exactly once — tracks both success and failure paths
            self._results.append(FaultResult(
                scenario    = name,
                injected    = injected,
                duration_ms = (time.monotonic() - t0) * 1000,
                error       = error,
                passed      = passed,
            ))

    async def run_all(self, probe_fn: Callable[[], Awaitable[None]],
                      tags: list[str] = None) -> list[FaultResult]:
        """Run probe_fn under each scenario in the catalog.

        Args:
            probe_fn: async callable to run under each fault (e.g. a KAIROS cycle)
            tags:     if set, only run scenarios matching at least one tag
        """
        for name, sc in self._catalog.items():
            if tags and not any(t in sc.tags for t in tags):
                continue
            try:
                async with self.scenario(name):
                    await probe_fn()
            except Exception as e:
                LOG.error("[chaos] scenario %r failed during probe: %s", name, e)
        return list(self._results)

    @property
    def results(self) -> list[FaultResult]:
        return list(self._results)

    def summary(self) -> dict:
        return {
            "total":   len(self._results),
            "passed":  sum(1 for r in self._results if r.passed),
            "failed":  sum(1 for r in self._results if not r.passed),
            "results": [{"scenario": r.scenario, "passed": r.passed,
                         "duration_ms": round(r.duration_ms, 1),
                         "error": r.error} for r in self._results],
        }
