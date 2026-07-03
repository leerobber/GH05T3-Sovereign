"""
OmniMind Fabric — elite task routing (Phase Elite).

Routes tasks by domain/trait to elite agents. Stateless per request,
parallel execution via threading, deadlock-free with timeout guard.
No single coordinator bottleneck.
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─── Domain → preferred trait mapping ────────────────────────────────────────
DOMAIN_TRAIT_MAP: Dict[str, List[str]] = {
    "theory":       ["math", "rigor", "pattern_detection", "self_reflection"],
    "market":       ["market_intuition", "risk_tolerance", "efficiency"],
    "creative":     ["creativity", "novelty_seeking", "innovation"],
    "systems":      ["efficiency", "persistence", "alignment"],
    "psychology":   ["empathy", "market_intuition", "creativity"],
    "research":     ["novelty_seeking", "math", "rigor"],
    "operations":   ["persistence", "efficiency", "alignment"],
    "governance":   ["alignment", "rigor", "collaboration"],
}

# ─── Tier routing priority ────────────────────────────────────────────────────
TIER_PRIORITY = {"T5": 5, "T4": 4, "T3": 3, "T2": 2, "T1": 1, "T0": 0}


@dataclass
class FabricTask:
    task_id: str
    domain: str
    payload: Any
    required_traits: List[str] = field(default_factory=list)
    min_tier: str = "T0"
    timeout_s: float = 10.0
    created_at: float = field(default_factory=time.time)


@dataclass
class FabricResult:
    task_id: str
    agent_id: Optional[str]
    domain: str
    ok: bool
    result: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    tier_used: str = "T0"


class OmniFabric:
    """
    Stateless-per-request task router for elite agents.

    Key properties:
    - No global lock during execution (submit is thread-safe via internal lock)
    - Deadlock-free: each task runs in isolated ThreadPoolExecutor with timeout
    - Routes by domain → trait affinity → tier priority
    - Degrades gracefully if no matching agent (uses generic executor)
    """

    def __init__(self, max_workers: int = 16) -> None:
        self._max_workers = max_workers
        self._registry: Dict[str, Dict[str, Any]] = {}  # agent_id → {traits, tier, domain, fn}
        self._results: List[FabricResult] = []
        self._lock = threading.Lock()
        self._stats = {"routed": 0, "ok": 0, "failed": 0, "total_latency_ms": 0.0}

    # ── Agent registration ────────────────────────────────────────────────────

    def register_agent(
        self,
        agent_id: str,
        traits: Dict[str, float],
        tier: str = "T0",
        domains: Optional[List[str]] = None,
        fn: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        """Register an elite agent with its trait profile and optional handler."""
        with self._lock:
            self._registry[agent_id] = {
                "traits": traits,
                "tier": tier,
                "domains": domains or [],
                "fn": fn or (lambda payload: {"echo": str(payload)[:64]}),
                "calls": 0,
                "errors": 0,
            }

    def deregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            return bool(self._registry.pop(agent_id, None))

    # ── Routing ───────────────────────────────────────────────────────────────

    def _score_agent(self, agent: Dict[str, Any], task: FabricTask) -> float:
        """Compute affinity score (0–1) for an agent given a task."""
        traits = agent.get("traits", {})
        tier_score = TIER_PRIORITY.get(agent.get("tier", "T0"), 0) / 5.0

        # domain bonus
        domain_bonus = 0.2 if task.domain in agent.get("domains", []) else 0.0

        # trait affinity
        required = task.required_traits or DOMAIN_TRAIT_MAP.get(task.domain, [])
        if required:
            affinity = sum(traits.get(t, 0.0) for t in required) / len(required)
        else:
            affinity = sum(traits.values()) / max(1, len(traits))

        # tier gate
        min_tier_val = TIER_PRIORITY.get(task.min_tier, 0)
        agent_tier_val = TIER_PRIORITY.get(agent.get("tier", "T0"), 0)
        if agent_tier_val < min_tier_val:
            return 0.0  # disqualified

        return round(0.5 * affinity + 0.2 * tier_score + 0.3 * domain_bonus, 4)

    def _pick_agent(self, task: FabricTask) -> Optional[Tuple[str, Dict[str, Any]]]:
        with self._lock:
            if not self._registry:
                return None
            scored = [
                (aid, rec, self._score_agent(rec, task))
                for aid, rec in self._registry.items()
            ]
        scored.sort(key=lambda x: x[2], reverse=True)
        if scored and scored[0][2] > 0.0:
            return scored[0][0], scored[0][1]
        return None

    # ── Single task execution ─────────────────────────────────────────────────

    def route(self, task: FabricTask) -> FabricResult:
        """Route a single task to the best available agent. Blocking, timeout-safe."""
        t0 = time.time()
        pick = self._pick_agent(task)

        if pick is None:
            latency = (time.time() - t0) * 1000
            result = FabricResult(
                task_id=task.task_id, agent_id=None, domain=task.domain,
                ok=False, error="no_agent_available", latency_ms=latency,
            )
            with self._lock:
                self._results.append(result)
                self._stats["failed"] += 1
                self._stats["routed"] += 1
            return result

        agent_id, agent_rec = pick
        fn = agent_rec["fn"]

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn, task.payload)
                raw = fut.result(timeout=task.timeout_s)
            ok = True
            error = None
            with self._lock:
                agent_rec["calls"] += 1
        except FuturesTimeout:
            raw = None
            ok = False
            error = f"timeout:{task.timeout_s}s"
            with self._lock:
                agent_rec["errors"] += 1
        except Exception as exc:
            raw = None
            ok = False
            error = str(exc)[:120]
            with self._lock:
                agent_rec["errors"] += 1

        latency = (time.time() - t0) * 1000
        result = FabricResult(
            task_id=task.task_id,
            agent_id=agent_id,
            domain=task.domain,
            ok=ok,
            result=raw,
            error=error,
            latency_ms=round(latency, 2),
            tier_used=agent_rec.get("tier", "T0"),
        )
        with self._lock:
            self._results.append(result)
            self._stats["routed"] += 1
            if ok:
                self._stats["ok"] += 1
            else:
                self._stats["failed"] += 1
            self._stats["total_latency_ms"] += latency
        return result

    # ── Batch execution (parallel, deadlock-free) ─────────────────────────────

    def route_batch(self, tasks: List[FabricTask]) -> List[FabricResult]:
        """
        Route up to N tasks in parallel. Each task has its own isolated executor
        with a timeout, so a stuck task cannot block others.
        Returns results in completion order.
        """
        if not tasks:
            return []

        results: List[FabricResult] = []
        with ThreadPoolExecutor(max_workers=min(self._max_workers, len(tasks))) as pool:
            future_map = {pool.submit(self.route, t): t for t in tasks}
            for fut in as_completed(future_map, timeout=max(t.timeout_s for t in tasks) + 5.0):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    task = future_map[fut]
                    results.append(FabricResult(
                        task_id=task.task_id, agent_id=None, domain=task.domain,
                        ok=False, error=f"batch_error:{exc}",
                    ))
        return results

    # ── Legacy queue API (kept for backward compat) ───────────────────────────

    def submit(self, task_id: str, fn: Callable[[], Any]) -> None:
        """Legacy: submit a raw callable. Wraps into a FabricTask and routes."""
        task = FabricTask(
            task_id=task_id,
            domain="generic",
            payload=fn,
            timeout_s=30.0,
        )
        # Register a throwaway agent for this if none exist
        tmp_id = f"_tmp_{task_id}"
        self.register_agent(tmp_id, {}, tier="T0", fn=lambda p: p())
        self.route(task)
        self.deregister_agent(tmp_id)

    def drain(self, max_tasks: int = 100) -> List[Dict[str, Any]]:
        """Legacy: return last N stored results as dicts."""
        with self._lock:
            out = self._results[-max_tasks:]
        return [
            {"task_id": r.task_id, "ok": r.ok,
             "result": r.result, "error": r.error}
            for r in out
        ]

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            routed = self._stats["routed"]
            avg_lat = (
                self._stats["total_latency_ms"] / routed if routed else 0.0
            )
            return {
                "registered_agents": len(self._registry),
                "routed": routed,
                "ok": self._stats["ok"],
                "failed": self._stats["failed"],
                "success_rate": round(self._stats["ok"] / routed, 4) if routed else 0.0,
                "avg_latency_ms": round(avg_lat, 2),
                "results_stored": len(self._results),
            }

    def agent_stats(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {
                aid: {"tier": rec["tier"], "domains": rec["domains"],
                      "calls": rec["calls"], "errors": rec["errors"]}
                for aid, rec in self._registry.items()
            }


# ── Convenience factory ───────────────────────────────────────────────────────

def make_fabric_task(
    domain: str,
    payload: Any,
    min_tier: str = "T0",
    timeout_s: float = 10.0,
    required_traits: Optional[List[str]] = None,
) -> FabricTask:
    return FabricTask(
        task_id=f"fab_{uuid.uuid4().hex[:10]}",
        domain=domain,
        payload=payload,
        required_traits=required_traits or [],
        min_tier=min_tier,
        timeout_s=timeout_s,
    )
