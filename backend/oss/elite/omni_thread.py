"""
OmniThread Engine — shardable agent execution (Phase Elite).

Supports:
- shard_task: splits a task into N micro-tasks
- execute_parallel: run micro-tasks across agent handles
- merge_results: consensus or best_score strategies
- Deadlock-free: per-task timeouts, no global locks in hot path
"""
from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


MergeStrategy = Literal["consensus", "best_score", "union", "first_ok"]


@dataclass
class MicroTask:
    micro_id: str
    shard_index: int
    total_shards: int
    parent_task_id: str
    payload: Any
    domain: str = "generic"
    timeout_s: float = 15.0


@dataclass
class ShardResult:
    micro_id: str
    shard_index: int
    ok: bool
    result: Any = None
    score: float = 0.0
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class MergedResult:
    parent_task_id: str
    strategy: MergeStrategy
    ok: bool
    result: Any
    shard_count: int
    ok_shards: int
    avg_score: float
    total_latency_ms: float
    errors: List[str] = field(default_factory=list)


class OmniThreadEngine:
    """
    Shards tasks into micro-tasks, runs them in parallel, merges results.

    Design:
    - max_workers controls global thread pool ceiling
    - Each shard has its own timeout so a stuck shard can't block the merge
    - merge strategies: consensus (majority vote), best_score, union, first_ok
    """

    def __init__(self, max_workers: int = 16) -> None:
        self.max_workers = max_workers
        self._history: List[MergedResult] = []

    # ── Sharding ──────────────────────────────────────────────────────────────

    def shard_task(
        self,
        task_id: str,
        payload: Any,
        n_shards: int,
        domain: str = "generic",
        timeout_s: float = 15.0,
        shard_fn: Optional[Callable[[Any, int, int], Any]] = None,
    ) -> List[MicroTask]:
        """
        Split a task payload into n_shards micro-tasks.

        shard_fn(payload, shard_index, total_shards) → shard_payload
        Default: passes the full payload to every shard (fan-out).
        """
        n_shards = max(1, n_shards)
        if shard_fn is None:
            shard_fn = lambda p, i, n: p  # noqa: E731

        return [
            MicroTask(
                micro_id=f"{task_id}_s{i}_{uuid.uuid4().hex[:6]}",
                shard_index=i,
                total_shards=n_shards,
                parent_task_id=task_id,
                payload=shard_fn(payload, i, n_shards),
                domain=domain,
                timeout_s=timeout_s,
            )
            for i in range(n_shards)
        ]

    # ── Parallel execution ────────────────────────────────────────────────────

    def execute_parallel(
        self,
        micro_tasks: List[MicroTask],
        fn: Callable[[MicroTask], Any],
        score_fn: Optional[Callable[[Any], float]] = None,
    ) -> List[ShardResult]:
        """
        Execute micro_tasks in parallel via ThreadPoolExecutor.
        Each task is individually timeout-guarded.

        fn(micro_task) → result_value
        score_fn(result_value) → float score (default: 1.0 if ok else 0.0)
        """
        if not micro_tasks:
            return []

        results: List[ShardResult] = []
        workers = min(self.max_workers, len(micro_tasks))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(fn, mt): mt for mt in micro_tasks}
            for fut in as_completed(future_map):
                mt = future_map[fut]
                t0 = time.time()
                try:
                    raw = fut.result(timeout=mt.timeout_s)
                    ok = True
                    error = None
                    score = score_fn(raw) if score_fn else 1.0
                except FuturesTimeout:
                    raw = None
                    ok = False
                    error = f"timeout:{mt.timeout_s}s"
                    score = 0.0
                except Exception as exc:
                    raw = None
                    ok = False
                    error = str(exc)[:120]
                    score = 0.0
                latency = (time.time() - t0) * 1000
                results.append(ShardResult(
                    micro_id=mt.micro_id,
                    shard_index=mt.shard_index,
                    ok=ok,
                    result=raw,
                    score=score,
                    error=error,
                    latency_ms=round(latency, 2),
                ))

        results.sort(key=lambda r: r.shard_index)
        return results

    # ── Legacy shim ───────────────────────────────────────────────────────────

    def execute_sharded(
        self,
        tasks: List[Dict[str, Any]],
        fn: Callable[[Dict[str, Any]], Any],
    ) -> List[Dict[str, Any]]:
        """Legacy API: accepts plain dicts, returns plain dicts."""
        out = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(fn, t): t for t in tasks}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    out.append({"task": task, "result": fut.result(), "ok": True})
                except Exception as e:
                    out.append({"task": task, "error": str(e), "ok": False})
        return out

    # ── Merge ─────────────────────────────────────────────────────────────────

    def merge_results(
        self,
        parent_task_id: str,
        shard_results: List[ShardResult],
        strategy: MergeStrategy = "best_score",
    ) -> MergedResult:
        """
        Merge shard results into a single outcome.

        Strategies:
        - best_score : return the result with the highest score
        - consensus  : majority vote on result (works for hashable results)
        - union      : collect all ok results into a list
        - first_ok   : return the first successful result
        """
        ok_shards = [r for r in shard_results if r.ok]
        errors = [r.error for r in shard_results if r.error]
        avg_score = (
            sum(r.score for r in ok_shards) / len(ok_shards)
            if ok_shards else 0.0
        )
        total_lat = sum(r.latency_ms for r in shard_results)

        if not ok_shards:
            return MergedResult(
                parent_task_id=parent_task_id,
                strategy=strategy,
                ok=False,
                result=None,
                shard_count=len(shard_results),
                ok_shards=0,
                avg_score=0.0,
                total_latency_ms=total_lat,
                errors=errors,
            )

        if strategy == "best_score":
            best = max(ok_shards, key=lambda r: r.score)
            merged = best.result

        elif strategy == "consensus":
            from collections import Counter
            votes = Counter()
            for r in ok_shards:
                try:
                    votes[str(r.result)] += 1
                except Exception:
                    votes[repr(r.result)] += 1
            winner_key = votes.most_common(1)[0][0]
            # Return the first result that matches the winner
            merged = next(
                (r.result for r in ok_shards if str(r.result) == winner_key), ok_shards[0].result
            )

        elif strategy == "union":
            merged = [r.result for r in ok_shards]

        elif strategy == "first_ok":
            merged = ok_shards[0].result

        else:
            merged = ok_shards[0].result

        result = MergedResult(
            parent_task_id=parent_task_id,
            strategy=strategy,
            ok=True,
            result=merged,
            shard_count=len(shard_results),
            ok_shards=len(ok_shards),
            avg_score=round(avg_score, 4),
            total_latency_ms=round(total_lat, 2),
            errors=errors,
        )
        self._history.append(result)
        if len(self._history) > 200:
            self._history.pop(0)
        return result

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        if not self._history:
            return {"runs": 0}
        ok_runs = [r for r in self._history if r.ok]
        return {
            "runs": len(self._history),
            "ok_runs": len(ok_runs),
            "avg_score": round(sum(r.avg_score for r in ok_runs) / len(ok_runs), 4) if ok_runs else 0.0,
            "avg_shards": round(sum(r.shard_count for r in self._history) / len(self._history), 1),
        }
