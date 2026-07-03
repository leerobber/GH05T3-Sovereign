"""
OSS metrics helpers.

Lightweight, dependency-free counters plus a Prometheus-compatible text payload
for the OSS contract tests and the gateway /oss/metrics endpoint.
"""

from __future__ import annotations

import time
from collections import Counter
from contextlib import contextmanager
from typing import Dict, Iterable, Tuple

_COUNTERS = Counter()
_LAST_CYCLE_DURATION = 0.0
_CYCLE_COUNT = 0
_CYCLE_DURATION_SUM = 0.0
_CYCLE_DRY_RUN_COUNT = 0


def record_cycle(duration_sec: float, dry_run: bool = False, outcome: str = "ok") -> None:
    """Record a single OSS cycle observation."""
    global _LAST_CYCLE_DURATION, _CYCLE_COUNT, _CYCLE_DURATION_SUM, _CYCLE_DRY_RUN_COUNT
    _LAST_CYCLE_DURATION = float(duration_sec)
    _CYCLE_COUNT += 1
    _CYCLE_DURATION_SUM += float(duration_sec)
    if dry_run:
        _CYCLE_DRY_RUN_COUNT += 1
    _COUNTERS[f"gh05t3_mvs_cycle_outcome_{outcome}_total"] += 1


def record_marketplace_success(action: str) -> None:
    _COUNTERS[f"gh05t3_marketplace_successes_total{{action=\"{action}\"}}"] += 1


def record_marketplace_failure(action: str) -> None:
    _COUNTERS[f"gh05t3_marketplace_failures_total{{action=\"{action}\"}}"] += 1


@contextmanager
def cycle_timer(dry_run: bool = False):
    """Context manager used by the OSS loop to time a cycle."""
    start = time.perf_counter()
    try:
        yield
    except Exception:
        record_cycle(time.perf_counter() - start, dry_run=dry_run, outcome="error")
        raise
    else:
        record_cycle(time.perf_counter() - start, dry_run=dry_run, outcome="ok")


def _metric_lines() -> Iterable[str]:
    yield "# HELP gh05t3_mvs_cycle_duration_seconds Duration of OSS cycle execution."
    yield "# TYPE gh05t3_mvs_cycle_duration_seconds summary"
    yield f"gh05t3_mvs_cycle_duration_seconds_sum {_CYCLE_DURATION_SUM:.6f}"
    yield f"gh05t3_mvs_cycle_duration_seconds_count {_CYCLE_COUNT}"
    if _CYCLE_COUNT:
        yield f"gh05t3_mvs_cycle_duration_seconds {_LAST_CYCLE_DURATION:.6f}"

    yield "# HELP gh05t3_mvs_cycle_dry_runs_total OSS cycles run in dry-run mode."
    yield "# TYPE gh05t3_mvs_cycle_dry_runs_total counter"
    yield f"gh05t3_mvs_cycle_dry_runs_total {_CYCLE_DRY_RUN_COUNT}"

    for key, value in sorted(_COUNTERS.items()):
        if "{" in key:
            metric, labels = key.split("{", 1)
            labels = "{" + labels
        else:
            metric, labels = key, ""
        yield f"{metric}{labels} {value}"


def metrics_payload() -> Tuple[bytes, str]:
    """Return body + media_type for the /oss/metrics endpoint."""
    body = "\n".join(_metric_lines()).encode("utf-8")
    return body, "text/plain; version=0.0.4; charset=utf-8"

