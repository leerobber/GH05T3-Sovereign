#!/usr/bin/env python3
"""Profile AgentHandle.act() p95 latency — Phase 1 gate metric."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.oss.omni_dna import create_omnidna
from backend.oss.genomic_substrate import AgentHandle


def main(n: int = 50) -> int:
    import os
    os.environ["MVS_DRY_RUN"] = "1"
    dna = create_omnidna("SCIENTIST", seed=42)
    handle = AgentHandle(genome_id=dna.genome_id, role="SCIENTIST", dna=dna)
    task = {"prompt": "Summarize volatility regime detection in 2 sentences.", "type": "analysis"}

    latencies: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        handle.act(task)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95_idx = min(len(latencies) - 1, int(len(latencies) * 0.95))
    p95 = latencies[p95_idx]
    mean = statistics.mean(latencies)

    print(f"act() dry-run profile n={n}")
    print(f"  mean: {mean:.1f}ms")
    print(f"  p50:  {p50:.1f}ms")
    print(f"  p95:  {p95:.1f}ms")
    print(f"  gate target: p95 < 50ms (Week 2) / <100ms (Week 1)")

    # Write metric hint for checkpoint CLI
    ok_week1 = p95 < 100
    ok_week2 = p95 < 50
    print(f"  week1 <100ms: {'PASS' if ok_week1 else 'FAIL'}")
    print(f"  week2 <50ms:  {'PASS' if ok_week2 else 'FAIL'}")
    return 0 if ok_week1 else 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    raise SystemExit(main(n))