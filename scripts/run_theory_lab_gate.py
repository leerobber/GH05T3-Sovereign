#!/usr/bin/env python3
"""Phase 2 gate — 500-cycle Theory Lab dry-run."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MVS_DRY_RUN", "1")

from backend.oss.lab.theory_lab import TheoryLab


def main() -> int:
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    t0 = time.time()
    lab = TheoryLab(cycles=cycles, live=False, fast_dry_run=True)
    lab.run()
    elapsed = time.time() - t0
    print(f"Gate complete: {cycles} cycles in {elapsed:.1f}s | volatility={lab._volatility_cycles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())