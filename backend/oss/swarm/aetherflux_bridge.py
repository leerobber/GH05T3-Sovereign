"""Aetherflux expert fitness bridge: turns aetherflux-zero's measured
per-domain routing stats (see leerobber/aetherflux-zero's
examples/specialist_pilot.rs, run separately -- this repo has no Rust
toolchain dependency of its own) into real fitness scores on real
BinaryLedger agent slots (backend/oss/core/binary_ledger.py).

Why `recall` and not an arbitrary reward/credit unit: `BinaryLedger.fitness`
is documented and enforced to live in [0,1] (backend/oss/core/binary_ledger.py's
_f16 clips silently). A domain's held-out routing recall (fraction of its own
held-out text the router actually recognized as belonging to it) is already
bounded in [0,1] by construction -- no invented conversion factor needed,
unlike a credit/penalty score that could go negative or unbounded.

Scope, stated plainly (see docs/architecture/0008-aetherflux-expert-fitness-bridge.md):
this wires the measurement into the ledger. It does NOT yet feed genesis-ops'
spawn/prune logic or bme's speciation -- real, separate, unbuilt future work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.oss.core.binary_ledger import SCRATCH_NEEDS_REVIEW, BinaryLedger, get_binary_ledger

_DEFAULT_SLOT_MAP_PATH = Path(__file__).resolve().parents[2] / "data" / "aetherflux_slots.json"


@dataclass
class BridgeResult:
    name: str
    slot: int
    action: str  # "registered" or "updated"
    fitness_written: float
    needs_review: bool


def _load_slot_map(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_slot_map(path: Path, slot_map: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(slot_map, f, indent=2, sort_keys=True)


def register_or_update_experts(
    ledger: Optional[BinaryLedger],
    export_path: str,
    slot_map_path: Path = _DEFAULT_SLOT_MAP_PATH,
    live: bool = False,
) -> list[BridgeResult]:
    """Reads a specialist_pilot JSON export and registers/updates one real
    BinaryLedger agent slot per domain, using `recall` as the fitness value.

    `live=False` (the default): computes and returns what WOULD be written,
    without opening the ledger for any mutating call at all -- matches the
    dry-run-by-default convention chosen for this bridge, since ledger
    writes go straight into a live, shared mmap file with no undo.
    `live=True`: performs the real `write_at_next_available_slot`/
    `update_fitness`/`set_scratch_bit`/`clear_scratch_bit` calls.

    Idempotent across repeated runs via `slot_map_path`: a domain already
    present in that JSON side-file gets `update_fitness` on its existing
    slot; a domain seen for the first time gets a new slot via
    `write_at_next_available_slot`, and that slot index is persisted back
    to `slot_map_path` so the next run updates it instead of registering a
    second agent for the same domain.
    """
    with open(export_path) as f:
        export = json.load(f)

    slot_map = _load_slot_map(slot_map_path)
    results: list[BridgeResult] = []

    for domain in export["domains"]:
        name = domain["name"]
        recall = float(domain["recall"])
        needs_review = not bool(domain["specialized"])

        if name in slot_map:
            slot = slot_map[name]
            action = "updated"
        else:
            slot = None  # assigned below only if live, else reported as "(new)"
            action = "registered"

        if not live:
            results.append(BridgeResult(name=name, slot=slot if slot is not None else -1, action=f"{action} (dry-run)", fitness_written=recall, needs_review=needs_review))
            continue

        if action == "registered":
            slot = ledger.write_at_next_available_slot(desires=(0.5,) * 7, maturity=1, fitness=recall, generation=0)
            slot_map[name] = slot
        else:
            ledger.update_fitness(slot, recall)

        if needs_review:
            ledger.set_scratch_bit(slot, SCRATCH_NEEDS_REVIEW)
        else:
            ledger.clear_scratch_bit(slot, SCRATCH_NEEDS_REVIEW)

        results.append(BridgeResult(name=name, slot=slot, action=action, fitness_written=recall, needs_review=needs_review))

    if live:
        _save_slot_map(slot_map_path, slot_map)

    return results


def _print_report(results: list[BridgeResult], live: bool) -> None:
    mode = "LIVE" if live else "DRY-RUN"
    print(f"--- aetherflux expert fitness bridge ({mode}) ---")
    for r in results:
        slot_str = str(r.slot) if r.slot >= 0 else "(new)"
        flag = " [NEEDS_REVIEW]" if r.needs_review else ""
        print(f"  slot={slot_str:>6}  {r.action:<20}  fitness={r.fitness_written:.4f}{flag}  {r.name}")
    if not live:
        print("\n  dry-run only -- no ledger writes performed. Re-run with --live to actually write.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="path to specialist_pilot's JSON export")
    parser.add_argument("--live", action="store_true", help="actually write to the ledger (default: dry-run)")
    parser.add_argument("--ledger-path", default=None, help="override BinaryLedger path (default: real aethyro_swarm.bin)")
    parser.add_argument("--slot-map", default=str(_DEFAULT_SLOT_MAP_PATH), help="path to the domain->slot side file")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"error: export file not found: {args.input}", file=sys.stderr)
        return 1

    ledger = get_binary_ledger(args.ledger_path) if args.live else None
    try:
        results = register_or_update_experts(
            ledger,
            args.input,
            slot_map_path=Path(args.slot_map),
            live=args.live,
        )
    finally:
        if ledger is not None:
            ledger.close()

    _print_report(results, args.live)
    return 0


if __name__ == "__main__":
    sys.exit(main())
