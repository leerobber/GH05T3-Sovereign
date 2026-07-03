#!/usr/bin/env python3
"""
BME stress runner.

Seeds a synthetic population into ChronosLedger + GenomePlane and then runs
the Binary Multiverse Engine for a configurable number of cycles.

Examples:
  python scripts/bme_stress.py --cycles 10 --population 28
  python scripts/bme_stress.py --cycles 100 --population 140 --target-universe 4 --no-migration
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.oss.core.aethyro_bridge import AethyroBridge
from backend.oss.core.bme_bridge import BMEBridge
from backend.oss.core.chronos_ledger import ChronosLedger, ROLE_TIER_SHIFT, UNIVERSE_SHIFT
from backend.oss.core.genome_plane import GENE_FLAG_ACTIVE, get_genome_plane
from backend.oss.core.skill_registry import UNIVERSE_NAMES, get_skill_registry


DESIRE_PROFILES = {
    0: np.array([0.45, 0.90, 0.70, 0.25, 0.45, 0.35, 0.25], dtype=np.float32),
    1: np.array([0.50, 0.85, 0.80, 0.30, 0.55, 0.30, 0.30], dtype=np.float32),
    2: np.array([0.65, 0.55, 0.55, 0.75, 0.50, 0.55, 0.30], dtype=np.float32),
    3: np.array([0.45, 0.55, 0.45, 0.80, 0.40, 0.80, 0.70], dtype=np.float32),
    4: np.array([0.60, 0.45, 0.45, 0.70, 0.55, 0.90, 0.45], dtype=np.float32),
    5: np.array([0.35, 0.75, 0.60, 0.25, 0.60, 0.85, 0.60], dtype=np.float32),
    6: np.array([0.55, 0.35, 0.70, 0.20, 0.80, 0.70, 0.25], dtype=np.float32),
    7: np.array([0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70], dtype=np.float32),
}


def _seed_population(
    ledger: ChronosLedger,
    genome,
    population: int,
    seed: int,
) -> tuple[AethyroBridge, list[int]]:
    rng = np.random.default_rng(seed)
    registry = get_skill_registry()
    bridge = AethyroBridge(ledger=ledger)
    active_slots: list[int] = []

    for idx in range(population):
        universe_id = idx % 8
        role_tier = idx % 6
        agent_id = f"bme-{UNIVERSE_NAMES.get(universe_id, universe_id).lower()}-{idx:04d}"
        slot = bridge.alloc_slot(agent_id)

        profile = DESIRE_PROFILES.get(universe_id, DESIRE_PROFILES[7])
        noise = rng.normal(0.0, 0.05, size=7).astype(np.float32)
        desires = np.clip(profile + noise, 0.0, 1.0)
        maturity = int(np.clip(1 + (idx % 8), 1, 8))
        fitness = float(np.clip(0.35 + 0.04 * (idx % 7) + rng.normal(0.0, 0.03), 0.05, 0.95))
        scratchpad = ((universe_id & 0b111) << UNIVERSE_SHIFT) | ((role_tier & 0b111) << ROLE_TIER_SHIFT)

        ledger.write_agent(
            index=slot,
            desires=tuple(float(v) for v in desires),
            maturity=maturity,
            fitness=fitness,
            generation=idx % 255,
            scratchpad=scratchpad,
        )

        genome.clear_genome(slot)
        seed_skills = registry.get_universe_skills(universe_id)
        genes = []
        for gene_idx, skill in enumerate(seed_skills[:3]):
            expr = float(np.clip(0.45 + 0.08 * role_tier + 0.04 * gene_idx, 0.05, 1.0))
            genes.append(
                (
                    int(skill["universe_id"]),
                    int(min(role_tier, skill["role_tier"])),
                    int(skill["skill_id"]),
                    expr,
                    GENE_FLAG_ACTIVE,
                )
            )
        if genes:
            genome.write_genome(slot, genes)

        active_slots.append(slot)

    ledger.flush()
    return bridge, active_slots


def _print_cycle(prefix: str, cycle: int, result: dict) -> None:
    universe_distribution = result.get("universe_distribution", {})
    role_counts = result.get("role_counts", {})
    reward_summary = result.get("reward_summary", {})
    trait_summary = result.get("trait_summary", {})
    print(
        f"{prefix}[{cycle:03d}] "
        f"agents={result.get('active_slots', 0)} "
        f"migrations={result.get('migrations', 0)} "
        f"promotions={result.get('promotions', 0)} "
        f"breakthroughs={result.get('breakthroughs', 0)} "
        f"fitness={reward_summary.get('mean_fitness', 0.0):.4f} "
        f"genes={trait_summary.get('mean_active_genes', 0.0)} "
        f"universes={universe_distribution} "
        f"roles={role_counts}"
    )


def run_stress(
    cycles: int,
    population: int,
    target_universe: int | None,
    allow_migration: bool,
    seed: int,
) -> int:
    ledger = ChronosLedger()
    genome = get_genome_plane()
    bridge, active_slots = _seed_population(ledger, genome, population, seed)
    bme = BMEBridge(ledger=ledger, genome=genome)
    effective_allow_migration = allow_migration and target_universe is None

    print("=== BME Stress Run ===")
    print(
        f"population={population} cycles={cycles} "
        f"target_universe={target_universe} allow_migration={effective_allow_migration}"
    )
    print(f"seeded_slots={len(active_slots)}")

    for cycle in range(cycles):
        result = bme.universe_pass(
            active_slots=active_slots,
            target_universe=target_universe,
            allow_migration=effective_allow_migration,
            allow_promotion=True,
            allow_breakthrough=True,
        )
        stats = bme.collect_stats(active_slots)
        result.update(stats)
        _print_cycle("BME", cycle, result)

    final_stats = bme.collect_stats(active_slots)
    print("=== Final Summary ===")
    print(final_stats)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--population", type=int, default=28)
    parser.add_argument("--target-universe", type=int, default=None)
    parser.add_argument("--no-migration", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return run_stress(
        cycles=args.cycles,
        population=args.population,
        target_universe=args.target_universe,
        allow_migration=not args.no_migration,
        seed=args.seed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
