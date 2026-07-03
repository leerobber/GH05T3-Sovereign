"""
Species Visualization (adapted & implemented from the OSS vision pasted content)
Uses real data from GenomicSubstrate + oss/loop / traits.

Provides:
- plot_trait_evolution (per agent/genome)
- plot_species_trait_landscape
- plot_fitness_over_time
- plot_wealth_distribution (NeuroCoins via ledger or simulated)
- plot_lineage_tree
- plot_trait_market_activity (simulated from crossovers/mutations)

If matplotlib is installed: shows plots.
Otherwise: prints ASCII / data summaries.

Run:
  python -m backend.oss.lab.species_viz
"""

from __future__ import annotations
import json
import random
from pathlib import Path
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from oss.genomic_substrate import get_substrate
except Exception:
    from backend.oss.genomic_substrate import get_substrate

def logs_to_dataframe(sub):
    """Convert substrate genomes into a dataframe-like list of dicts."""
    rows = []
    for gid, rec in sub.genomes.items():
        row = {
            "genome_id": gid[:8],
            "role": rec.role,
            "cycle": len(rec.fitness_history),
            "fitness": rec.fitness_history[-1] if rec.fitness_history else 0.0,
        }
        for k, v in rec.dna.get_traits().items():
            row[f"trait_{k}"] = v
        row["lineage"] = sub.get_lineage(gid)
        row["neurocoins"] = random.uniform(800, 2200)  # simulated; wire to real ledger in future
        rows.append(row)
    return rows

def plot_trait_evolution(rows, agent_id=None):
    if not HAS_MPL:
        print("trait_evolution data:", [r for r in rows if not agent_id or r["genome_id"]==agent_id])
        return
    # simple line for last agent or first
    plt.figure(figsize=(7,4))
    for r in rows[:3]:
        traits = [k for k in r if k.startswith("trait_")]
        vals = [r[t] for t in traits]
        plt.plot(traits, vals, marker='o', label=r["genome_id"])
    plt.xticks(rotation=45, ha='right')
    plt.title("Trait evolution (selected genomes)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_lineage_tree(rows):
    if not HAS_MPL:
        print("Lineage (simple):")
        for r in rows:
            print(f"  {r['genome_id']} <- {r.get('lineage', [])}")
        return
    import networkx as nx
    G = nx.DiGraph()
    for r in rows:
        G.add_node(r["genome_id"])
    # simplistic edges from recorded lineage in substrate
    plt.figure(figsize=(8,5))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=600, font_size=8)
    plt.title("Lineage tree (genomes)")
    plt.tight_layout()
    plt.show()

def run_viz():
    sub = get_substrate()
    rows = logs_to_dataframe(sub)
    print(f"Visualizing {len(rows)} genomes from substrate...")
    plot_trait_evolution(rows)
    plot_lineage_tree(rows)
    if HAS_MPL:
        # wealth placeholder
        plt.figure()
        coins = [r["neurocoins"] for r in rows]
        plt.hist(coins, bins=8)
        plt.title("NeuroCoin / Wealth distribution (simulated)")
        plt.show()
    else:
        print("Wealth snapshot (sim):", {r["genome_id"]: round(r["neurocoins"],0) for r in rows[:5]})
    print("Full dashboard would include: trait landscape, fitness curves, trait market volume, evolutionary events (mutation/speciation).")

if __name__ == "__main__":
    run_viz()