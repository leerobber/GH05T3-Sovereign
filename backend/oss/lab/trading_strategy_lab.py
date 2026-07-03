"""
OSS Lab: Old Stack vs GenomicSubstrate (New Substrate) — Trading Strategy Design
================================================================================

Same concrete problem:
  "Design and iteratively refine a trading strategy for a synthetic volatile market."

Old stack (files / classes / APIs):
  - Static TradingStrategy class
  - Manual param tweaking by human
  - Fixed backtester "API"

New OSS stack (GenomicSubstrate + agents + Omni-DNA + Mind + Economy):
  - Living genomes with traits (risk_tolerance, pattern_detection...)
  - Agents spawned from substrate
  - Evolution via mutate/crossover driven by fitness (rewards)
  - Emergent strategies from DNA, not written code
  - NeuroCoin rewards + trait influence

Run:
  python -m backend.oss.lab.trading_strategy_lab

Watch how the "system" thinks and evolves differently.
"""

from __future__ import annotations
import random
import json
import time
from pathlib import Path
import sys

# Robust imports
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

try:
    from ..mvs import get_mvs, create_omnidna, get_theorist_population
    from oss.loop import run_cycle
except Exception:
    from backend.oss.mvs import get_mvs, create_omnidna, get_theorist_population
    from backend.oss.loop import run_cycle

# ---------------------------------------------------------------------------
# OLD STACK (traditional — files/classes/APIs)
# ---------------------------------------------------------------------------

class TradingStrategy:
    """Static class. Behavior frozen at definition time."""
    def __init__(self, params):
        self.params = params  # e.g. {"fast": 10, "slow": 50, "risk": 0.5}

    def generate_signals(self, prices: list[float]) -> list[str]:
        # Classic moving average crossover — rigid
        fast, slow = self.params.get("fast", 10), self.params.get("slow", 50)
        signals = []
        for i in range(len(prices)):
            if i < slow:
                signals.append("HOLD")
                continue
            ma_fast = sum(prices[i-fast+1:i+1]) / fast
            ma_slow = sum(prices[i-slow+1:i+1]) / slow
            if ma_fast > ma_slow:
                signals.append("BUY")
            elif ma_fast < ma_slow:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        return signals

class Backtester:
    """The 'API' / service class."""
    def __init__(self, prices: list[float], strategy: TradingStrategy):
        self.prices = prices
        self.strategy = strategy

    def run(self) -> dict:
        signals = self.strategy.generate_signals(self.prices)
        equity = 100.0
        position = 0
        for i, sig in enumerate(signals):
            price = self.prices[i]
            if sig == "BUY" and position == 0:
                position = equity / price
                equity = 0
            elif sig == "SELL" and position > 0:
                equity = position * price
                position = 0
        final = equity + (position * self.prices[-1] if position else 0)
        return {
            "final_equity": round(final, 2),
            "return_pct": round((final - 100) / 100 * 100, 1),
            "params_used": self.strategy.params
        }

def old_stack_lab(prices: list[float], initial_params: dict) -> dict:
    """Human-driven iteration (old paradigm)."""
    print("\n=== OLD STACK (files/classes/APIs) ===")
    strategy = TradingStrategy(initial_params)
    results = Backtester(prices, strategy).run()
    print("Initial backtest:", results)

    # Human manually edits "the code"
    new_params = {**initial_params, "fast": initial_params.get("fast", 10) + 3, "risk": 0.6}
    strategy2 = TradingStrategy(new_params)
    results2 = Backtester(prices, strategy2).run()
    print("After human tweak:", results2)
    print("Evolution: manual code change + redeploy. No learning in the artifact itself.")
    return results2

# ---------------------------------------------------------------------------
# NEW OSS STACK (GenomicSubstrate + living agents)
# ---------------------------------------------------------------------------

def make_synthetic_prices(length: int = 120, volatility: float = 0.018) -> list[float]:
    price = 100.0
    prices = []
    for _ in range(length):
        price *= (1 + random.gauss(0.0008, volatility))
        prices.append(max(30, price))
    return prices

def new_stack_lab(prices: list[float], cycles: int = 5) -> dict:
    """Substrate + agents + evolution (new paradigm)."""
    print("\n=== OSS STACK (MVS: OmniDNA + GenomicSubstrate + Mind + Economy) ===")

    mvs = get_mvs()
    sub = mvs["substrate"]

    # Bootstrap using stabilized MVS OmniDNA
    gids = []
    dnas = []
    for _ in range(3):
        dna = create_omnidna("INVESTOR")
        gid = sub.register_genome(dna)
        gids.append(gid)
        dnas.append(dna)
        print(f"  Registered genome {gid} with OmniDNA traits")

    best_fitness = 0.0
    best_gid = None

    for c in range(cycles):
        # Query living genomes (no import, no class lookup)
        candidates = sub.query_by_capability(domain="markets", skill="trading", min_level=0.6) or gids
        agents = [sub.spawn_agent(g, role="Investor") for g in candidates[:2]]

        cycle_fitnesses = []
        for idx, agent in enumerate(agents):
            dna = dnas[idx % len(dnas)]
            # Phase 2: DNA-conditioned LLM for strategy design (MVS sole path)
            design_task = {
                "prompt": "As an Investor agent, design or refine a trading strategy for a volatile synthetic market. Suggest specific MA periods or risk rules that fit your traits.",
                "summary": "trading strategy design"
            }
            act_result = agent.act(design_task)
            llm_design = act_result.get("raw_output", "")
            print(f"  DNA-conditioned design (traits modulate LLM): {llm_design[:120]}...")

            t = dna.get_traits()
            params = {"fast_ma": 10 + int(t.get("creativity", 0.5) * 8), "slow_ma": 40 + int(t.get("risk_tolerance", 0.5) * 30)}

            # Run mini-backtest using the emergent params (simulates phenotype from DNA)
            strat_like = TradingStrategy({"fast": params["fast_ma"], "slow": params["slow_ma"]})
            res = Backtester(prices, strat_like).run()
            fitness = min(0.99, max(0.1, (res["return_pct"] + 20) / 60 ))
            cycle_fitnesses.append(fitness)

            sub.record_fitness(agent.genome_id, fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_gid = agent.genome_id

            # Evolve using MVS OmniDNA
            sub.mutate(agent.genome_id, intensity=0.08 if fitness < 0.6 else 0.04)

        # Occasionally crossover (speciation / memetic spread)
        if c % 2 == 1 and len(candidates) >= 2:
            child_a, child_b = sub.crossover(candidates[0], candidates[1])
            print(f"  Crossover produced new genomes {child_a}, {child_b}")

        print(f"  Cycle {c}: best fitness this round {max(cycle_fitnesses):.3f} (global {best_fitness:.3f})")

    # Final evolved strategy
    if best_gid:
        final_handle = sub.spawn_agent(best_gid)
        final_design = final_handle.act({"prompt": "Summarize best strategy."})
        print(f"\nEvolved best genome {best_gid} DNA phenotype: {str(final_design.get('raw_output',''))[:100]}...")

    print("Evolution: automatic, reward-driven, DNA-level (via MVS). Strategies are living, not coded.")
    print("MVS Substrate stats:", sub.stats())

    # Bonus: feed into OSS loop for full state machine
    print("\nFeeding one cycle into full OSS loop (state machines + NeuroCoins)...")
    run_cycle(99, dry_run=True, verbose=False)

    return {"best_fitness": round(best_fitness, 4), "best_genome": best_gid, "designs": []}

# ---------------------------------------------------------------------------
# Visualization stubs (adapted from pasted vision)
# ---------------------------------------------------------------------------

def simple_lineage_and_fitness_viz(sub: GenomicSubstrate):
    """Minimal matplotlib-free viz using prints + dicts (real version would use the pasted plots)."""
    print("\n--- Species Snapshot (Lineage + Fitness) ---")
    for gid, rec in list(sub.genomes.items())[:4]:
        print(f"{gid[:8]} | role={rec.role} | fitness_avg={sum(rec.fitness_history)/max(1,len(rec.fitness_history)):.3f} | lineage={rec.lineage}")
    print("(Full viz would call plot_lineage_tree, plot_fitness_over_time, plot_trait_evolution, plot_wealth_distribution from the vision)")

if __name__ == "__main__":
    random.seed(42)  # reproducible
    prices = make_synthetic_prices(100, 0.022)

    old_results = old_stack_lab(prices, {"fast": 8, "slow": 35})
    new_results = new_stack_lab(prices, cycles=6)

    print("\n" + "="*70)
    print("COMPARISON")
    print(f"Old stack final return: {old_results.get('return_pct', 'N/A')}%  (manual human edits only)")
    print(f"New OSS stack peak fitness: {new_results.get('best_fitness', 0)}  (autonomous DNA evolution + swarm)")
    print("The substrate makes the strategy a living, evolvable entity — no files to edit, no classes to rewrite, no APIs to call.")
    print("="*70)

    sub = get_mvs()["substrate"]
    simple_lineage_and_fitness_viz(sub)
    print("\nLab complete. The future is genomes, swarms, and economies — not files, classes, or endpoints.")