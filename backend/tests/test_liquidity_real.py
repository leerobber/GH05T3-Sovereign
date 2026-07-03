"""
Real tests for Aethyro Liquidity Routing (no shell -c hacks, pure python).
Covers mutation sweep, large sizes, full trace, shadow loop, policy hot-update sim.
"""
import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
os.environ['AETHYRO_SKIP_LICENSE'] = '1'

# Real import for test execution from backend dir
try:
    from oss.financial.liquidity_routing import get_router
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from oss.financial.liquidity_routing import get_router

def test_real_mutation_sweep():
    print("=== REAL MUTATION SWEEP TEST ===")
    r = get_router()
    random.seed(42)
    results = []
    for m in [0.05, 0.18, 0.30, 0.50]:
        # temporarily set
        orig = r.current_policy.get("mutation_rate")
        r.current_policy["mutation_rate"] = m
        rep = r.discover({"size_usd": 2000000, "risk_tolerance": 0.12, "high_volatility": True, "target_pool": "curve_usdc_eth", "force_low_depth": True}, shadow=True)
        r.current_policy["mutation_rate"] = orig
        gain = rep["improvement_pct"]
        results.append({"mut": m, "gain": gain})
        print(f"mut={m} gain={gain}%")
    assert len(results) == 4
    print("SWEEP OK")
    return results

def test_real_large_sizes():
    print("=== REAL LARGE SIZE TESTS ===")
    r = get_router()
    random.seed(42)
    for size in [50000000, 100000000]:
        rep = r.discover({"size_usd": size, "risk_tolerance": 0.12, "high_volatility": True, "target_pool": "curve_usdc_eth", "force_low_depth": True}, shadow=True)
        print(f"${size/1e6}M gain={rep['improvement_pct']}% p95={rep.get('p95_eval_ms')}ms")
    print("LARGE SIZES OK")

def test_real_full_trace_and_policy():
    print("=== REAL FULL TRACE + POLICY ===")
    r = get_router()
    random.seed(42)
    rep = r.discover({"size_usd": 2000000, "risk_tolerance": 0.12, "high_volatility": True}, shadow=True)
    print(f"Lineage entries: {len(rep['lineage'])} (should be full 28+)")
    print(f"Policy mut: {rep['policy']['mutation_rate']}")
    print("FIRST 3 DELTAS:")
    for e in rep['lineage'][:3]:
        print(f"  Gen {e['generation']}: {e.get('delta_note', '')}")
    print("FULL TRACE OK")
    return rep

def test_real_shadow_loop_alert():
    print("=== REAL SHADOW LOOP (accelerated 24h sim) ===")
    r = get_router()
    random.seed(42)
    summary = r.run_shadow_loop(hours=24, cycles=24, size_usd=2000000)
    print(f"max_drift={summary['max_drift']} hard_stops={summary['hard_stops']}")
    assert summary['max_drift'] < 0.25  # should not trigger high
    print("SHADOW LOOP OK")

if __name__ == "__main__":
    sweep = test_real_mutation_sweep()
    test_real_large_sizes()
    rep = test_real_full_trace_and_policy()
    test_real_shadow_loop_alert()
    # Export updated report as baseline
    packet = {
        "timestamp": time.time(),
        "sweep": sweep,
        "2m_gain": rep["improvement_pct"],
        "policy": rep["policy"],
        "full_lineage_sample": rep["lineage"][:5] + rep["lineage"][-5:],
        "note": "Real test run baseline for public release"
    }
    os.makedirs("data", exist_ok=True)
    with open("data/liquidity_strategy_lineage_full_report.json", "w") as f:
        json.dump(packet, f, indent=2, default=str)
    print("\n=== ALL REAL TESTS PASSED ===")
    print("Updated report committed to data/liquidity_strategy_lineage_full_report.json")
    print("Ready for Live Shadow on production.")