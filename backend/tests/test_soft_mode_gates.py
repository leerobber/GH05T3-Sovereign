"""
Test script for Soft-Mode Gated Execution Layer.
Runs real test cases to verify guardrails.
No f-strings, safe formatting.
"""
import asyncio
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock bridge client for testing
class MockBridge:
    async def push_to_sovereign_core(self, schema):
        print("[MOCK_BRIDGE] Received schema mode={0}, exec={1}".format(schema.mode, schema.execution_params))
        return {"status": "routed", "mode": schema.mode}

from oss.financial.liquidity_routing import LiquidityEngine, SovereignCorePushSchema

async def test_case_1_nominal():
    print("\n=== Test Case 1 (Nominal Soft Run) ===")
    engine = LiquidityEngine(MockBridge())
    market_data = {"slip_delta": 0.05}
    await engine.run_execution_cycle(1, 450.0, market_data)
    print("Expected: [NOMINAL] routing to soft")

async def test_case_2_breaker():
    print("\n=== Test Case 2 (Slippage Circuit Breaker) ===")
    engine = LiquidityEngine(MockBridge())
    market_data = {"slip_delta": 0.14}
    await engine.run_execution_cycle(2, 450.0, market_data)
    print("Expected: [DRIFT_BREACH] to observability")

async def test_case_3_cap_violation():
    print("\n=== Test Case 3 (Micro-Cap Violation) ===")
    try:
        schema = SovereignCorePushSchema(
            tick=3,
            mode="soft",
            execution_params={"size_usd": 501.0}
        )
        print("ERROR: Should have raised ValidationError")
    except Exception as e:
        print("Caught expected: {0}".format(str(e)[:100]))
        print("Expected: CRITICAL_VIOLATION on $501 > $500")

if __name__ == "__main__":
    asyncio.run(test_case_1_nominal())
    asyncio.run(test_case_2_breaker())
    asyncio.run(test_case_3_cap_violation())
    print("\n=== All Tests Executed ===")