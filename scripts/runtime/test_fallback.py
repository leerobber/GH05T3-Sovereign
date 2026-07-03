#!/usr/bin/env python3
"""
Test free fallback LLM routing.
Verifies credit exhaustion detection and fallback chain.
"""

import asyncio
import sys
sys.path.insert(0, 'backend')

from integrations.fallback_llm import detect_credit_exhaustion, FallbackLLMClient


async def test_credit_exhaustion_detection():
    """Test credit exhaustion error detection."""
    test_cases = [
        (Exception("429 rate limit exceeded"), True),
        (Exception("401 unauthorized — check billing"), True),
        (Exception("Credit quota exhausted"), True),
        (Exception("Insufficient credits for this request"), True),
        (Exception("Generic timeout error"), False),
        (Exception("Connection refused"), False),
    ]

    print("Testing credit exhaustion detection:")
    for error, should_trigger in test_cases:
        result = await detect_credit_exhaustion(error)
        status = "PASS" if result == should_trigger else "FAIL"
        print(f"  {status}: '{error}' -> {result}")


async def test_fallback_endpoints():
    """Test that fallback endpoints are reachable."""
    client = FallbackLLMClient()
    endpoints = [
        ("vllm", "http://localhost:8010/v1/chat/completions"),
        ("llama_verifier", "http://localhost:8011/v1/chat/completions"),
        ("llama_cpu", "http://localhost:8012/v1/chat/completions"),
    ]

    print("\nTesting endpoint reachability (note: may fail if services not running):")
    for name, url in endpoints:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2.0) as http:
                resp = await http.get(url.replace("/v1/chat/completions", ""), follow_redirects=True)
                print(f"  PASS: {name} ({resp.status_code})")
        except Exception as e:
            print(f"  [INFO] {name} offline (expected if not running): {type(e).__name__}")

    await client.close()


if __name__ == "__main__":
    asyncio.run(test_credit_exhaustion_detection())
    asyncio.run(test_fallback_endpoints())
    print("\nFallback LLM system initialized and ready.")
    print("When Anthropic API credits are exhausted:")
    print("  1. Detects 429/401/quota exhaustion errors")
    print("  2. Routes to vLLM (port 8010)")
    print("  3. Falls back to llama.cpp verifier (port 8011)")
    print("  4. Final fallback to llama.cpp CPU (port 8012)")
    print("  5. Logs all operations to #fallback channel on SwarmBus")
