import asyncio
import logging
from sentinel_core import SentinelCore
from sentinel_event import EventType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("OmniSentinelTest")

# Test cases
TEST_CASES = [
    {
        "name": "Valid Agent Spawn",
        "event": {
            "event_type": EventType.AGENT_SPAWN,
            "source": "agent_123",  # Must exist in aethyro_swarm.bin
            "payload": {
                "agent_type": "worker",
                "mutation_id": "mut_456"
            },
            "lineage": ["agent_789", "agent_123"]  # Must match ledger
        },
        "expected": {
            "threat_score": "<0.5",
            "action": "log"
        }
    },
    {
        "name": "DNA Violation (Rogue Agent)",
        "event": {
            "event_type": EventType.AGENT_MUTATION,
            "source": "agent_999",  # Doesn't exist in ledger
            "payload": {
                "mutation_id": "rogue_mut_1"
            },
            "lineage": ["agent_789"]
        },
        "expected": {
            "threat_score": ">=0.5",
            "action": "block"
        }
    },
    {
        "name": "Lineage Violation",
        "event": {
            "event_type": EventType.MEMORY_ACCESS,
            "source": "agent_123",
            "payload": {
                "memory_id": "mem_789"
            },
            "lineage": ["agent_999"]  # Incorrect lineage
        },
        "expected": {
            "threat_score": ">=0.2",
            "action": "notify"
        }
    }
]

async def run_test_case(test_case: dict) -> dict:
    """Run a single test case and return results."""
    logger.info(f"\n=== Testing: {test_case['name']} ===")
    logger.info(f"Input: {test_case['event']}")

    success = await SentinelCore.process_event(test_case["event"])

    return {
        "name": test_case["name"],
        "success": success,
        "expected": test_case["expected"]
    }

async def main():
    """Run all test cases and report results."""
    logger.info("Starting Omni Sentinel test suite...")

    results = await asyncio.gather(*[run_test_case(tc) for tc in TEST_CASES])

    logger.info("\n=== Test Results ===")
    for result in results:
        status = "PASS" if result["success"] else "FAIL"
        logger.info(f"{status}: {result['name']}")
        logger.info(f"  Expected: {result['expected']}")
        if not result["success"]:
            logger.error(f"  ERROR: Pipeline failed for {result['name']}")

    logger.info("\nTest suite completed.")

if __name__ == "__main__":
    asyncio.run(main())