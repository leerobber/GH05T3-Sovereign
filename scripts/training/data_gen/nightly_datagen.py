"""
nightly_datagen.py — Runs nightly to auto-grow the training dataset.

Add to APScheduler in server.py:
    scheduler.add_job(_run_datagen, "cron", hour=1, minute=30)

Or run standalone:
    python data_gen/nightly_datagen.py
"""

import os
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_gen.synth_generator import generate, push_to_hf
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / "backend" / ".env")

log = logging.getLogger("sovereign.datagen")

# How many new rows to generate per agent per night
NIGHTLY_COUNTS = {
    "forge":    30,   # code gen — templates are fast
    "sentinel": 20,   # security — templates cover all OWASP top 10
    "avery":    15,   # business — local model for variety
    "nexus":    15,   # orchestration — templates good
    "oracle":   10,   # retrieval — smaller set, quality over quantity
    "codex":    10,   # docs — slower, local model
}


def run_nightly(push: bool = True):
    token = os.environ.get("HF_TOKEN", "")
    dataset_repo = os.environ.get("HF_DATASET", "tastytator/sovereign-economy")

    log.info("Nightly data generation starting...")
    total = 0

    for agent, count in NIGHTLY_COUNTS.items():
        try:
            log.info("Generating %d rows for %s...", count, agent)
            rows = generate(agent, count, method="auto")

            if rows and push and token:
                push_to_hf(rows, agent, token, dataset_repo)
                log.info("%s: pushed %d rows", agent, len(rows))

            # Also generate DPO pairs for agents that benefit most
            if agent in ("forge", "sentinel"):
                dpo_rows = generate(agent, count // 2, method="auto", dpo=True)
                if dpo_rows and push and token:
                    push_to_hf(dpo_rows, agent, token, dataset_repo)
                    log.info("%s DPO: pushed %d pairs", agent, len(dpo_rows))

            total += len(rows)
        except Exception as e:
            log.error("Failed generating %s: %s", agent, e)

    log.info("Nightly datagen complete. Total rows generated: %d", total)
    return total


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    run_nightly(push="--no-push" not in sys.argv)
