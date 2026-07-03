"""
upload_economy_data.py
======================
Loads all 113K+ economy training records from the agent-economy simulation,
normalizes them into messages (SFT) format, and pushes to HuggingFace as:
  tastytator/sovereign-economy  [config=economy, split=train]

Also uploads a filtered causal_insight split for research/monetization:
  tastytator/sovereign-economy  [config=economy, split=causal]

Run: python upload_economy_data.py
     python upload_economy_data.py --dry-run   (count only, no upload)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
ECONOMY_DIR = Path(r"C:\Users\leer4\Documents\agent-economy\data\training")
HF_REPO     = "tastytator/sovereign-economy"

# Disable XET transfer to prevent mid-stream failures on large uploads
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def _load_env():
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

_load_env()
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set in .env")
    sys.exit(1)


# ── Load all economy JSONL files ──────────────────────────────────────────────
def load_all_records() -> list[dict]:
    files = sorted(ECONOMY_DIR.glob("training_*.jsonl"))
    print(f"Found {len(files)} JSONL files in {ECONOMY_DIR}")
    records = []
    for f in files:
        try:
            for line in f.open(encoding="utf-8"):
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
    print(f"Loaded {len(records):,} raw records")
    return records


# ── Normalize to messages format ──────────────────────────────────────────────
def normalize_record(r: dict) -> dict | None:
    """
    Convert any economy record schema into messages format for SFT.
    Returns None if the record should be skipped (too short / no content).
    """
    # Already in messages format (new exports)
    if "messages" in r:
        msgs = r["messages"]
        user_turns = [m["content"] for m in msgs if m.get("role") == "user"]
        asst_turns = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if not user_turns or not asst_turns:
            return None
        instruction = user_turns[0].strip()
        response    = asst_turns[-1].strip()
        if len(instruction) < 20 or len(response) < 20:
            return None
        return {
            "messages": [
                {"role": "user",      "content": instruction},
                {"role": "assistant", "content": response},
            ],
            "type":   r.get("type", "task_matching"),
            "source": "economy_simulation",
        }

    # Flat format: instruction + context (causal insights, task records)
    instruction = (r.get("instruction") or r.get("prompt") or "").strip()
    context     = (r.get("context") or "").strip()
    record_type = r.get("type", "economy")

    if not instruction or len(instruction) < 20:
        return None

    # Build assistant response from context
    if context and context != instruction:
        response = context
    elif record_type == "causal_insight":
        # The instruction IS the finding — make a Q&A pair
        # e.g. "reputation → credit_earned ATE +166.3..."
        question = "What does causal analysis of the agent economy reveal about " + \
                   instruction.split(":")[0].replace("Agent economy causal analysis tick ", "economic performance at tick ") + "?"
        response = instruction.split(":", 1)[-1].strip() if ":" in instruction else instruction
        return {
            "messages": [
                {"role": "user",      "content": question},
                {"role": "assistant", "content": response},
            ],
            "type":   "causal_insight",
            "source": "economy_simulation",
        }
    else:
        response = instruction  # fallback — shouldn't happen often

    if len(response) < 20:
        return None

    return {
        "messages": [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": response},
        ],
        "type":   record_type,
        "source": "economy_simulation",
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(dry_run: bool = False):
    print(f"\n{'='*55}")
    print(f"  Economy Data -> HuggingFace Upload")
    print(f"  Repo  : {HF_REPO}")
    print(f"  Mode  : {'DRY RUN' if dry_run else 'LIVE UPLOAD'}")
    print(f"{'='*55}\n")

    # Load all records
    raw = load_all_records()

    # Normalize
    normalized, skipped = [], 0
    causal = []

    for r in raw:
        result = normalize_record(r)
        if result is None:
            skipped += 1
            continue
        normalized.append(result)
        if result.get("type") == "causal_insight":
            causal.append(result)

    print(f"\nNormalized : {len(normalized):,} records")
    print(f"Skipped    : {skipped:,} records")
    print(f"Causal     : {len(causal):,} causal insight records")

    # Type breakdown
    from collections import Counter
    types = Counter(r.get("type", "unknown") for r in normalized)
    print("\nRecord types:")
    for t, n in types.most_common():
        print(f"  {t:<30} {n:>6,}")

    if dry_run:
        print("\nDRY RUN — no upload performed.")
        return

    from datasets import Dataset

    # Upload full economy split
    print(f"\nUploading {len(normalized):,} records → [{HF_REPO}] economy/train ...")
    ds_full = Dataset.from_list(normalized)
    ds_full.push_to_hub(
        HF_REPO,
        config_name="economy",
        split="train",
        token=HF_TOKEN,
        private=False,
    )
    print(f"  ✅ economy/train pushed: {len(ds_full):,} rows")

    # Upload causal insights as separate split (monetization / research)
    if causal:
        print(f"\nUploading {len(causal):,} causal records → [{HF_REPO}] economy/causal ...")
        ds_causal = Dataset.from_list(causal)
        ds_causal.push_to_hub(
            HF_REPO,
            config_name="economy",
            split="causal",
            token=HF_TOKEN,
            private=False,
        )
        print(f"  ✅ economy/causal pushed: {len(ds_causal):,} rows")

    print(f"\n{'='*55}")
    print(f"  Upload complete.")
    print(f"  View: https://huggingface.co/datasets/{HF_REPO}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Count only, no upload")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
