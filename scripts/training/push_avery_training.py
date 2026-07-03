#!/usr/bin/env python3
"""
push_avery_training.py
======================
Consolidates all economy + Avery training data and pushes to HuggingFace.

Sources:
  - C:/Users/leer4/Documents/agent-economy/data/training/  (~115K records)
  - C:/Users/leer4/GH05T3/training_data/                   (~652 Avery records)

Output:
  tastytator/sovereign-economy  config=sft  split=train

Usage:
  python push_avery_training.py
  python push_avery_training.py --dry-run   # count only, no push
  python push_avery_training.py --sample 5  # print 5 sample rows
"""
import argparse
import json
import os
import sys
from pathlib import Path

HF_TOKEN    = os.environ.get("HF_TOKEN") or (
    lambda p: p.read_text().strip() if p.exists() else ""
)(Path("C:/Users/leer4/.cache/huggingface/token"))
HF_REPO     = "tastytator/sovereign-economy"
CONFIG_NAME = "sft"

ECONOMY_DIR = Path("C:/Users/leer4/Documents/agent-economy/data/training")
AVERY_DIR   = Path("C:/Users/leer4/GH05T3/training_data")

# ─────────────────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass
    except Exception as e:
        print(f"  WARN: {path.name}: {e}", file=sys.stderr)


def to_row(rec: dict) -> dict | None:
    """Normalize any known format to {instruction, response}."""
    # ── messages format ──────────────────────────────────────────────────────
    if "messages" in rec:
        msgs = rec["messages"]
        users = [m["content"] for m in msgs if m.get("role") == "user"]
        asst  = [m["content"] for m in msgs if m.get("role") == "assistant"]
        if users and asst:
            instr = users[0].strip()
            resp  = asst[-1].strip()
            if len(instr) > 20 and len(resp) > 20:
                return {"instruction": instr, "response": resp}

    # ── instruction / response ────────────────────────────────────────────────
    if "instruction" in rec and "response" in rec:
        instr = str(rec["instruction"]).strip()
        resp  = str(rec["response"]).strip()
        if len(instr) > 20 and len(resp) > 20:
            return {"instruction": instr, "response": resp}

    # ── causal insight (has instruction but no response — skip) ──────────────
    return None


def load_dir(directory: Path, label: str) -> list[dict]:
    rows = []
    files = sorted(directory.rglob("*.jsonl"))
    for f in files:
        for rec in iter_jsonl(f):
            row = to_row(rec)
            if row:
                rows.append(row)
    print(f"  {label}: {len(rows):,} rows from {len(files)} files")
    return rows


def dedup(rows: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for r in rows:
        key = r["instruction"][:120]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",  action="store_true", help="Count only, no push")
    ap.add_argument("--sample",   type=int, default=0, help="Print N sample rows")
    args = ap.parse_args()

    print("=" * 60)
    print("  Avery Training Data → HuggingFace Push")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\nLoading sources:")
    economy_rows = load_dir(ECONOMY_DIR, "Economy (agent-economy)")
    avery_rows   = load_dir(AVERY_DIR,   "Avery local (GH05T3)")

    all_rows = economy_rows + avery_rows
    print(f"\n  Combined raw: {len(all_rows):,}")

    unique = dedup(all_rows)
    print(f"  After dedup:  {len(unique):,}")

    if args.sample:
        import random
        for r in random.sample(unique, min(args.sample, len(unique))):
            print(f"\n  INSTRUCTION: {r['instruction'][:120]}")
            print(f"  RESPONSE:    {r['response'][:120]}")

    if args.dry_run:
        print("\n  --dry-run: no push.")
        return

    if not HF_TOKEN:
        print("\nERROR: HF_TOKEN not set. Set env var or add token to ~/.cache/huggingface/token")
        sys.exit(1)

    # ── Push ──────────────────────────────────────────────────────────────────
    print(f"\nPushing {len(unique):,} rows to {HF_REPO} (config={CONFIG_NAME})...")

    try:
        from datasets import Dataset
        from huggingface_hub import login
    except ImportError:
        print("ERROR: pip install datasets huggingface_hub")
        sys.exit(1)

    login(token=HF_TOKEN, add_to_git_credential=False)

    ds = Dataset.from_list(unique)
    ds.push_to_hub(
        HF_REPO,
        config_name=CONFIG_NAME,
        split="train",
        token=HF_TOKEN,
        commit_message=f"Update: {len(unique):,} records (economy + Avery)"
    )

    print(f"\n✓ Done — {len(unique):,} records pushed to {HF_REPO}/{CONFIG_NAME}/train")
    print(f"  https://huggingface.co/datasets/{HF_REPO}")


if __name__ == "__main__":
    main()
