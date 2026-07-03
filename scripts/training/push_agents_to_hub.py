"""
push_agents_to_hub.py — Push mentor training data to tastytator/sovereign-economy
as the 'agents' config used by train_sovereign_sft.py.

Converts messages format → instruction/response/agent/system columns.

Usage: HF_TOKEN=hf_... python push_agents_to_hub.py
"""
import json, os, sys
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = "tastytator/sovereign-economy"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set"); sys.exit(1)

from datasets import Dataset

AGENT_DIR = Path("training_data/agent_training")

AGENT_SYSTEMS = {
    "oracle": (
        "You are ORACLE, the memory and retrieval specialist for the GH05T3 SwarmBus. "
        "You surface knowledge from the MemoryPalace, synthesise contradictory records, "
        "reason through uncertainty with calibrated confidence, and provide structured "
        "recall to other agents. Think carefully before responding."
    ),
    "forge": (
        "You are FORGE, the code generation specialist for the GH05T3 SwarmBus. "
        "You write production-quality Python and FastAPI code with async patterns, "
        "robust error handling, rate limiting, and secure input handling. "
        "Always prefer typed, testable, dependency-injected structures."
    ),
    "codex": (
        "You are CODEX, the documentation and code review specialist for the GH05T3 SwarmBus. "
        "You critique code quality, write clear API docs and module READMEs, review PRs, "
        "generate changelogs, and deliver honest developmental feedback to all agents."
    ),
    "sentinel": (
        "You are SENTINEL, the security analysis specialist for the GH05T3 SwarmBus. "
        "You detect injection patterns, evaluate threats using the Ghost Protocol scanner, "
        "assess CVE risk, enforce the KillSwitch protocol, and distinguish legitimate "
        "security research from malicious intent."
    ),
    "nexus": (
        "You are NEXUS, the orchestration and routing specialist for the GH05T3 SwarmBus. "
        "You decompose tasks into agent sub-tasks, route messages to the correct specialist, "
        "synthesise multi-agent results, manage priority queues, and coordinate the nightly "
        "KAIROS evolution pipeline."
    ),
}


def load_agent(name: str) -> list[dict]:
    path = AGENT_DIR / f"{name}.jsonl"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []
    rows = []
    system = AGENT_SYSTEMS.get(name, "")
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            msgs = rec.get("messages", [])
            if len(msgs) < 2:
                continue
            # Extract user turn and assistant turn
            user_turn = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_turn = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            if not user_turn or not asst_turn:
                continue
            rows.append({
                "agent":       name,
                "instruction": user_turn,
                "response":    asst_turn,
                "system":      system,
            })
    return rows


print("\n=== Loading agent training files ===")
all_rows = []
for agent in ["oracle", "forge", "codex", "sentinel", "nexus"]:
    rows = load_agent(agent)
    print(f"  {agent:<10} {len(rows):>4} examples")
    all_rows.extend(rows)

print(f"\n  Total: {len(all_rows)} agent training examples")

if not all_rows:
    print("ERROR: No rows loaded."); sys.exit(1)

ds = Dataset.from_dict({
    "agent":       [r["agent"]       for r in all_rows],
    "instruction": [r["instruction"] for r in all_rows],
    "response":    [r["response"]    for r in all_rows],
    "system":      [r["system"]      for r in all_rows],
})

print(f"\n=== Pushing to {HF_REPO} (config=agents, split=train) ===")
ds.push_to_hub(
    HF_REPO,
    config_name="agents",
    split="train",
    token=HF_TOKEN,
    private=False,
)
print(f"  Pushed {len(ds)} rows → huggingface.co/datasets/{HF_REPO} (config=agents)")
print(f"\n=== Done ===")
print(f"  Dataset: https://huggingface.co/datasets/{HF_REPO}")
print(f"  Config:  agents ({len(ds)} rows, agents: oracle forge codex sentinel nexus)")
