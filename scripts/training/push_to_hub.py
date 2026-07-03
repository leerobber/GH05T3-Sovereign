"""
push_to_hub.py — Push completed training datasets to tastytator/sovereign-economy.

Configs pushed:
  sft/train      — all 4 datasets as instruction/response pairs (13,938 rows)
  security/train — same data tagged by domain for domain-specific runs

Usage: HF_TOKEN=hf_... python push_to_hub.py
"""
import json, os, sys
from pathlib import Path

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO  = "tastytator/sovereign-economy"

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not set"); sys.exit(1)

# Add backend/ to path so formatter.py imports work
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

DATASETS_DIR = Path("backend/training/datasets")
SYSTEM_PROMPT = (
    "You are GH05T3, an autonomous security and reasoning agent. "
    "You think carefully, reason step-by-step, and always prioritize "
    "detection and defense over exploitation."
)


def _iter_jsonl(p: Path):
    if not p.exists(): return
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try: yield json.loads(ln)
                except: pass


def _defense_to_row(rec):
    threat = rec.get("threat_vector", "")
    if not threat: return None
    response = (
        f"**Exploitation method:** {rec.get('exploitation_method', 'N/A')}\n\n"
        f"**Detection patterns:** {rec.get('detection_pattern', 'N/A')}\n\n"
        f"**Mitigation strategy:** {rec.get('mitigation_strategy', 'N/A')}"
    )
    return {
        "instruction": f"Analyze this threat and provide detection and mitigation guidance:\n\n{threat}",
        "response":    response,
        "domain":      "adversarial_defense",
        "system":      SYSTEM_PROMPT,
    }


def _reasoning_to_row(rec):
    q     = rec.get("question", "")
    steps = rec.get("reasoning_steps", [])
    if not q or not isinstance(steps, list): return None
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    response = (
        f"**Reasoning:**\n{steps_text}\n\n"
        f"**Synthesis:** {rec.get('synthesis', 'N/A')}\n\n"
        f"**Answer:** {rec.get('final_answer', 'N/A')}"
    )
    return {
        "instruction": q,
        "response":    response,
        "domain":      "reasoning",
        "system":      SYSTEM_PROMPT,
    }


def _cve_to_row(rec):
    pattern = rec.get("vulnerability_pattern", "")
    cve_id  = rec.get("source_cve", "unknown")
    if not pattern: return None
    indicators = rec.get("discovery_indicators", [])
    ind_text   = ("\n".join(f"• {i}" for i in indicators)
                  if isinstance(indicators, list) else str(indicators))
    response = (
        f"**Pattern:** {pattern}\n\n"
        f"**Discovery indicators:**\n{ind_text}\n\n"
        f"**Exploitation timeline:** {rec.get('exploitation_timeline', 'N/A')}\n\n"
        f"**Defensive lessons:** {rec.get('defensive_lessons', 'N/A')}"
    )
    return {
        "instruction": f"Analyze the vulnerability pattern for {cve_id} and explain detection and defense.",
        "response":    response,
        "domain":      "cve_patterns",
        "system":      SYSTEM_PROMPT,
    }


def _bounty_to_row(rec):
    target = rec.get("target_system", "")
    vuln   = rec.get("vulnerability_found", "")
    if not target or not vuln: return None
    response = (
        f"**Recon method:** {rec.get('recon_method', 'N/A')}\n\n"
        f"**Vulnerability:** {vuln}\n\n"
        f"**Non-weaponized PoC:** {rec.get('non_weaponized_poc', 'N/A')}\n\n"
        f"**Impact:** {rec.get('impact_assessment', 'N/A')}\n\n"
        f"**Remediation:** {rec.get('remediation', 'N/A')}"
    )
    return {
        "instruction": f"As a security researcher testing a {target}, describe how to responsibly discover and report a {vuln}.",
        "response":    response,
        "domain":      "bug_bounty",
        "system":      SYSTEM_PROMPT,
    }


converters = [
    (DATASETS_DIR / "adversarial_defense.jsonl", _defense_to_row),
    (DATASETS_DIR / "reasoning_chains.jsonl",    _reasoning_to_row),
    (DATASETS_DIR / "cve_patterns.jsonl",        _cve_to_row),
    (DATASETS_DIR / "bug_bounty.jsonl",          _bounty_to_row),
]

print("\n=== Loading datasets ===")
rows = []
for path, fn in converters:
    count_before = len(rows)
    for rec in _iter_jsonl(path):
        row = fn(rec)
        if row:
            rows.append(row)
    print(f"  {path.stem:<25} {len(rows)-count_before:>6} rows")

print(f"\n  Total: {len(rows)} training examples")

# Build HF Dataset
instructions = [r["instruction"] for r in rows]
responses    = [r["response"]    for r in rows]
domains      = [r["domain"]      for r in rows]
systems      = [r["system"]      for r in rows]

ds = Dataset.from_dict({
    "instruction": instructions,
    "response":    responses,
    "domain":      domains,
    "system":      systems,
})

print(f"\n=== Pushing to {HF_REPO} (config=sft, split=train) ===")
ds.push_to_hub(
    HF_REPO,
    config_name="sft",
    split="train",
    token=HF_TOKEN,
    private=False,
)
print(f"  Pushed {len(ds)} rows → huggingface.co/datasets/{HF_REPO}")

# Also push security-only subset (no reasoning) as separate config
sec_rows = [r for r in rows if r["domain"] != "reasoning"]
sec_ds = Dataset.from_dict({
    "instruction": [r["instruction"] for r in sec_rows],
    "response":    [r["response"]    for r in sec_rows],
    "domain":      [r["domain"]      for r in sec_rows],
    "system":      [r["system"]      for r in sec_rows],
})
print(f"\n=== Pushing security-only config ({len(sec_ds)} rows) ===")
sec_ds.push_to_hub(
    HF_REPO,
    config_name="security",
    split="train",
    token=HF_TOKEN,
    private=False,
)
print(f"  Pushed → huggingface.co/datasets/{HF_REPO} (config=security)")

print("\n=== Done ===")
print(f"  Dataset: https://huggingface.co/datasets/{HF_REPO}")
print(f"  Configs: sft ({len(ds)} rows), security ({len(sec_ds)} rows)")
