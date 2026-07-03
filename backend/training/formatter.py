"""Convert GH05T3 training datasets → ChatML instruction pairs for SFT."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

DATASETS_DIR = Path(__file__).parent / "datasets"

SYSTEM_PROMPT = (
    "You are GH05T3, an autonomous security and reasoning agent. "
    "You think carefully, reason step-by-step, and always prioritize "
    "detection and defense over exploitation."
)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def _defense_to_chatml(rec: dict) -> dict | None:
    threat = rec.get("threat_vector", "")
    if not threat:
        return None
    response = (
        f"**Exploitation method:** {rec.get('exploitation_method', 'N/A')}\n\n"
        f"**Detection patterns:** {rec.get('detection_pattern', 'N/A')}\n\n"
        f"**Mitigation strategy:** {rec.get('mitigation_strategy', 'N/A')}"
    )
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Analyze this threat and provide detection and mitigation guidance:\n\n{threat}"},
            {"role": "assistant", "content": response},
        ]
    }


def _reasoning_to_chatml(rec: dict) -> dict | None:
    question = rec.get("question", "")
    steps    = rec.get("reasoning_steps", [])
    if not question or not isinstance(steps, list):
        return None
    steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
    response = (
        f"**Reasoning:**\n{steps_text}\n\n"
        f"**Synthesis:** {rec.get('synthesis', 'N/A')}\n\n"
        f"**Answer:** {rec.get('final_answer', 'N/A')}"
    )
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": response},
        ]
    }


def _cve_to_chatml(rec: dict) -> dict | None:
    pattern = rec.get("vulnerability_pattern", "")
    cve_id  = rec.get("source_cve", "Unknown CVE")
    if not pattern:
        return None
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
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Analyze the vulnerability pattern for {cve_id} and explain detection and defense."},
            {"role": "assistant", "content": response},
        ]
    }


def _bounty_to_chatml(rec: dict) -> dict | None:
    target = rec.get("target_system", "")
    vuln   = rec.get("vulnerability_found", "")
    if not target or not vuln:
        return None
    response = (
        f"**Recon method:** {rec.get('recon_method', 'N/A')}\n\n"
        f"**Vulnerability:** {vuln}\n\n"
        f"**Non-weaponized PoC:** {rec.get('non_weaponized_poc', 'N/A')}\n\n"
        f"**Impact:** {rec.get('impact_assessment', 'N/A')}\n\n"
        f"**Remediation:** {rec.get('remediation', 'N/A')}"
    )
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"As a security researcher testing a {target}, describe how to responsibly discover and report a {vuln}."},
            {"role": "assistant", "content": response},
        ]
    }


def load_all_as_chatml(max_per_dataset: int | None = None,
                       seed: int = 42) -> list[dict]:
    """Load all 4 datasets, convert to ChatML, return shuffled list."""
    converters = [
        (DATASETS_DIR / "adversarial_defense.jsonl", _defense_to_chatml),
        (DATASETS_DIR / "reasoning_chains.jsonl",    _reasoning_to_chatml),
        (DATASETS_DIR / "cve_patterns.jsonl",        _cve_to_chatml),
        (DATASETS_DIR / "bug_bounty.jsonl",          _bounty_to_chatml),
    ]
    examples: list[dict] = []
    for path, converter in converters:
        count = 0
        for rec in _iter_jsonl(path):
            if max_per_dataset and count >= max_per_dataset:
                break
            ex = converter(rec)
            if ex:
                examples.append(ex)
                count += 1
    rng = random.Random(seed)
    rng.shuffle(examples)
    return examples


def dataset_sizes() -> dict[str, int]:
    names = ["adversarial_defense", "reasoning_chains", "cve_patterns", "bug_bounty"]
    sizes: dict[str, int] = {}
    for name in names:
        path = DATASETS_DIR / f"{name}.jsonl"
        if path.exists():
            with open(path) as f:
                sizes[name] = sum(1 for line in f if line.strip())
        else:
            sizes[name] = 0
    return sizes


def to_chatml_text(example: dict) -> str:
    """Format a ChatML example dict as a flat string ready for tokenization."""
    parts = []
    for m in example["messages"]:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)
