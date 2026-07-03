"""
Curriculum-specific formatter.
Converts raw curriculum shards (scientist/investor/...) into ChatML ready for SFT
using the exact role+stage system prompts defined in curriculum.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

try:
    from .curriculum import get_system_prompt, Stage as CurriculumStage
except Exception:
    from curriculum import get_system_prompt, Stage as CurriculumStage  # type: ignore

Stage = CurriculumStage

DATASETS_DIR = Path(__file__).parent / "datasets"


def _iter_jsonl(p: Path | str) -> Iterator[dict]:
    path = Path(p)
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def format_curriculum_shard(role: str, stage: Stage, input_path: Path, output_path: Optional[Path] = None) -> int:
    """
    Read a raw curriculum jsonl (mixed fields) and emit ChatML using the canonical
    system prompt for (role, stage).
    """
    system = get_system_prompt(role, stage)
    count = 0
    out = []
    for rec in _iter_jsonl(input_path):
        # Flexible extraction — support several common raw shapes
        user = rec.get("question") or rec.get("scenario") or rec.get("goal") or rec.get("proposal") or rec.get("product") or rec.get("incident")
        if not user:
            continue

        # Build a rich assistant response from available fields
        parts = []
        if "reasoning_steps" in rec:
            parts.append("**Reasoning:**\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(rec["reasoning_steps"])))
        if "analysis" in rec:
            parts.append("**Analysis:** " + rec["analysis"])
        if "recommended_action" in rec or "decision" in rec:
            action = rec.get("recommended_action") or rec.get("decision")
            parts.append("**Action:** " + action)
        if "synthesis" in rec:
            parts.append("**Synthesis:** " + rec["synthesis"])
        if "final_answer" in rec:
            parts.append("**Answer:** " + rec["final_answer"])
        if "lessons" in rec:
            parts.append("**Lessons:** " + "; ".join(rec["lessons"]) if isinstance(rec["lessons"], list) else str(rec["lessons"]))
        if "rationale" in rec:
            parts.append("**Rationale:** " + rec["rationale"])

        assistant = "\n\n".join(parts) if parts else json.dumps({k: v for k, v in rec.items() if k not in ("question", "scenario")})

        out.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": str(user)},
                {"role": "assistant", "content": assistant}
            ],
            "meta": {
                "role": role,
                "stage": stage.value,
                "domain": rec.get("domain"),
                "difficulty": rec.get("difficulty"),
                "source": rec.get("source", str(input_path))
            }
        })
        count += 1

    if output_path is None:
        output_path = Path(input_path).with_suffix(".chatml.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return count


if __name__ == "__main__":
    # Example usage
    examples = [
        ("SCIENTIST", Stage.BASE, "curriculum_scientist_base.jsonl"),
        ("INVESTOR", Stage.SPECIALIST, "curriculum_investor_specialist.jsonl"),
        ("OPERATOR", Stage.SPECIALIST, "curriculum_operator_specialist.jsonl"),
        ("GOVERNOR", Stage.FRONTIER, "curriculum_governor_frontier.jsonl"),
        ("BUILDER", Stage.SPECIALIST, "curriculum_builder_specialist.jsonl"),
    ]
    for role, stage, fname in examples:
        p = DATASETS_DIR / fname
        if p.exists():
            n = format_curriculum_shard(role, stage, p)
            print(f"Formatted {n} examples -> {p.with_suffix('.chatml.jsonl').name}")
        else:
            print(f"Skip (missing): {fname}")
