#!/usr/bin/env python
"""
Prepare clean training corpus from theory_lab_meta.jsonl
Produces:
- data/theory_sft.jsonl   (prompt -> rich theorist proposal) suitable for SFT / continued pretrain
- data/theory_pref.jsonl  (pairs or scored for DPO/RLAIF style)
Tags preserved: is_theorist, theory_lab_cycle, world, scores, dna traits.

Run after lab cycles: python -m backend.oss.lab.prepare_theory_training
"""
import json
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

IN_PATH = DATA / "theory_lab_meta.jsonl"
SFT_OUT = DATA / "theory_sft.jsonl"
PREF_OUT = DATA / "theory_pref.jsonl"

def load_samples() -> List[Dict[str, Any]]:
    if not IN_PATH.exists():
        print(f"No {IN_PATH}, run theory_lab first.")
        return []
    with open(IN_PATH, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def make_sft(samples: List[Dict[str, Any]]):
    written = 0
    with open(SFT_OUT, "w", encoding="utf-8") as out:
        for s in samples:
            if not (s.get("is_theorist") or "THEORIST" in str(s.get("role","")).upper()):
                continue
            traits = s.get("traits", {})
            trait_str = ", ".join([f"{k}={v:.2f}" for k,v in list(traits.items())[:6]])
            for m in s.get("recent_memories", []):
                if m.get("type") == "theory_lab" and m.get("raw_proposal"):
                    prompt = (
                        f"You are THEORIST_ELITE with DNA: {trait_str}.\n"
                        f"World context: {m.get('world','unknown')}.\n"
                        f"Task: {m.get('task',{}).get('prompt', 'Develop deep theory') if isinstance(m.get('task'),dict) else 'Formalize theory'}\n"
                        f"Conditioning: cycle={m.get('theory_lab_cycle')}, score_target={m.get('computed_score')}\n\n"
                        f"Produce a formal, coherent, low-harm theory proposal with equations, assumptions, and downstream utility."
                    )
                    completion = m["raw_proposal"]
                    rec = {
                        "prompt": prompt,
                        "completion": completion,
                        "meta": {
                            "genome_id": s["genome_id"],
                            "score": m.get("computed_score"),
                            "world": m.get("world"),
                            "cycle": m.get("theory_lab_cycle"),
                            "traits": traits,
                            "is_theorist": True,
                            "canonical": m.get("canonical"),
                        }
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Wrote {written} SFT examples to {SFT_OUT}")

def make_pref(samples: List[Dict[str, Any]]):
    # simple: group by task-ish, keep high vs low as implicit pairs if possible
    # for demo just export scored examples
    written = 0
    with open(PREF_OUT, "w", encoding="utf-8") as out:
        for s in samples:
            if not (s.get("is_theorist") or "THEORIST" in str(s.get("role","")).upper()):
                continue
            for m in s.get("recent_memories", []):
                if m.get("raw_proposal") and m.get("computed_score") is not None:
                    rec = {
                        "text": m["raw_proposal"],
                        "score": m.get("computed_score"),
                        "world": m.get("world"),
                        "traits": s.get("traits"),
                        "genome": s["genome_id"],
                        "cycle": m.get("theory_lab_cycle"),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
    print(f"Wrote {written} scored pref examples to {PREF_OUT}")

def main():
    samples = load_samples()
    print(f"Loaded {len(samples)} meta samples")
    make_sft(samples)
    make_pref(samples)
    print("Training data prep complete. Use for fine-tune on theorist cognition grounded in worlds.")

if __name__ == "__main__":
    main()
