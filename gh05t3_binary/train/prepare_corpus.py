"""Extracts a plaintext training corpus from a local GH05T3 reasoning
dataset (backend/data/training/reasoning_chains.jsonl by default).

That file is untracked/gitignored (backend/data/ is excluded as
regenerated training data, not a durable repo artifact) -- it's real,
substantial, on-theme English text that happens to already be present on
this machine from the original backend/ copy, not something fetched over
the network. Anyone cloning this repo fresh won't have it; this script's
--source flag lets them point at whatever real text they do have, and
gh05t3_binary/train/data.txt remains the tiny fallback demo corpus for
when no real dataset is available at all.
"""
from __future__ import annotations

import argparse
import json
import os

DEFAULT_SOURCE = os.path.join(
    os.path.dirname(__file__), "..", "..", "backend", "data", "training", "reasoning_chains.jsonl"
)
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "checkpoints", "corpus.txt")
DEFAULT_FIELDS = ("question", "synthesis", "final_answer")


def extract_text_lines(source_path: str, fields=DEFAULT_FIELDS):
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in fields:
                value = obj.get(field)
                if isinstance(value, str) and value.strip():
                    yield value.strip()
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            yield item.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    args = parser.parse_args()

    if not os.path.isfile(args.source):
        raise FileNotFoundError(
            f"Corpus source not found: {args.source}\n"
            "This is real local data left over from the original backend/ copy, "
            "not something this repo ships or fetches automatically -- point "
            "--source at whatever real text corpus you have available."
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    count = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for text in extract_text_lines(args.source, args.fields):
            out_f.write(text.replace("\n", " ") + "\n")
            count += 1

    size_kb = os.path.getsize(args.out) / 1024
    print(f"Wrote {count} lines ({size_kb:.1f} KB) to {args.out}")


if __name__ == "__main__":
    main()
