"""
sft_to_dpo.py — Convert SFT mentor pairs to DPO format (no API required).

Takes instruction/response pairs from mentor_pairs.jsonl and creates
DPO chosen/rejected pairs by using the full response as chosen and
a truncated generic response as rejected.

Run:
  python sft_to_dpo.py
  python sft_to_dpo.py --min-len 300
"""
import argparse, json, sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MENTOR_FILE    = DATA / "mentor_pairs.jsonl"
BOOTSTRAP_FILE = DATA / "agents_bootstrap.jsonl"

_WEAK_SUFFIX = (
    "\n\nThis is a general response. A better answer would be more specific to "
    "SovereignNation's architecture, use structured frameworks, and provide "
    "concrete next steps rather than broad advice."
)

_GENERIC_INTROS = [
    "Here's a general approach to this question:",
    "There are several ways to address this:",
    "This is a common challenge in business development:",
    "A standard approach would involve:",
    "Generally speaking, the best practice is:",
]


def _make_rejected(response: str, idx: int) -> str:
    """Create a weaker version by taking the first sentence/paragraph."""
    intro = _GENERIC_INTROS[idx % len(_GENERIC_INTROS)]
    lines = response.strip().split('\n')
    # Take first 1-2 non-empty lines as the weak response
    short_lines = []
    for line in lines:
        cleaned = line.strip('*# ').strip()
        if len(cleaned) > 30:
            short_lines.append(cleaned)
        if len(short_lines) >= 2:
            break
    if not short_lines:
        short_lines = [response[:150].strip()]
    return intro + " " + " ".join(short_lines)[:200] + "..." + _WEAK_SUFFIX


def main(min_response_len: int):
    if not MENTOR_FILE.exists():
        print(f"ERROR: {MENTOR_FILE} not found"); return

    # Load existing DPO keys to avoid duplicates (key = (agent, prompt[:120]))
    existing_dpo = set()
    existing_all = []
    if BOOTSTRAP_FILE.exists():
        for line in BOOTSTRAP_FILE.open(encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line)
            existing_all.append(r)
            if "chosen" in r and "rejected" in r:
                p = r.get("prompt","") or r.get("instruction","")
                existing_dpo.add((r.get("agent",""), p[:120]))

    mentor_rows = []
    for line in MENTOR_FILE.open(encoding="utf-8"):
        if not line.strip(): continue
        mentor_rows.append(json.loads(line))

    print(f"Loaded {len(mentor_rows)} mentor pairs, {len(existing_dpo)} existing DPO keys")

    new_dpo = []
    for i, r in enumerate(mentor_rows):
        agent  = r.get("agent", r.get("domain", "avery"))
        inst   = r.get("instruction") or r.get("prompt") or ""
        resp   = r.get("response") or r.get("chosen") or ""

        # Skip short or already-covered
        if len(resp) < min_response_len:
            continue
        key = (agent, inst[:120])
        if key in existing_dpo:
            continue

        rejected = _make_rejected(resp, i)
        dpo_row = {
            "agent":    agent,
            "prompt":   inst,
            "chosen":   resp,
            "rejected": rejected,
            "domain":   agent,
        }
        new_dpo.append(dpo_row)
        existing_dpo.add(key)

    if not new_dpo:
        print("No new DPO pairs to add.")
        return

    all_rows = existing_all + new_dpo
    BOOTSTRAP_FILE.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n",
        encoding="utf-8",
    )
    print(f"Added {len(new_dpo)} DPO pairs → agents_bootstrap.jsonl now has {len(all_rows)} rows")

    # Show per-agent breakdown of new DPO
    by_agent = {}
    for r in new_dpo:
        a = r["agent"]
        by_agent[a] = by_agent.get(a, 0) + 1
    for a, c in sorted(by_agent.items()):
        print(f"  {a}: +{c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-len", type=int, default=200,
                    help="Min response length to include as chosen")
    args = ap.parse_args()
    main(args.min_len)
