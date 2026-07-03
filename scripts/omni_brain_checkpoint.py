#!/usr/bin/env python3
"""
Omni Brain checkpoint CLI — read/update build resume state.

Usage:
  python scripts/omni_brain_checkpoint.py                    # print status
  python scripts/omni_brain_checkpoint.py --complete P1-W1-001 --next "Migrate traits.py"
  python scripts/omni_brain_checkpoint.py --set-phase 2 --week 3
  python scripts/omni_brain_checkpoint.py --metric theory_lab_1000_cycles=true
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / ".omni-brain"
CHECKPOINT = BRAIN / "CHECKPOINT.json"
STATE = BRAIN / "STATE.md"
BUILD_LOG = BRAIN / "topics" / "build-log.md"


def _load() -> dict:
    with open(CHECKPOINT, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:
    data["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _sync_state_md(data)


def _sync_state_md(data: dict) -> None:
    lines = [
        "# Omni Brain — Build State",
        "",
        "**Resume here** when continuing the Omni-Sentient build.",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **Phase** | {data.get('current_phase')} — {data.get('current_phase_name', '')} |",
        f"| **Week** | {data.get('current_week', '?')} |",
        f"| **Checkpoint ID** | {data.get('next_action_id', '?')} |",
        f"| **Next action** | {data.get('next_action', '?')} |",
        f"| **Gate** | Phase {data.get('current_phase')} in progress |",
        "",
        "## Completed recent",
        "",
    ]
    for item in data.get("completed_recent", [])[-8:]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Resume command",
        "",
        f"`{data.get('resume_command', '')}`",
        "",
    ])
    STATE.write_text("\n".join(lines), encoding="utf-8")


def _append_log(msg: str) -> None:
    BUILD_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not BUILD_LOG.exists():
        BUILD_LOG.write_text("# Build Log\n\n", encoding="utf-8")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(BUILD_LOG, "a", encoding="utf-8") as f:
        f.write(f"- {ts} — {msg}\n")


def cmd_status(data: dict) -> None:
    print(json.dumps({
        "phase": data["current_phase"],
        "phase_name": data.get("current_phase_name"),
        "next_action_id": data.get("next_action_id"),
        "next_action": data.get("next_action"),
        "gate_status": data.get("gate_status"),
        "metrics": data.get("metrics"),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Omni Brain checkpoint")
    parser.add_argument("--complete", metavar="ID", help="Mark action ID complete")
    parser.add_argument("--next", metavar="TEXT", help="Set next action text")
    parser.add_argument("--next-id", metavar="ID", help="Set next action ID")
    parser.add_argument("--set-phase", type=int, metavar="N", help="Set current phase")
    parser.add_argument("--week", type=int, help="Set current week")
    parser.add_argument("--metric", action="append", metavar="K=V", help="Set metric (bool or string)")
    parser.add_argument("--blocker", metavar="TEXT", help="Add blocker")
    parser.add_argument("--clear-blockers", action="store_true")
    args = parser.parse_args()

    data = _load()

    if args.complete:
        data.setdefault("completed_recent", []).append(f"{args.complete}: {data.get('next_action', '')}")
        if len(data["completed_recent"]) > 20:
            data["completed_recent"] = data["completed_recent"][-20:]
        _append_log(f"Completed {args.complete}")
        if args.next:
            data["next_action"] = args.next
        if args.next_id:
            data["next_action_id"] = args.next_id
        data["master_checklist_index"] = data.get("master_checklist_index", 1) + 1

    if args.next and not args.complete:
        data["next_action"] = args.next
    if args.next_id and not args.complete:
        data["next_action_id"] = args.next_id

    if args.set_phase is not None:
        data["current_phase"] = str(args.set_phase)
        names = {
            "1": "Strengthen MVS", "2": "OmniWorld v1", "3": "OmniMind v1.5",
            "4": "OmniDNA v2", "5": "Omni-Net Alpha", "6": "Omni-Evolution v2",
            "7": "Omni-Singularity", "8": "Omni-Financial Sector",
        }
        data["current_phase_name"] = names.get(str(args.set_phase), "Unknown")
        key = f"phase_{args.set_phase}" if args.set_phase != 8 else "phase_8_financial"
        if args.set_phase == 8:
            key = "phase_8_financial"
        else:
            key = f"phase_{args.set_phase}"
        data.setdefault("gate_status", {})[key] = "in_progress"

    if args.week is not None:
        data["current_week"] = str(args.week)

    if args.metric:
        metrics = data.setdefault("metrics", {})
        for m in args.metric:
            k, _, v = m.partition("=")
            if v.lower() in ("true", "false"):
                metrics[k] = v.lower() == "true"
            else:
                metrics[k] = v

    if args.blocker:
        data.setdefault("blockers", []).append(args.blocker)
    if args.clear_blockers:
        data["blockers"] = []

    if any([args.complete, args.next, args.next_id, args.set_phase, args.week, args.metric, args.blocker, args.clear_blockers]):
        _save(data)

    cmd_status(data)


if __name__ == "__main__":
    main()