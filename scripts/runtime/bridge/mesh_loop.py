#!/usr/bin/env python
# mesh_loop.py — Phase 2 of the Claude <-> Gemini bridge.
#
# An autonomous back-and-forth between Claude (Anthropic API) and Gemini on a
# single task. Each side sees the running transcript and the other's last reply.
# Default roles: Claude drives architecture/coding, Gemini reviews & inspects
# (incl. vision via --image). A shared transcript is written to bridge/transcripts/.
#
#   python bridge/mesh_loop.py --task "Design the Aethyro onboarding flow" --turns 6
#   python bridge/mesh_loop.py --task "Critique this dashboard" --image dash.png --turns 4
#
# The loop ends at --turns rounds, or early when both agents emit the consensus
# marker [[DONE]] on the same round.
import argparse
import datetime as dt
import sys
from pathlib import Path

from _common import TRANSCRIPT_DIR
from gemini_bridge import GeminiBridge
from claude_client import ClaudeClient

DONE = "[[DONE]]"

CLAUDE_ROLE = (
    "You are CLAUDE in a two-agent engineering mesh, paired with GEMINI. "
    "You lead on architecture, code, and concrete implementation. GEMINI reviews "
    "your work, inspects visuals, and pushes back. Read the transcript, advance the "
    "task with specific, buildable output, and respond directly to GEMINI's last "
    f"points. When you believe the task is fully resolved, end your message with {DONE}."
)
GEMINI_ROLE = (
    "You are GEMINI in a two-agent engineering mesh, paired with CLAUDE. "
    "You review CLAUDE's proposals for correctness, edge cases, and (when images are "
    "present) visual/UX issues. Be concrete and critical, but converge — don't bikeshed. "
    f"When you agree the task is fully resolved, end your message with {DONE}."
)


def _fmt(role: str, text: str) -> str:
    return f"\n### {role}\n{text}\n"


def run(task: str, turns: int, image: list[str] | None) -> Path:
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = TRANSCRIPT_DIR / f"mesh_{ts}.md"

    claude = ClaudeClient()
    gemini = GeminiBridge()

    header = f"# Mesh session — {ts}\n\n**Task:** {task}\n"
    if image:
        header += f"**Images:** {', '.join(image)}\n"
    transcript = header + "\n---\n"
    out_path.write_text(transcript, encoding="utf-8")
    print(header)

    claude_done = gemini_done = False
    for rnd in range(1, turns + 1):
        round_hdr = f"\n## Round {rnd}\n"
        transcript += round_hdr
        print(round_hdr.strip())

        # CLAUDE goes first each round
        c_prompt = (
            f"TASK: {task}\n\nTRANSCRIPT SO FAR:\n{transcript}\n\n"
            "Give your next contribution as CLAUDE."
        )
        c_text = claude.ask(CLAUDE_ROLE, c_prompt)
        block = _fmt("CLAUDE", c_text)
        transcript += block
        out_path.write_text(transcript, encoding="utf-8")
        print(block)
        claude_done = DONE in c_text

        # GEMINI responds, seeing Claude's fresh reply (+ images on round 1)
        g_prompt = (
            f"TASK: {task}\n\nTRANSCRIPT SO FAR:\n{transcript}\n\n"
            "Give your next contribution as GEMINI, responding to CLAUDE's latest message."
        )
        g_text = gemini.ask(g_prompt, images=image if rnd == 1 else None, system=GEMINI_ROLE)
        block = _fmt("GEMINI", g_text)
        transcript += block
        out_path.write_text(transcript, encoding="utf-8")
        print(block)
        gemini_done = DONE in g_text

        if claude_done and gemini_done:
            print(f"\n[mesh] Consensus reached on round {rnd}.")
            break

    # Final synthesis by Claude
    synth = claude.ask(
        "You are CLAUDE. Synthesize the mesh discussion into a single final answer: "
        "the agreed design/plan/code, plus any unresolved disagreements flagged explicitly.",
        f"TASK: {task}\n\nTRANSCRIPT:\n{transcript}",
    )
    transcript += "\n---\n\n## Final synthesis (Claude)\n" + synth + "\n"
    out_path.write_text(transcript, encoding="utf-8")
    print("\n=== FINAL SYNTHESIS ===\n" + synth)
    print(f"\n[mesh] Transcript saved: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Autonomous Claude<->Gemini mesh loop.")
    ap.add_argument("--task", required=True, help="The task both agents work on.")
    ap.add_argument("--turns", type=int, default=5, help="Max rounds (default 5).")
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="Image given to Gemini on round 1 (repeatable).")
    args = ap.parse_args()
    try:
        run(args.task, args.turns, args.image)
    except KeyboardInterrupt:
        sys.exit("\n[mesh] interrupted.")


if __name__ == "__main__":
    main()
