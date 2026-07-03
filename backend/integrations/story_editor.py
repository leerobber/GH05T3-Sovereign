"""
GH05T3 / Avery — Developmental Story Editor
=============================================

Stateful, session-aware story editor backed by the full prompt in
prompts/story_editor.md.  Tracks the 11-step intake process per session
so the editor always knows which question to ask next.

Sessions stored in memory (dict) — reset on server restart.
Persist to MemoryPalace if you want sessions to survive restarts.

Intake stages:
    0  greeting / form+stage question
    1  premise
    2  protagonist want
    3  protagonist need
    4  antagonist force
    5  existing material
    6  diagnosis + spine build (LLM generates full output_format block)
    7+ pressure-testing individual beats (open-ended)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("gh0st3.story_editor")

# ─────────────────────────────────────────────
# PROMPT LOADING
# ─────────────────────────────────────────────

_PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "story_editor.md"

def _load_system_prompt() -> str:
    if _PROMPT_FILE.exists():
        return _PROMPT_FILE.read_text(encoding="utf-8")
    log.error("story_editor.md not found at %s", _PROMPT_FILE)
    return "You are a developmental story editor. Diagnose story structure problems."


SYSTEM_PROMPT = _load_system_prompt()

# ─────────────────────────────────────────────
# INTAKE QUESTIONS
# (used to seed the first message when a stage starts fresh)
# ─────────────────────────────────────────────

INTAKE_OPENERS = {
    0: (
        "What form is the story, and where are you in it?\n\n"
        "Examples:\n"
        "• \"Literary novel, 30,000 words drafted, stuck at the midpoint.\"\n"
        "• \"Half-hour comedy pilot, blank page, only a logline.\"\n"
        "• \"Twenty-hour RPG main quest, world locked, characters sketched, no plot pulse.\""
    ),
    1: (
        "Give me the story in one sentence. Protagonist plus their world plus the disturbance.\n\n"
        "Examples:\n"
        "• \"A widow returns to her grandmother's village in coastal Portugal and finds letters from a stranger.\"\n"
        "• \"A former competitive eater becomes a kindergarten teacher in a town obsessed with food festivals.\"\n"
        "• \"A silent courier delivers memory shards between settlements in a fungal post-apocalypse.\""
    ),
    2: (
        "What does your protagonist say they want? The visible, stateable goal.\n\n"
        "Examples:\n"
        "• \"She wants to sell the grandmother's house and leave by the end of summer.\"\n"
        "• \"He wants to keep his eating past private from his new students.\"\n"
        "• \"She wants to complete her delivery route without losing memories.\""
    ),
    3: (
        "What does your protagonist need but doesn't yet know they need? "
        "The hidden truth the story forces them toward.\n\n"
        "Examples:\n"
        "• \"She needs to grieve a relationship she never named while her grandmother was alive.\"\n"
        "• \"He needs to stop hiding the part of himself audiences once loved him for.\"\n"
        "• \"She needs to choose which memories belong to her instead of carrying everyone else's.\""
    ),
    4: (
        "What's fighting the protagonist? A person, a system, a community, "
        "the protagonist's own pattern, a ticking clock, or some combination.\n\n"
        "Examples:\n"
        "• \"Her uncle, who wants the same house and has a stronger legal claim.\"\n"
        "• \"The town's annual food festival, where his secret will surface in five episodes.\"\n"
        "• \"The fungal network itself, which rewrites memories the longer she carries shards.\""
    ),
    5: (
        "Do you have draft pages, an outline, or scene notes you want me to read against the spine?\n\n"
        "Examples:\n"
        "• \"Yes, I'll paste 12 chapter summaries.\"\n"
        "• \"Yes, here are the first 22 pages of the pilot.\"\n"
        "• \"No, only what I've told you so far.\""
    ),
}

# ─────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────

_sessions: dict[str, dict] = {}


def _new_session() -> dict:
    return {
        "stage":    0,
        "history":  [],      # list of {"role": "...", "content": "..."}
        "story":    {},      # collected intake data keyed by field name
        "created":  time.time(),
        "updated":  time.time(),
    }


def get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = _new_session()
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    _sessions[session_id] = _new_session()


def list_sessions() -> list[dict]:
    return [
        {
            "id":      sid,
            "stage":   s["stage"],
            "created": s["created"],
            "updated": s["updated"],
            "story":   s["story"],
        }
        for sid, s in _sessions.items()
    ]


# ─────────────────────────────────────────────
# STAGE FIELD NAMES (for story data collection)
# ─────────────────────────────────────────────

_STAGE_FIELDS = {
    0: "form_and_stage",
    1: "premise",
    2: "protagonist_want",
    3: "protagonist_need",
    4: "antagonist_force",
    5: "existing_material",
}


# ─────────────────────────────────────────────
# CORE TURN HANDLER
# ─────────────────────────────────────────────

async def story_editor_turn(
    session_id: str,
    user_message: str,
    llm_call,          # async callable(system, conversation_history) -> str
) -> dict:
    """
    Process one turn of the story editor conversation.

    llm_call must accept:
        system   : str   — the full system prompt
        messages : list  — [{"role": str, "content": str}, ...]
    and return str.

    Returns:
        {
            "reply":   str,   — the editor's response
            "stage":   int,   — current intake stage
            "story":   dict,  — collected story data so far
            "done":    bool,  — True once spine is delivered (stage >= 6)
        }
    """
    sess = get_session(session_id)
    stage = sess["stage"]

    # Record user turn
    sess["history"].append({"role": "user", "content": user_message})
    sess["updated"] = time.time()

    # Store intake answer for completed stages 0-5
    if stage in _STAGE_FIELDS and user_message.strip():
        sess["story"][_STAGE_FIELDS[stage]] = user_message.strip()

    # Advance stage
    next_stage = stage + 1
    sess["stage"] = next_stage

    # Build messages for LLM: full history so far
    messages = list(sess["history"])

    # If transitioning into a known intake stage, prime the next question
    # so the LLM asks it correctly rather than improvising.
    if next_stage in INTAKE_OPENERS:
        messages.append({
            "role": "assistant",
            "content": f"[NEXT QUESTION — stage {next_stage}]: {INTAKE_OPENERS[next_stage]}"
        })
        # Immediately return that question without an LLM call for speed
        reply = INTAKE_OPENERS[next_stage]
    else:
        # Stage 6+: full LLM generation (diagnosis, spine, or beat pressure-testing)
        reply = await llm_call(SYSTEM_PROMPT, messages)

    # Record assistant turn (the actual reply, not the stage hint)
    sess["history"].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "stage": next_stage,
        "story": sess["story"],
        "done":  next_stage >= 6,
    }


# ─────────────────────────────────────────────
# GREETING (call once to open a session)
# ─────────────────────────────────────────────

def story_editor_greeting() -> str:
    return INTAKE_OPENERS[0]
