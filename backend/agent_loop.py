"""ReAct agent loop for GH05T3.

Wraps the LLM call with a tool-use loop:
  1. Send message to LLM with tool instructions in system prompt
  2. If response contains <use_tool>...</use_tool>, execute the tool
  3. Append <observation> result back into context
  4. Repeat up to MAX_ROUNDS
  5. Return final text answer + engine tag

Works with ANY provider (Anthropic, Groq, Google, Ollama) — pure text protocol.

Triggered when:
  - Message contains task/audit/inspect/check/verify/read/build/fix keywords
  - OR caller explicitly passes force_tools=True
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Awaitable

from ghost_tools import TOOL_DESCRIPTIONS, dispatch

LOG = logging.getLogger("ghost.agent_loop")

MAX_ROUNDS = 6   # max tool calls per turn

# keywords that suggest the user wants GH05T3 to DO something, not just chat
_TASK_KEYWORDS = (
    "read ", "list ", "inspect ", "audit ", "check ", "verify ",
    "build ", "fix ", "implement ", "show me ", "what functions",
    "what does ", "what is in ", "is it stubbed", "is it complete",
    "run ", "execute ", "test ", "search for ", "find ",
    "cfo_agent", "ghost_llm", "server.py", "codebase",
    "sovereignnation", "sovereign nation",
)

_TOOL_RE = re.compile(
    r"<use_tool>\s*<name>(.*?)</name>\s*<input>(.*?)</input>\s*</use_tool>",
    re.DOTALL | re.IGNORECASE,
)

AGENT_SUFFIX = f"""

━━━ TOOL USE ━━━
You can inspect files, run code, and search memory to give accurate answers.
Only use tools when they will improve the accuracy of your answer.
{TOOL_DESCRIPTIONS}
When you have your final answer, respond normally — no tool tags.
━━━━━━━━━━━━━━━"""


def _needs_tools(message: str) -> bool:
    low = message.lower()
    return any(kw in low for kw in _TASK_KEYWORDS)


async def run(
    session: str,
    system: str,
    user: str,
    llm_fn: Callable[[str, str, str], Awaitable[tuple[str, str]]],
    db=None,
    force_tools: bool = False,
) -> tuple[str, str]:
    """Run the ReAct loop.

    Args:
        session:    session id
        system:     base system prompt
        user:       user message
        llm_fn:     async fn(session, system, user) -> (text, engine_tag)
        db:         motor db for memory search tool
        force_tools: bypass the keyword heuristic

    Returns:
        (final_text, engine_tag)
    """
    if not force_tools and not _needs_tools(user):
        # straight-through — no tool overhead for simple chat
        return await llm_fn(session, system, user)

    tool_system = system + AGENT_SUFFIX
    # accumulated context: starts as user message, grows with tool observations
    context = user
    engine_tag = "unknown"

    for round_num in range(MAX_ROUNDS):
        raw, engine_tag = await llm_fn(session, tool_system, context)

        match = _TOOL_RE.search(raw)
        if not match:
            # no tool call — this is the final answer
            # strip any stray tool tags from final answer
            clean = _TOOL_RE.sub("", raw).strip()
            return clean or raw, engine_tag

        tool_name  = match.group(1).strip()
        tool_input = match.group(2).strip()

        LOG.info("[agent_loop] round=%d tool=%s input=%s",
                 round_num + 1, tool_name, tool_input[:80])

        observation = await dispatch(tool_name, tool_input, db=db)

        # build up context with tool call + observation for next round
        context = (
            f"{context}\n\n"
            f"[Tool call: {tool_name}({tool_input[:120]})]\n"
            f"<observation>\n{observation[:4000]}\n</observation>\n\n"
            f"Continue — use what you observed to answer the original question. "
            f"Call another tool if still needed, or give your final answer."
        )

    # hit max rounds — return whatever the last LLM response was
    LOG.warning("[agent_loop] hit MAX_ROUNDS=%d for session=%s", MAX_ROUNDS, session)
    final, engine_tag = await llm_fn(session, tool_system, context + "\n\nProvide your final answer now.")
    return _TOOL_RE.sub("", final).strip(), engine_tag
