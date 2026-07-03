"""
Contractor agent base — extends SiteAgent with Memory Cortex user-profile awareness.
All 4 contractor agents inherit from this.
"""
from __future__ import annotations
import asyncio
import logging
import sys
from pathlib import Path

from .base import SiteAgent

LOG = logging.getLogger("site_agents.contractor")

# Memory Cortex lives in backend/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from memory_cortex import get_cortex, inject_memory, commit_exchange
    _MC = True
except Exception as _e:
    _MC = False
    LOG.warning("Memory Cortex unavailable: %s", _e)


class ContractorAgent(SiteAgent):
    """
    Base for all contractor agents.
    Adds:
      - user_profile(user_id) → dict of stored biz settings
      - think_with_memory(task, user_id, session_id) → LLM response with memory context
      - remember_exchange(user_id, task, result, agent_id, session_id)
    """

    async def user_profile(self, user_id: str) -> dict:
        """Pull the user's stored business profile from Memory Cortex core memory."""
        if not _MC or not user_id:
            return {}
        try:
            cortex = get_cortex()
            core = cortex._read_core(user_id)
            if not core:
                return {}
            # Ask LLM to parse profile JSON from core memory
            # Use a fast, cheap parse — just return raw core for now
            return {"raw_profile": core}
        except Exception as e:
            LOG.debug("user_profile fetch failed: %s", e)
            return {}

    async def think_with_memory(
        self,
        task: str,
        user_id: str = "",
        session_id: str = "",
        extra_context: str = "",
    ) -> str:
        """LLM call with Memory Cortex context injected into system prompt."""
        from ghost_llm import chat_once

        system = self.system_prompt
        if extra_context:
            system += f"\n\n--- TASK CONTEXT ---\n{extra_context}\n--- END CONTEXT ---"

        if _MC and user_id:
            try:
                cortex = get_cortex()
                mem_ctx = await cortex.read(user_id, task, self.name, session_id)
                if mem_ctx:
                    system = mem_ctx + system
            except Exception as e:
                LOG.debug("memory read failed: %s", e)

        try:
            text, provider = await chat_once(self._session, system, task)
            LOG.debug("[%s] provider=%s", self.name, provider)
            return text
        except Exception as e:
            LOG.error("[%s] LLM failed: %s", self.name, e)
            return f"[{self.name}] LLM unavailable: {e}"

    async def remember_exchange(
        self,
        user_id: str,
        task: str,
        result: str,
        agent_id: str = "",
        session_id: str = "",
        importance: float = 0.6,
    ) -> None:
        """Non-blocking memory commit after a task completes."""
        if not _MC or not user_id:
            return
        try:
            await commit_exchange(
                user_id, task, result,
                agent_id=agent_id or self.name,
                session_id=session_id,
                importance=importance,
            )
        except Exception as e:
            LOG.debug("memory commit failed: %s", e)

    async def save_user_setting(
        self, user_id: str, key: str, value: str, importance: float = 0.85
    ) -> None:
        """Store a persistent user setting (rates, biz name, etc.) to core memory."""
        if not _MC or not user_id:
            return
        try:
            cortex = get_cortex()
            await cortex.write(
                user_id, f"{key}: {value}",
                agent_id=self.name, importance=importance, tier="core",
            )
        except Exception as e:
            LOG.debug("save_user_setting failed: %s", e)
