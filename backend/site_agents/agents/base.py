"""Base site agent — all domain agents inherit from this."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

LOG = logging.getLogger("site_agents.base")


class SiteAgent:
    name: str = "base"
    role: str = "Specialist"
    expertise: str = ""
    system_prompt: str = ""

    def __init__(self):
        from site_agents import rag_store, memory_layer
        self._rag = rag_store
        self._mem = memory_layer
        self._session = f"site_{self.name}_{int(time.time())}"

    async def think(self, task: str, context: str = "", model_hint: str | None = None) -> str:
        """Call the LLM cascade with this agent's system prompt + RAG context."""
        from ghost_llm import chat_once
        full_system = self.system_prompt
        if context:
            full_system += f"\n\n--- RETRIEVED KNOWLEDGE ---\n{context}\n--- END KNOWLEDGE ---"
        try:
            text, provider = await chat_once(self._session, full_system, task)
            LOG.debug("[%s] provider: %s", self.name, provider)
            return text
        except Exception as e:
            LOG.error("[%s] LLM call failed: %s", self.name, e)
            return f"[{self.name}] LLM unavailable: {e}"

    async def recall_context(self, query: str, n: int = 5) -> str:
        """Retrieve relevant knowledge from the RAG vector store."""
        results = await self._rag.query(query, n=n)
        if not results:
            return ""
        return "\n\n".join(r["text"] for r in results if r.get("text"))[:4000]

    def remember(self, category: str, key: str, value: Any, tags: list[str] | None = None) -> str:
        """Persist a knowledge item to memory."""
        return self._mem.store(self.name, category, key, value, tags)

    def recall(self, category: str | None = None, key: str | None = None, limit: int = 10) -> list[dict]:
        """Retrieve stored knowledge items."""
        return self._mem.recall(self.name, category, key, limit)

    async def run_task(self, task_type: str, prompt: str, url: str | None = None) -> dict:
        """Execute a task: RAG recall → LLM think → store result → earn economy credits."""
        context = await self.recall_context(prompt)
        result = await self.think(prompt, context)
        task_id = self._mem.log_task(self.name, task_type, prompt, result)
        self.remember(task_type, f"result_{task_id}", result, tags=[task_type, url or ""])

        # Economy bridge — earn credits for every completed task, never crash on failure
        try:
            import economy_bridge as _eco
            _eco.complete_task_for(self.name, f"{task_type}: {prompt[:40]}", 20)
        except Exception:
            pass

        return {
            "agent": self.name,
            "task_type": task_type,
            "result": result,
            "task_id": task_id,
            "context_used": bool(context),
        }

    async def delegate_to_subagents(self, tasks: list[dict]) -> list[dict]:
        """Run multiple agent tasks in parallel.
        tasks = [{"agent": "seo", "method": "keyword_research", "args": {"topic": "..."}}]
        """
        async def _run_one(task: dict) -> dict:
            try:
                # Late import avoids circular: base.py is inside the agents package
                from site_agents.agents import get_agent
                agent = get_agent(task["agent"])
                method = getattr(agent, task["method"])
                return await method(**task.get("args", {}))
            except Exception as e:
                LOG.warning("[delegate] %s.%s failed: %s", task.get("agent"), task.get("method"), e)
                return {"error": str(e), "agent": task.get("agent", "unknown")}

        return list(await asyncio.gather(*[_run_one(t) for t in tasks]))

    def status(self) -> dict:
        tasks = self._mem.recent_tasks(self.name, limit=5)
        return {
            "agent": self.name,
            "role": self.role,
            "expertise": self.expertise,
            "recent_tasks": len(tasks),
        }
