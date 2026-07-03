"""MOE Router — maps SemanticWord intent to the right expert agent."""
from __future__ import annotations

import sys
import os

_SOVEREIGN_PATH = os.environ.get("SOVEREIGN_CORE_PATH", "")
if _SOVEREIGN_PATH and _SOVEREIGN_PATH not in sys.path:
    sys.path.insert(0, _SOVEREIGN_PATH)

from src.isa.opcodes import Opcode
from src.isa.instruction import Instruction
from src.semantics.semantic_word import SemanticWord, IntentType
from src.kernel.runtime import Runtime
from backend.integration.kernel_adapter import KernelAdapter
from backend.core.expert_registry import ExpertRegistry

# intent → (role, opcode) routing table
_INTENT_ROUTE: dict[int, tuple[str, Opcode]] = {
    int(IntentType.PLAN):      ("planner", Opcode.PLAN),
    int(IntentType.CRITIQUE):  ("critic",  Opcode.CRITIQUE),
    int(IntentType.REFLECT):   ("critic",  Opcode.REFLECT),
    int(IntentType.EXECUTE):   ("builder", Opcode.EMIT_RESULT),
    int(IntentType.EMIT):      ("builder", Opcode.EMIT_RESULT),
    int(IntentType.SUMMARIZE): ("biz",     Opcode.SUMMARIZE_MEMORY),
    int(IntentType.QUERY):     ("infra",   Opcode.RUN_WORKFLOW),
}


class MOERouter:
    """Routes a SemanticWord to the appropriate expert and returns emitted words."""

    def __init__(self) -> None:
        self._adapter = KernelAdapter()
        self._registry = ExpertRegistry()
        self._expert_ids: dict[str, int] = {}

    def load_experts(self) -> None:
        """Import and register all built-in expert classes."""
        from backend.experts.planner_agent import PlannerAgent
        from backend.experts.critic_agent import CriticAgent
        from backend.experts.builder_agent import BuilderAgent
        from backend.experts.biz_agent import BizAgent
        from backend.experts.infra_agent import InfraAgent

        experts = {
            "planner": PlannerAgent,
            "critic":  CriticAgent,
            "builder": BuilderAgent,
            "biz":     BizAgent,
            "infra":   InfraAgent,
        }
        for role, cls in experts.items():
            self._registry.register(role, cls)
            agent_id = self._adapter.spawn(role)
            # Replace the default Agent instance with our typed subclass.
            self._adapter.runtime.agents[agent_id] = cls(id=agent_id)
            self._expert_ids[role] = agent_id

    def route(self, word_int: int) -> list[int]:
        """Dispatch word_int to the matching expert; return emitted words."""
        sw = SemanticWord.decode(word_int)
        intent = sw.intent

        role, opcode = _INTENT_ROUTE.get(intent, ("planner", Opcode.PLAN))
        agent_id = self._expert_ids.get(role)
        if agent_id is None:
            raise RuntimeError(f"Expert '{role}' not loaded — call load_experts() first")

        self._adapter.send(0, agent_id, word_int)
        return self._adapter.dispatch(agent_id, opcode)

    def status(self) -> dict:
        return self._adapter.status()
