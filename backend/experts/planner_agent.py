"""PlannerAgent — sovereign-core expert that handles PLAN instructions."""
from __future__ import annotations

from src.agents.base_agent import Agent
from src.agents.agent_state import AgentState
from src.isa.opcodes import Opcode
from src.isa.instruction import Instruction
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


class PlannerAgent(Agent):
    """Generates structured plans from PLAN instructions."""

    def step(self, instruction: Instruction) -> list[int]:
        if instruction.opcode == Opcode.PLAN:
            self.state = AgentState.PROCESSING
            self.log.append(("PLAN", list(self.inbox)))

            plan_word = SemanticWord.make(
                type=WordType.RESULT,
                intent=IntentType.PLAN,
                channel=ChannelType.INTERNAL,
                priority=200,
                confidence=0.9,
            ).encode()
            self.inbox.clear()
            self.state = AgentState.IDLE
            return [plan_word]

        return super().step(instruction)
