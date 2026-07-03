"""BuilderAgent — sovereign-core expert that handles EXECUTE/EMIT instructions."""
from __future__ import annotations

from src.agents.base_agent import Agent
from src.agents.agent_state import AgentState
from src.isa.opcodes import Opcode
from src.isa.instruction import Instruction
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


class BuilderAgent(Agent):
    """Executes plans and emits results."""

    def step(self, instruction: Instruction) -> list[int]:
        if instruction.opcode == Opcode.EMIT_RESULT:
            self.state = AgentState.PROCESSING
            self.log.append(("EMIT_RESULT", list(self.inbox)))

            results = list(self.inbox)
            self.inbox.clear()
            self.state = AgentState.IDLE
            return results or [
                SemanticWord.make(
                    type=WordType.RESULT,
                    intent=IntentType.EMIT,
                    channel=ChannelType.INTERNAL,
                    priority=200,
                    confidence=0.95,
                ).encode()
            ]

        return super().step(instruction)
