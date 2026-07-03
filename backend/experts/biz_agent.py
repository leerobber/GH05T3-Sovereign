"""BizAgent — sovereign-core expert for business/strategy reasoning."""
from __future__ import annotations

from src.agents.base_agent import Agent
from src.agents.agent_state import AgentState
from src.isa.opcodes import Opcode
from src.isa.instruction import Instruction
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


class BizAgent(Agent):
    """Handles business strategy queries and summarization."""

    def step(self, instruction: Instruction) -> list[int]:
        if instruction.opcode == Opcode.SUMMARIZE_MEMORY:
            self.state = AgentState.PROCESSING
            self.log.append(("SUMMARIZE_MEMORY", list(self.inbox)))
            summary = SemanticWord.make(
                type=WordType.MEMORY,
                intent=IntentType.SUMMARIZE,
                channel=ChannelType.INTERNAL,
                priority=160,
                confidence=0.88,
            ).encode()
            self.inbox.clear()
            self.state = AgentState.IDLE
            return [summary]

        return super().step(instruction)
