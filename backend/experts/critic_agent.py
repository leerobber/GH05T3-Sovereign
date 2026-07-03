"""CriticAgent — sovereign-core expert that handles CRITIQUE/REFLECT instructions."""
from __future__ import annotations

from src.agents.base_agent import Agent
from src.agents.agent_state import AgentState
from src.isa.opcodes import Opcode
from src.isa.instruction import Instruction
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


class CriticAgent(Agent):
    """Evaluates plans produced by other agents."""

    def step(self, instruction: Instruction) -> list[int]:
        if instruction.opcode == Opcode.CRITIQUE:
            self.state = AgentState.REFLECTING
            self.log.append(("CRITIQUE", list(self.inbox)))

            scores = []
            for word_int in self.inbox:
                sw = SemanticWord.decode(word_int)
                confidence = max(0.0, sw.confidence_f - 0.05)
                scores.append(
                    SemanticWord.make(
                        type=WordType.RESULT,
                        intent=IntentType.CRITIQUE,
                        channel=ChannelType.INTERNAL,
                        priority=180,
                        confidence=confidence,
                    ).encode()
                )
            self.inbox.clear()
            self.state = AgentState.IDLE
            return scores or [
                SemanticWord.make(
                    type=WordType.RESULT,
                    intent=IntentType.CRITIQUE,
                    channel=ChannelType.INTERNAL,
                    priority=180,
                    confidence=0.5,
                ).encode()
            ]

        if instruction.opcode == Opcode.REFLECT:
            self.state = AgentState.REFLECTING
            self.log.append(("REFLECT", list(self.inbox)))
            reflect_word = SemanticWord.make(
                type=WordType.RESULT,
                intent=IntentType.REFLECT,
                channel=ChannelType.INTERNAL,
                priority=150,
                confidence=0.85,
            ).encode()
            self.inbox.clear()
            self.state = AgentState.IDLE
            return [reflect_word]

        return super().step(instruction)
