"""InfraAgent — sovereign-core expert for infrastructure / tool execution."""
from __future__ import annotations

from src.agents.base_agent import Agent
from src.agents.agent_state import AgentState
from src.isa.opcodes import Opcode
from src.isa.instruction import Instruction
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


class InfraAgent(Agent):
    """Handles RUN_WORKFLOW and tool-dispatch instructions."""

    def step(self, instruction: Instruction) -> list[int]:
        if instruction.opcode == Opcode.RUN_WORKFLOW:
            self.state = AgentState.PROCESSING
            workflow_id = instruction.args[0] if instruction.args else 0
            self.log.append(("RUN_WORKFLOW", workflow_id))
            result = SemanticWord.make(
                type=WordType.TOOL,
                intent=IntentType.EXECUTE,
                channel=ChannelType.INTERNAL,
                priority=190,
                confidence=0.92,
                payload_ref=int(workflow_id) & 0xFFFF,
            ).encode()
            self.inbox.clear()
            self.state = AgentState.IDLE
            return [result]

        return super().step(instruction)
