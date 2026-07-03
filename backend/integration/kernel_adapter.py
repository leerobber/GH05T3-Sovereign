"""GH05T3 → sovereign-core Python Runtime adapter.

Wraps the pure-Python sovereign-core Runtime.
Expert agents extend Agent and depend on its inbox / state-machine API.

This file has NO knowledge of sovereign_core_rs (Rust).
Do not import, detect, or reference it here.
"""
from __future__ import annotations

import os
import sys

_SOVEREIGN_PATH = os.environ.get("SOVEREIGN_CORE_PATH", "")
if _SOVEREIGN_PATH and _SOVEREIGN_PATH not in sys.path:
    sys.path.insert(0, _SOVEREIGN_PATH)

from src.kernel.runtime import Runtime
from src.isa.instruction import Instruction
from src.isa.opcodes import Opcode
from src.semantics.semantic_word import SemanticWord, WordType, IntentType, ChannelType


class KernelAdapter:
    """Python sovereign-core Runtime wrapper for GH05T3 expert agents."""

    def __init__(self) -> None:
        self.runtime = Runtime()
        self._agent_roles: dict[int, str] = {}

    # ── agent lifecycle ──────────────────────────────────────────────────────

    def spawn(self, role: str) -> int:
        agent_id = self.runtime.spawn_agent()
        self._agent_roles[agent_id] = role
        return agent_id

    def kill(self, agent_id: int) -> None:
        self.runtime.kill_agent(agent_id)
        self._agent_roles.pop(agent_id, None)

    def role_of(self, agent_id: int) -> str:
        return self._agent_roles.get(agent_id, "unknown")

    # ── encoding helpers ─────────────────────────────────────────────────────

    @staticmethod
    def encode(
        intent: IntentType = IntentType.NONE,
        word_type: WordType = WordType.CONTROL,
        channel: ChannelType = ChannelType.INTERNAL,
        priority: int = 128,
        confidence: float = 1.0,
        payload_ref: int = 0,
    ) -> int:
        return SemanticWord.make(
            type=word_type,
            intent=intent,
            channel=channel,
            priority=priority,
            confidence=confidence,
            payload_ref=payload_ref,
        ).encode()

    @staticmethod
    def decode(word_int: int) -> SemanticWord:
        return SemanticWord.decode(word_int)

    # ── dispatch ─────────────────────────────────────────────────────────────

    def send(self, sender_id: int, receiver_id: int, word_int: int) -> None:
        self.runtime.route_message(sender_id, receiver_id, word_int)

    def broadcast(self, sender_id: int, word_int: int) -> None:
        self.runtime.broadcast(sender_id, word_int)

    def dispatch(self, agent_id: int, opcode: Opcode, args: list[int] | None = None) -> list[int]:
        return self.runtime.dispatch_instruction(agent_id, Instruction(opcode=opcode, args=args or []))

    # ── payload store ────────────────────────────────────────────────────────

    def store(self, obj: object) -> int:
        return self.runtime.store_payload(obj)

    def load(self, ref: int) -> object:
        return self.runtime.get_payload(ref)

    # ── observability ────────────────────────────────────────────────────────

    def add_hook(self, fn) -> None:
        self.runtime.add_hook(fn)

    def status(self) -> dict:
        s = self.runtime.status()
        s["roles"] = dict(self._agent_roles)
        return s
