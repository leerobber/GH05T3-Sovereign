"""Registry for sovereign-core expert agent classes."""
from __future__ import annotations

from typing import Type

_SOVEREIGN_AVAILABLE = False
try:
    from src.agents.base_agent import Agent
    _SOVEREIGN_AVAILABLE = True
except ImportError:
    Agent = object  # type: ignore[misc,assignment]


class ExpertRegistry:
    """Maps role names to Agent subclasses and instantiates them on demand."""

    def __init__(self) -> None:
        self._registry: dict[str, Type] = {}

    def register(self, role: str, cls: Type) -> None:
        self._registry[role] = cls

    def create(self, role: str, agent_id: int):
        if role not in self._registry:
            raise KeyError(f"No expert registered for role '{role}'")
        return self._registry[role](id=agent_id)

    def roles(self) -> list[str]:
        return list(self._registry.keys())

    def has(self, role: str) -> bool:
        return role in self._registry
