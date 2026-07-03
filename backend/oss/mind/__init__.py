"""OmniMind v1.5 — collective intelligence layer (Phase 3)."""

from .consensus import Vote, WeightedConsensusEngine
from .canonical_memory import CanonicalMemory, CanonicalMemorySystem
from .goal_generator_v2 import GoalGeneratorV2
from .swarm_reasoning import SwarmTask, SwarmReasoningEngine
from .knowledge_graph import KnowledgeNode, KnowledgeGraph

__all__ = [
    "Vote",
    "WeightedConsensusEngine",
    "CanonicalMemory",
    "CanonicalMemorySystem",
    "GoalGeneratorV2",
    "SwarmTask",
    "SwarmReasoningEngine",
    "KnowledgeNode",
    "KnowledgeGraph",
]