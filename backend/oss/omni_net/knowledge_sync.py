"""5.3 Knowledge Sync — Phase 5.

KnowledgeSyncEngine + SyncEvent. Wires to Phase 3 KnowledgeGraph.
Target: knowledge latency <100ms in sims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid

try:
    from backend.oss.mind.knowledge_graph import KnowledgeGraph, KnowledgeNode
except Exception:
    KnowledgeGraph = None
    KnowledgeNode = dict


@dataclass
class SyncEvent:
    event_id: str
    source_agent: str
    target: str
    payload: Dict[str, Any]
    ts: float = field(default_factory=time.time)
    latency_ms: float = 0.0


class KnowledgeSyncEngine:
    """Syncs knowledge between agents/mind/net. Low latency focus."""

    def __init__(self, kg: Optional[Any] = None):
        self.kg = kg or (KnowledgeGraph() if KnowledgeGraph else None)
        self.sync_log: List[SyncEvent] = []
        self._cache: Dict[str, Any] = {}

    def sync(self, source_agent: str, target: str, knowledge: Dict[str, Any]) -> SyncEvent:
        start = time.time()
        event = SyncEvent(
            event_id=f"sync_{uuid.uuid4().hex[:8]}",
            source_agent=source_agent,
            target=target,
            payload=knowledge,
        )

        # Wire to KG if available
        if self.kg and hasattr(self.kg, "add_node"):
            try:
                label = knowledge.get("label") or knowledge.get("concept", "synced")
                nid = self.kg.add_node(label, node_type=knowledge.get("type", "synced"), properties=knowledge)
                if "related" in knowledge:
                    for rel in knowledge.get("related", []):
                        self.kg.add_edge(nid, rel)  # best effort
            except Exception:
                pass

        # simulate transport latency
        latency = (time.time() - start) * 1000 + 2  # + base
        event.latency_ms = round(latency, 2)
        self.sync_log.append(event)
        if len(self.sync_log) > 500:
            self.sync_log = self.sync_log[-400:]

        # cache for fast pull
        self._cache[target] = knowledge
        return event

    def pull(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return [e.payload for e in self.sync_log if e.target == agent_id][-limit:]

    def avg_latency(self) -> float:
        if not self.sync_log:
            return 0.0
        return sum(e.latency_ms for e in self.sync_log[-50:]) / len(self.sync_log[-50:])

    def wire_to_graph(self, graph) -> bool:
        """Explicit wire for Phase 3 KG."""
        self.kg = graph
        return True
