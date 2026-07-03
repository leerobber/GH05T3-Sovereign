"""
Knowledge Graph v1 — Phase 3.5

Lightweight graph for theory lab and canonical memory traversal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import uuid


@dataclass
class KnowledgeNode:
    node_id: str
    label: str
    node_type: str = "concept"
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """Adjacency-list knowledge graph (no networkx required for CI)."""

    def __init__(self) -> None:
        self._nodes: Dict[str, KnowledgeNode] = {}
        self._edges: Dict[str, Set[str]] = {}
        self._edge_labels: Dict[tuple, str] = {}

    def add_node(self, label: str, node_type: str = "concept", properties: Optional[Dict[str, Any]] = None) -> str:
        nid = f"kn_{uuid.uuid4().hex[:8]}"
        self._nodes[nid] = KnowledgeNode(node_id=nid, label=label, node_type=node_type, properties=properties or {})
        self._edges.setdefault(nid, set())
        return nid

    def add_edge(self, source_id: str, target_id: str, relation: str = "relates_to") -> bool:
        if source_id not in self._nodes or target_id not in self._nodes:
            return False
        self._edges.setdefault(source_id, set()).add(target_id)
        self._edge_labels[(source_id, target_id)] = relation
        return True

    def query(self, term: str, limit: int = 20) -> List[KnowledgeNode]:
        t = term.lower()
        hits = [n for n in self._nodes.values() if t in n.label.lower() or t in n.node_type.lower()]
        return hits[:limit]

    def get_related_nodes(self, node_id: str, depth: int = 1) -> List[KnowledgeNode]:
        if node_id not in self._nodes:
            return []
        seen = {node_id}
        frontier = {node_id}
        related: List[KnowledgeNode] = []
        for _ in range(depth):
            nxt: Set[str] = set()
            for nid in frontier:
                for tgt in self._edges.get(nid, set()):
                    if tgt not in seen:
                        seen.add(tgt)
                        nxt.add(tgt)
                        if tgt in self._nodes:
                            related.append(self._nodes[tgt])
            frontier = nxt
        return related

    def sync_from_canonical(self, memories: List[Dict[str, Any]]) -> int:
        count = 0
        for mem in memories:
            label = mem.get("content", "")[:80] or mem.get("domain", "memory")
            nid = self.add_node(label, node_type="canonical", properties=mem)
            domain = mem.get("domain", "general")
            domain_nodes = self.query(domain, limit=1)
            if domain_nodes:
                self.add_edge(domain_nodes[0].node_id, nid, relation="contains")
            count += 1
        return count

    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self._nodes), "edges": sum(len(v) for v in self._edges.values())}