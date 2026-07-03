"""
Canonical Memory System — Phase 3.2

Promotes high-fitness artifacts for reuse across agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class CanonicalMemory:
    memory_id: str
    agent_id: str
    content: str
    fitness: float
    usage_count: int = 0
    novelty: float = 0.0
    domain: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def is_canonical(self) -> bool:
        return self.fitness >= 0.85 and self.usage_count >= 5 and self.novelty >= 0.8


class CanonicalMemorySystem:
    FITNESS_THRESHOLD = 0.85
    USAGE_THRESHOLD = 5
    NOVELTY_THRESHOLD = 0.8

    def __init__(self) -> None:
        self._memories: Dict[str, CanonicalMemory] = {}

    def promote(
        self,
        agent_id: str,
        content: str,
        fitness: float,
        novelty: float = 0.5,
        domain: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[CanonicalMemory]:
        if fitness < self.FITNESS_THRESHOLD:
            return None
        mid = f"canon_{uuid.uuid4().hex[:10]}"
        mem = CanonicalMemory(
            memory_id=mid,
            agent_id=agent_id,
            content=content[:2000],
            fitness=fitness,
            novelty=novelty,
            domain=domain,
            metadata=metadata or {},
        )
        self._memories[mid] = mem
        return mem

    def search_memories(self, query: str, domain: Optional[str] = None, limit: int = 10) -> List[CanonicalMemory]:
        q = query.lower()
        hits = []
        for mem in self._memories.values():
            if domain and mem.domain != domain:
                continue
            if q in mem.content.lower() or q in mem.domain.lower():
                hits.append(mem)
        hits.sort(key=lambda m: (m.fitness, m.usage_count), reverse=True)
        return hits[:limit]

    def update_usage(self, memory_id: str) -> bool:
        mem = self._memories.get(memory_id)
        if not mem:
            return False
        mem.usage_count += 1
        return True

    def list_canonical(self) -> List[CanonicalMemory]:
        return [m for m in self._memories.values() if m.is_canonical or m.fitness >= self.FITNESS_THRESHOLD]

    def inheritance_bundle(self, domain: str = "volatility") -> List[Dict[str, Any]]:
        """Memories new agents can read on spawn."""
        return [
            {"memory_id": m.memory_id, "content": m.content[:500], "fitness": m.fitness, "domain": m.domain}
            for m in self.list_canonical()
            if m.domain == domain or domain == "all"
        ]

    def stats(self) -> Dict[str, Any]:
        total = len(self._memories)
        canonical = len(self.list_canonical())
        return {"total": total, "canonical": canonical, "domains": list({m.domain for m in self._memories.values()})}