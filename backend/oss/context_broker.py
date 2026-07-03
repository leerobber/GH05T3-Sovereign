"""
ContextBroker — unified pre-execution context assembly (Phase 3).

Before an agent acts, assemble a ranked context packet from:
  1. ChromaDB semantic memory  (via NPU embed service at :8111)
  2. CanonicalMemorySystem     (top-fitness memories)
  3. KnowledgeGraph            (related concept nodes)

All embedding calls route through the NPU /embed_query endpoint so we
never load sentence-transformers directly in this process.

Usage:
    broker = ContextBroker()
    packet = await broker.assemble(query="analyse CVE risk", agent_id="oracle", task=task)
    # packet.entries are ranked by relevance; packet.hit_rate → fitness signal
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

LOG = logging.getLogger("ghost.context_broker")

NPU_URL    = "http://localhost:8111"
CHROMA_URL = None   # set to None → lazy-import ChromaDB client when available


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ContextEntry:
    entry_id:  str
    content:   str
    source:    str           # "chroma" | "canonical" | "knowledge_graph"
    relevance: float = 0.0
    metadata:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id":  self.entry_id,
            "content":   self.content,
            "source":    self.source,
            "relevance": self.relevance,
            "metadata":  self.metadata,
        }


@dataclass
class ContextPacket:
    query:      str
    agent_id:   str
    entries:    List[ContextEntry] = field(default_factory=list)
    hit_rate:   float = 0.0     # fraction of entries actually useful (0-1)
    maturity:   int   = 1       # snapshot of agent's context maturity level
    latency_ms: float = 0.0
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query":      self.query,
            "agent_id":   self.agent_id,
            "entries":    [e.to_dict() for e in self.entries],
            "hit_rate":   self.hit_rate,
            "maturity":   self.maturity,
            "latency_ms": self.latency_ms,
        }

    def as_text_block(self, max_entries: int = 5) -> str:
        """Format top entries as a text block for injection into agent prompts."""
        lines = [f"[Context for: {self.query}]"]
        for e in self.entries[:max_entries]:
            lines.append(f"  [{e.source}] {e.content[:200]}")
        return "\n".join(lines)


# ── Embedding helper (routes through NPU service) ─────────────────────────────

async def _embed(text: str, timeout: float = 5.0) -> Optional[List[float]]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{NPU_URL}/embed_query", json={"text": text})
            if r.status_code == 200:
                return r.json().get("embedding")
    except Exception as exc:
        LOG.debug("NPU embed failed (%s) — falling back to null embedding", exc)
    return None


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── ChromaDB accessor (lazy, graceful fallback) ───────────────────────────────

_chroma_collection = None

def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    try:
        import chromadb
        client = chromadb.Client()
        _chroma_collection = client.get_or_create_collection("omni_context")
    except Exception as exc:
        LOG.debug("ChromaDB unavailable: %s", exc)
        _chroma_collection = None
    return _chroma_collection


async def _query_chroma(query_embedding: Optional[List[float]], query_text: str, n: int = 5) -> List[ContextEntry]:
    col = _get_chroma_collection()
    if col is None:
        return []
    try:
        results = await asyncio.to_thread(
            col.query,
            query_texts=[query_text],
            n_results=min(n, col.count() or 1),
        )
        entries = []
        ids   = (results.get("ids") or [[]])[0]
        docs  = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        for eid, doc, dist in zip(ids, docs, dists):
            entries.append(ContextEntry(
                entry_id=eid,
                content=doc,
                source="chroma",
                relevance=max(0.0, 1.0 - dist),
            ))
        return entries
    except Exception as exc:
        LOG.debug("ChromaDB query error: %s", exc)
        return []


# ── Canonical memory accessor ─────────────────────────────────────────────────

async def _query_canonical(query_embedding: Optional[List[float]], n: int = 5) -> List[ContextEntry]:
    try:
        from backend.oss.mind.canonical_memory import CanonicalMemorySystem
        canon = CanonicalMemorySystem()
        stats = canon.stats()
        if stats.get("total", 0) == 0:
            return []
        memories = await asyncio.to_thread(canon.get_top, n)
        entries = []
        for mem in memories:
            content = mem.get("content") or mem.get("summary", "")
            if not content:
                continue
            rel = mem.get("fitness_score", 0.5)
            if query_embedding:
                emb = mem.get("embedding")
                if emb:
                    rel = max(rel, _cosine(query_embedding, emb))
            entries.append(ContextEntry(
                entry_id=mem.get("memory_id", str(uuid.uuid4())),
                content=content,
                source="canonical",
                relevance=round(rel, 4),
                metadata={"fitness": mem.get("fitness_score", 0.5)},
            ))
        return entries
    except Exception as exc:
        LOG.debug("canonical memory query error: %s", exc)
        return []


# ── Knowledge graph accessor ──────────────────────────────────────────────────

async def _query_knowledge_graph(query: str, n: int = 5) -> List[ContextEntry]:
    try:
        from backend.oss.mind.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        nodes = await asyncio.to_thread(kg.search, query, limit=n)
        return [
            ContextEntry(
                entry_id=node.get("node_id", str(uuid.uuid4())),
                content=node.get("content", ""),
                source="knowledge_graph",
                relevance=node.get("centrality", 0.4),
            )
            for node in nodes if node.get("content")
        ]
    except Exception as exc:
        LOG.debug("knowledge graph query error: %s", exc)
        return []


# ── ContextBroker ─────────────────────────────────────────────────────────────

class ContextBroker:
    """Assembles a ranked context packet before any agent action."""

    def __init__(self, top_k: int = 8, min_relevance: float = 0.2):
        self.top_k = top_k
        self.min_relevance = min_relevance

    async def assemble(
        self,
        query: str,
        agent_id: str,
        task: Optional[Dict[str, Any]] = None,
        maturity_level: int = 1,
    ) -> ContextPacket:
        t0 = time.time()

        # Embed query via NPU (single async call)
        q_embedding = await _embed(query)

        # Fan out to all sources concurrently
        chroma_coro  = _query_chroma(q_embedding, query, n=self.top_k)
        canon_coro   = _query_canonical(q_embedding, n=self.top_k // 2)
        kg_coro      = _query_knowledge_graph(query, n=self.top_k // 2)

        chroma_entries, canon_entries, kg_entries = await asyncio.gather(
            chroma_coro, canon_coro, kg_coro, return_exceptions=True
        )

        # Merge (ignore exceptions from individual sources)
        all_entries: List[ContextEntry] = []
        for result in (chroma_entries, canon_entries, kg_entries):
            if isinstance(result, list):
                all_entries.extend(result)

        # Re-score with query embedding if available
        if q_embedding:
            for entry in all_entries:
                if entry.source == "chroma":
                    continue  # already distance-based
                # Re-rank via embed similarity when possible
                entry_emb = await _embed(entry.content[:300])
                if entry_emb:
                    sim = _cosine(q_embedding, entry_emb)
                    entry.relevance = round(max(entry.relevance, sim), 4)

        # Filter + rank
        filtered = [e for e in all_entries if e.relevance >= self.min_relevance]
        ranked   = sorted(filtered, key=lambda e: e.relevance, reverse=True)[:self.top_k]

        # hit_rate: how much of the context is above a "useful" threshold (0.5)
        useful   = sum(1 for e in ranked if e.relevance >= 0.5)
        hit_rate = useful / len(ranked) if ranked else 0.0

        latency = (time.time() - t0) * 1000
        LOG.debug(
            "context_broker: agent=%s query=%r entries=%d hit_rate=%.2f latency=%.1fms",
            agent_id, query[:60], len(ranked), hit_rate, latency,
        )

        return ContextPacket(
            query=query,
            agent_id=agent_id,
            entries=ranked,
            hit_rate=round(hit_rate, 4),
            maturity=maturity_level,
            latency_ms=round(latency, 1),
            metadata={"task_id": (task or {}).get("task_id", "")},
        )

    async def enrich_task(
        self,
        task: Dict[str, Any],
        agent_id: str,
        maturity_level: int = 1,
    ) -> Tuple[Dict[str, Any], ContextPacket]:
        """Return a copy of task with a 'context' key injected, plus the packet."""
        prompt = task.get("prompt", task.get("task", ""))
        packet = await self.assemble(prompt, agent_id, task=task, maturity_level=maturity_level)
        enriched = dict(task)
        if packet.entries:
            enriched["context"] = packet.as_text_block()
            enriched["context_hit_rate"] = packet.hit_rate
        return enriched, packet


# ── Store new context entries into ChromaDB ───────────────────────────────────

async def store_context(content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Push a piece of synthesised knowledge into the ChromaDB context store."""
    col = _get_chroma_collection()
    if col is None:
        return False
    try:
        emb = await _embed(content)
        doc_id = str(uuid.uuid4())
        kwargs: Dict[str, Any] = {"ids": [doc_id], "documents": [content]}
        if emb:
            kwargs["embeddings"] = [emb]
        if metadata:
            kwargs["metadatas"] = [metadata]
        await asyncio.to_thread(col.add, **kwargs)
        return True
    except Exception as exc:
        LOG.debug("store_context error: %s", exc)
        return False
