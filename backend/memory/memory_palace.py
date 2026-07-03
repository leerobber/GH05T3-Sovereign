"""GH05T3 — Memory Palace: SQLite-backed shard storage + optional Qdrant semantic search.

Qdrant is optional — falls back to BM25-scored substring match when unavailable.
Config:
  QDRANT_URL            = http://localhost:6333  (default)
  QDRANT_COLLECTION     = memory_palace
  OLLAMA_EMBED_URL      = http://localhost:11434  (for embeddings)
  OLLAMA_EMBED_MODEL    = nomic-embed-text
"""
from __future__ import annotations
import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path

LOG = logging.getLogger("ghost.memory")
DB_PATH = Path(os.environ.get("MEMORY_DB_PATH", "memory/palace.db"))

_qdrant_client = None
_qdrant_ok     = False
_COLLECTION    = os.environ.get("QDRANT_COLLECTION", "memory_palace")


def _init_qdrant():
    global _qdrant_client, _qdrant_ok
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        c = QdrantClient(url=url, timeout=3)
        existing = [col.name for col in c.get_collections().collections]
        if _COLLECTION not in existing:
            c.create_collection(
                _COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        _qdrant_client = c
        _qdrant_ok = True
        LOG.info("Qdrant connected: %s", url)
    except ImportError:
        LOG.debug("qdrant_client not installed — using BM25 recall")
    except Exception as e:
        LOG.debug("Qdrant unavailable (%s) — using BM25 recall", e)


async def _embed(text: str) -> list[float] | None:
    """Get embedding from Ollama /api/embed. Returns None on failure."""
    url = os.environ.get("OLLAMA_EMBED_URL",
                         os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(f"{url}/api/embed",
                             json={"model": model, "input": text[:1000]})
            if r.status_code == 200:
                data = r.json()
                return data.get("embeddings", [[]])[0] or None
    except Exception as e:
        LOG.debug("embed failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# BM25 fallback scorer — no extra dependencies required
# ---------------------------------------------------------------------------
def _bm25_score(query_terms: set[str], words: list[str], avg_len: float = 80.0,
                k1: float = 1.5, b: float = 0.75) -> float:
    """Simplified BM25 relevance score for substring-fallback recall.

    Accepts pre-split lowercase words to avoid redundant string operations.
    """
    doc_len = len(words)
    if not words or not query_terms:
        return 0.0
    matched = query_terms & set(words)
    if not matched:
        return 0.0
    score = 0.0
    for term in matched:
        tf = words.count(term)
        idf = math.log(1 + 1.0)   # simplified: single-document IDF = log(2)
        score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_len))
    return score


class MemoryPalace:
    """Persistent memory store. Shards are text snippets tagged by room.

    Search priority:
      1. Qdrant semantic search (768-dim cosine) — when available
      2. BM25-scored word-overlap fallback         — always available
    """

    def __init__(self, db_path: Path = None):
        self._db = db_path or DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._shards: list[dict] = []
        self._init_db()
        _init_qdrant()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shards (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    room      TEXT    DEFAULT 'general',
                    content   TEXT    NOT NULL,
                    timestamp REAL    NOT NULL,
                    tags      TEXT    DEFAULT '[]'
                )
            """)
            # Indexes for efficient room + tag filtering
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shards_room ON shards(room)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_shards_ts   ON shards(timestamp)")
        self._load()

    def _load(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, room, content, timestamp, tags FROM shards ORDER BY id"
            ).fetchall()
        self._shards = [
            {"id": r[0], "room": r[1], "content": r[2],
             "timestamp": r[3], "tags": json.loads(r[4] or "[]")}
            for r in rows
        ]

    def store(self, content: str, room: str = "general", tags: list = None) -> dict:
        now = time.time()
        tags_json = json.dumps(tags or [])
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO shards (room, content, timestamp, tags) VALUES (?,?,?,?)",
                (room, content, now, tags_json),
            )
            shard_id = cur.lastrowid
        shard = {"id": shard_id, "room": room, "content": content,
                  "timestamp": now, "tags": tags or []}
        self._shards.append(shard)

        # Background Qdrant upsert — non-blocking in both async and sync callers
        if _qdrant_ok:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._qdrant_upsert(shard))
            except RuntimeError:
                # No running event loop (called from sync context) — spin up a thread
                import threading
                threading.Thread(
                    target=lambda: asyncio.run(self._qdrant_upsert(shard)),
                    daemon=True,
                ).start()
            except Exception:
                pass

        return shard

    async def _qdrant_upsert(self, shard: dict):
        vec = await _embed(shard["content"])
        if not vec or not _qdrant_ok:
            return
        try:
            from qdrant_client.models import PointStruct
            point = PointStruct(
                id=shard["id"],
                vector=vec,
                payload={"room": shard["room"], "content": shard["content"],
                         "tags": shard["tags"]},
            )
            await asyncio.to_thread(
                _qdrant_client.upsert,
                collection_name=_COLLECTION,
                points=[point],
            )
        except Exception as e:
            LOG.debug("qdrant upsert failed: %s", e)

    async def recall(self, query: str, room: str = None, top_k: int = 5,
                     tags: list = None) -> list[dict]:
        # Semantic recall via Qdrant first
        if _qdrant_ok:
            results = await self._qdrant_recall(query, room, top_k, tags)
            if results:
                return results

        # BM25-scored fallback
        return self._bm25_recall(query, room, top_k, tags)

    async def _qdrant_recall(self, query: str, room: str = None,
                              top_k: int = 5, tags: list = None) -> list[dict]:
        vec = await _embed(query)
        if not vec:
            return []
        try:
            filt = None
            conditions = []
            if room:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                conditions.append(FieldCondition(key="room", match=MatchValue(value=room)))
            if tags:
                from qdrant_client.models import Filter, FieldCondition, MatchAny
                conditions.append(FieldCondition(key="tags", match=MatchAny(any=tags)))
            if conditions:
                from qdrant_client.models import Filter
                filt = Filter(must=conditions)
            hits = await asyncio.to_thread(
                _qdrant_client.search,
                collection_name=_COLLECTION,
                query_vector=vec,
                limit=top_k,
                query_filter=filt,
            )
            return [{"id": h.id, "content": h.payload["content"],
                     "room": h.payload.get("room", "general"),
                     "tags": h.payload.get("tags", []),
                     "score": h.score} for h in hits]
        except Exception as e:
            LOG.debug("qdrant recall failed: %s", e)
            return []

    def _bm25_recall(self, query: str, room: str = None,
                     top_k: int = 5, tags: list = None) -> list[dict]:
        """BM25-scored word-overlap recall across in-memory shards."""
        query_terms = set(query.lower().split())
        stopwords = {"the", "a", "an", "is", "it", "to", "of", "in", "and", "or"}
        query_terms -= stopwords

        candidates = [
            s for s in self._shards
            if (room is None or s["room"] == room)
            and (not tags or any(t in s["tags"] for t in tags))
        ]

        if not query_terms:
            return sorted(candidates, key=lambda x: x["timestamp"], reverse=True)[:top_k]

        # Pre-split once — used both for avg_len and inside _bm25_score
        candidate_words = [(s, s["content"].lower().split()) for s in candidates]
        avg_len = (sum(len(w) for _, w in candidate_words) / len(candidates)
                   if candidates else 80.0)
        avg_len = max(avg_len, 1.0)  # guard: all-empty-content shards must not cause ZeroDivisionError

        scored = [
            (s, _bm25_score(query_terms, w, avg_len))
            for s, w in candidate_words
        ]
        scored = [(s, sc) for s, sc in scored if sc > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{**s, "score": sc} for s, sc in scored[:top_k]]

    def prune(self, max_shards: int = 5000) -> int:
        """Delete oldest shards to keep total at or below max_shards. Returns count removed."""
        total = len(self._shards)
        if total <= max_shards:
            return 0
        to_remove = total - max_shards
        oldest_ids = [s["id"] for s in self._shards[:to_remove]]
        with self._conn() as conn:
            conn.execute(
                f"DELETE FROM shards WHERE id IN ({','.join('?' * len(oldest_ids))})",
                oldest_ids,
            )
        self._shards = self._shards[to_remove:]
        return to_remove

    def stats(self) -> dict:
        rooms: dict[str, int] = {}
        for s in self._shards:
            rooms[s["room"]] = rooms.get(s["room"], 0) + 1
        return {
            "total_shards": len(self._shards),
            "rooms":        rooms,
            "db_path":      str(self._db),
            "qdrant":       _qdrant_ok,
        }
