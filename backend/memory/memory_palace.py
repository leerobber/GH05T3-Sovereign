"""GH05T3 — Memory Palace: SQLite-backed shard storage + optional Qdrant semantic search.

Qdrant is optional — falls back to substring match when unavailable.
Config:
  QDRANT_URL   = http://localhost:6333  (default)
  QDRANT_COLLECTION = memory_palace
  OLLAMA_EMBED_URL  = http://localhost:11434  (for embeddings)
"""
from __future__ import annotations
import json
import logging
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
        LOG.debug("qdrant_client not installed — using substring recall")
    except Exception as e:
        LOG.debug("Qdrant unavailable (%s) — using substring recall", e)


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


class MemoryPalace:
    """Persistent memory store. Shards are text snippets tagged by room."""

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

        # Async Qdrant upsert runs in background — don't await here
        if _qdrant_ok:
            try:
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._qdrant_upsert(shard))
                except RuntimeError:
                    pass
            except Exception:
                pass

        return shard

    async def _qdrant_upsert(self, shard: dict):
        vec = await _embed(shard["content"])
        if not vec or not _qdrant_ok:
            return
        try:
            from qdrant_client.models import PointStruct
            _qdrant_client.upsert(
                collection_name=_COLLECTION,
                points=[PointStruct(
                    id=shard["id"],
                    vector=vec,
                    payload={"room": shard["room"], "content": shard["content"]},
                )],
            )
        except Exception as e:
            LOG.debug("qdrant upsert failed: %s", e)

    async def recall(self, query: str, room: str = None, top_k: int = 5) -> list[dict]:
        # Try semantic recall via Qdrant first
        if _qdrant_ok:
            results = await self._qdrant_recall(query, room, top_k)
            if results:
                return results

        # Fallback: substring match
        q = query.lower()
        hits = [
            s for s in self._shards
            if q in s["content"].lower() and (room is None or s["room"] == room)
        ]
        return sorted(hits, key=lambda x: x["timestamp"], reverse=True)[:top_k]

    async def _qdrant_recall(self, query: str, room: str = None,
                              top_k: int = 5) -> list[dict]:
        vec = await _embed(query)
        if not vec:
            return []
        try:
            filt = None
            if room:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                filt = Filter(must=[FieldCondition(
                    key="room", match=MatchValue(value=room))])
            hits = _qdrant_client.search(
                collection_name=_COLLECTION,
                query_vector=vec,
                limit=top_k,
                query_filter=filt,
            )
            return [{"id": h.id, "content": h.payload["content"],
                     "room": h.payload.get("room", "general"),
                     "score": h.score} for h in hits]
        except Exception as e:
            LOG.debug("qdrant recall failed: %s", e)
            return []

    def prune(self, max_shards: int = 5000) -> int:
        """Delete oldest shards so total stays at or below max_shards. Returns count removed."""
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



class MemoryPalace:
    """Persistent memory store. Shards are text snippets tagged by room."""

    def __init__(self, db_path: Path = None):
        self._db = db_path or DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._shards: list[dict] = []
        self._init_db()

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
        return shard

    async def recall(self, query: str, room: str = None, top_k: int = 5) -> list[dict]:
        q = query.lower()
        hits = [
            s for s in self._shards
            if q in s["content"].lower() and (room is None or s["room"] == room)
        ]
        return sorted(hits, key=lambda x: x["timestamp"], reverse=True)[:top_k]

    def prune(self, max_shards: int = 5000) -> int:
        """Delete oldest shards so total stays at or below max_shards. Returns count removed."""
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
        }
