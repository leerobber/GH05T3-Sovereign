"""
Aethyro Memory Cortex — three-tier user memory for the AIOS.

Tiers:
  Core Memory    — compressed always-present facts per user (SQLite, ≤2K tokens)
  Recall Memory  — semantic vector search (ChromaDB, nomic-embed-text via Ollama)
  Archival Memory — full event log, keyword-searchable (SQLite)

Identity scopes: user_id, agent_id, session_id, org_id
Audit log: append-only SQLite with SHA-256 chain — the Privacy Guarantee.

Zero new dependencies: uses ChromaDB + Ollama already in the stack.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from embeddings import EmbedResult, embed_semantic

LOG = logging.getLogger("aethyro.memory_cortex")

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE       = Path(__file__).parent / "data"
_DB_PATH    = _BASE / "memory_cortex.db"
_CHROMA_PATH = _BASE / "user_memory_chroma"

# ── ChromaDB collection name ──────────────────────────────────────────────────
COLLECTION = "aethyro_user_memory"


# ─────────────────────────────────────────────────────────────────────────────
# DB bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    _BASE.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _init_db() -> None:
    with _conn() as c:
        # Core memory: one compressed summary per user (always in context)
        c.execute("""
            CREATE TABLE IF NOT EXISTS core_memory (
                user_id    TEXT PRIMARY KEY,
                content    TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL DEFAULT 0
            )
        """)
        # Archival memory: full event log
        c.execute("""
            CREATE TABLE IF NOT EXISTS archival_memory (
                id         TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                agent_id   TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                org_id     TEXT NOT NULL DEFAULT '',
                content    TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 0.5,
                tags       TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS arch_user ON archival_memory(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS arch_agent ON archival_memory(agent_id)")
        c.execute("CREATE INDEX IF NOT EXISTS arch_ts ON archival_memory(created_at)")
        # Audit log: append-only SHA-256 chain
        c.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                agent_id   TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                action     TEXT NOT NULL,
                summary    TEXT NOT NULL DEFAULT '',
                prev_hash  TEXT NOT NULL DEFAULT '',
                row_hash   TEXT NOT NULL,
                ts         REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS audit_user ON audit_log(user_id)")


_init_db()


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB (Recall Memory)
# ─────────────────────────────────────────────────────────────────────────────

_chroma_client = None
_chroma_col = None
_embed_dim_cache: int | None = None


def _stored_embed_dim(col) -> int | None:
    """Read embed_dim stamped on the Chroma collection metadata."""
    md = col.metadata or {}
    raw = md.get("embed_dim")
    if raw is None:
        return None
    return int(raw)


def _peek_embed_dim(col) -> int | None:
    """Infer embedding dimension from the first stored vector, if any."""
    try:
        if col.count() == 0:
            return None
        peek = col.peek(limit=1)
        embs = peek.get("embeddings")
        if embs and embs[0] is not None:
            return len(embs[0])
    except Exception:
        pass
    return None


def _collection_needs_recreate(col, expected_dim: int) -> bool:
    """True when collection vectors/metadata disagree with the active embedder."""
    stored = _stored_embed_dim(col)
    if stored is not None:
        return stored != expected_dim
    peek_dim = _peek_embed_dim(col)
    if peek_dim is not None:
        return peek_dim != expected_dim
    # Empty legacy collection without embed_dim metadata — stamp on first use.
    return col.count() == 0


def _is_dim_mismatch_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "dimension" in msg or "expecting" in msg


def _chroma():
    global _chroma_col, _chroma_client
    if _chroma_col is not None:
        return _chroma_col
    try:
        import chromadb
        _CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        _chroma_col = _chroma_client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        LOG.info("[cortex] ChromaDB recall collection ready (%d docs)", _chroma_col.count())
    except Exception as e:
        LOG.error("[cortex] ChromaDB init failed: %s", e)
        _chroma_col = None
        _chroma_client = None
    return _chroma_col


def _recreate_collection(embed_dim: int) -> None:
    """Delete and recreate the recall collection for a new embedding space."""
    global _chroma_col
    if _chroma_client is None:
        return
    old_dim = _stored_embed_dim(_chroma_col) if _chroma_col else _peek_embed_dim(_chroma_col)
    try:
        _chroma_client.delete_collection(name=COLLECTION)
    except Exception:
        pass
    _chroma_col = _chroma_client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine", "embed_dim": embed_dim},
    )
    LOG.warning(
        "[cortex] ChromaDB collection recreated: embed_dim %s -> %d (docs cleared)",
        old_dim,
        embed_dim,
    )


async def _current_embed_dim() -> int:
    global _embed_dim_cache
    if _embed_dim_cache is None:
        result = await embed_semantic("dim_probe")
        _embed_dim_cache = result.dim
    return _embed_dim_cache


async def _ensure_chroma_col():
    """Return recall collection after reconciling embed_dim with the active embedder."""
    col = _chroma()
    if col is None:
        return None
    expected = await _current_embed_dim()
    if _collection_needs_recreate(col, expected):
        _recreate_collection(expected)
    return _chroma_col


# ─────────────────────────────────────────────────────────────────────────────
# Embedding helper
# ─────────────────────────────────────────────────────────────────────────────

async def _embed(text: str) -> EmbedResult | None:
    try:
        return await embed_semantic(text[:1000])
    except Exception as e:
        LOG.debug("[cortex] embed failed: %s", e)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Audit log helpers
# ─────────────────────────────────────────────────────────────────────────────

def _last_hash(user_id: str) -> str:
    with _conn() as c:
        row = c.execute(
            "SELECT row_hash FROM audit_log WHERE user_id=? ORDER BY seq DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    return row["row_hash"] if row else "genesis"


def _audit(user_id: str, agent_id: str, session_id: str, action: str, summary: str) -> None:
    prev = _last_hash(user_id)
    ts = time.time()
    raw = f"{user_id}|{agent_id}|{session_id}|{action}|{summary}|{prev}|{ts}"
    row_hash = hashlib.sha256(raw.encode()).hexdigest()
    with _conn() as c:
        c.execute(
            """INSERT INTO audit_log
               (user_id, agent_id, session_id, action, summary, prev_hash, row_hash, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, agent_id, session_id, action, summary[:500], prev, row_hash, ts),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main MemoryCortex class
# ─────────────────────────────────────────────────────────────────────────────

class MemoryCortex:
    """
    The three-tier user memory system for Aethyro AIOS.

    Usage:
        cortex = MemoryCortex()

        # Before agent responds:
        context = await cortex.read(user_id="u123", query="business plan", agent_id="oracle")
        # Prepend context to system prompt.

        # After agent responds:
        await cortex.write(user_id="u123", content="User is building a CPA firm AI tool",
                           agent_id="oracle", session_id="s456", importance=0.8)
    """

    # ── Read: build context string for injection into agent prompt ─────────────

    async def read(
        self,
        user_id: str,
        query: str,
        agent_id: str = "",
        session_id: str = "",
        recall_k: int = 5,
    ) -> str:
        """Return a formatted memory context string ready to prepend to a system prompt."""
        parts: list[str] = []

        # 1. Core memory (always present)
        core = self._read_core(user_id)
        if core:
            parts.append(f"[user context]\n{core}")

        # 2. Recall memory (semantic search)
        recall_hits = await self._search_recall(user_id, query, agent_id, recall_k)
        if recall_hits:
            lines = "\n".join(f"- {h['content']}" for h in recall_hits)
            parts.append(f"[relevant memories]\n{lines}")

        if not parts:
            return ""

        _audit(user_id, agent_id, session_id, "memory_read", f"query={query[:80]}")
        return "\n\n".join(parts) + "\n\n"

    # ── Write: store a new memory item ────────────────────────────────────────

    async def write(
        self,
        user_id: str,
        content: str,
        agent_id: str = "",
        session_id: str = "",
        org_id: str = "",
        importance: float = 0.5,
        tags: list[str] | None = None,
        tier: str = "auto",
    ) -> str:
        """
        Store content to the appropriate memory tier.
        tier: 'core' | 'recall' | 'archival' | 'auto'
        auto: importance >= 0.8 → core update; >= 0.4 → recall; always archival
        Returns the entry ID.
        """
        eid = str(uuid.uuid4())[:16]
        now = time.time()

        # Always archive
        self._write_archival(eid, user_id, agent_id, session_id, org_id,
                             content, importance, tags or [], now)

        effective_tier = tier
        if tier == "auto":
            effective_tier = "recall" if importance >= 0.4 else "archival"

        # Write to recall (vector)
        if effective_tier in ("recall", "core"):
            await self._write_recall(eid, user_id, agent_id, session_id, content, importance, tags or [])

        # Update core if high importance
        if importance >= 0.8 or effective_tier == "core":
            self._update_core(user_id, content)

        _audit(user_id, agent_id, session_id, "memory_write",
               f"importance={importance:.2f} tier={effective_tier} content={content[:60]}")
        return eid

    # ── Update core memory (compressed always-present facts) ──────────────────

    def update_core_directly(self, user_id: str, content: str) -> None:
        """Directly set/replace the core memory for a user (max 2000 chars)."""
        self._update_core(user_id, content, replace=True)
        _audit(user_id, "", "", "core_update", content[:80])

    # ── Compact: summarize old recall entries → archival ──────────────────────

    async def compact(self, user_id: str, llm_summarize_fn=None) -> dict:
        """
        Move recall entries older than 30 days to archival-only.
        Optionally pass an async summarize function to generate a core summary.
        """
        col = await _ensure_chroma_col()
        if col is None:
            return {"compacted": 0}

        cutoff = time.time() - (30 * 86400)
        try:
            results = col.get(
                where={"$and": [{"user_id": {"$eq": user_id}},
                                {"created_at": {"$lt": cutoff}}]},
                include=["documents", "metadatas", "ids"],
            )
            ids = results.get("ids", [])
            if ids:
                col.delete(ids=ids)
                LOG.info("[cortex] compacted %d old recall entries for %s", len(ids), user_id)

            if llm_summarize_fn and ids:
                docs = results.get("documents", [])
                summary = await llm_summarize_fn("\n".join(docs[:20]))
                self._update_core(user_id, summary, replace=False)

            return {"compacted": len(ids)}
        except Exception as e:
            LOG.error("[cortex] compact failed: %s", e)
            return {"compacted": 0, "error": str(e)}

    # ── Audit log export ───────────────────────────────────────────────────────

    def export_audit(self, user_id: str, limit: int = 500) -> list[dict]:
        """Export the audit log for a user — the Privacy Guarantee proof."""
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM audit_log WHERE user_id=? ORDER BY seq DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def verify_audit_chain(self, user_id: str) -> dict:
        """Verify the SHA-256 chain integrity for a user's audit log."""
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM audit_log WHERE user_id=? ORDER BY seq ASC",
                (user_id,),
            ).fetchall()
        rows = [dict(r) for r in rows]
        if not rows:
            return {"valid": True, "entries": 0}

        broken_at = None
        prev = "genesis"
        for i, row in enumerate(rows):
            raw = (f"{row['user_id']}|{row['agent_id']}|{row['session_id']}|"
                   f"{row['action']}|{row['summary']}|{prev}|{row['ts']}")
            expected = hashlib.sha256(raw.encode()).hexdigest()
            if expected != row["row_hash"] or row["prev_hash"] != prev:
                broken_at = i
                break
            prev = row["row_hash"]

        return {
            "valid": broken_at is None,
            "entries": len(rows),
            "broken_at_seq": rows[broken_at]["seq"] if broken_at is not None else None,
        }

    def stats(self, user_id: str) -> dict:
        col = _chroma()
        recall_count = 0
        if col:
            try:
                result = col.get(where={"user_id": {"$eq": user_id}})
                recall_count = len(result.get("ids", []))
            except Exception:
                pass

        with _conn() as c:
            arch_count = c.execute(
                "SELECT COUNT(*) FROM archival_memory WHERE user_id=?", (user_id,)
            ).fetchone()[0]
            audit_count = c.execute(
                "SELECT COUNT(*) FROM audit_log WHERE user_id=?", (user_id,)
            ).fetchone()[0]
            core_row = c.execute(
                "SELECT content FROM core_memory WHERE user_id=?", (user_id,)
            ).fetchone()

        return {
            "user_id": user_id,
            "core_chars": len(core_row["content"]) if core_row else 0,
            "recall_entries": recall_count,
            "archival_entries": arch_count,
            "audit_entries": audit_count,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _read_core(self, user_id: str) -> str:
        with _conn() as c:
            row = c.execute(
                "SELECT content FROM core_memory WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["content"].strip() if row else ""

    def _update_core(self, user_id: str, new_content: str, replace: bool = False) -> None:
        with _conn() as c:
            existing = c.execute(
                "SELECT content FROM core_memory WHERE user_id=?", (user_id,)
            ).fetchone()
            if replace or not existing:
                merged = new_content[:2000]
            else:
                current = existing["content"]
                # Append new fact, trim to 2000 chars keeping newest content
                merged = (current + "\n" + new_content).strip()[-2000:]
            c.execute(
                """INSERT INTO core_memory (user_id, content, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET content=excluded.content,
                                                      updated_at=excluded.updated_at""",
                (user_id, merged, time.time()),
            )

    def _write_archival(
        self, eid: str, user_id: str, agent_id: str, session_id: str,
        org_id: str, content: str, importance: float, tags: list, ts: float,
    ) -> None:
        with _conn() as c:
            c.execute(
                """INSERT OR IGNORE INTO archival_memory
                   (id, user_id, agent_id, session_id, org_id, content, importance, tags, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (eid, user_id, agent_id, session_id, org_id,
                 content[:4000], importance, json.dumps(tags), ts),
            )

    async def _write_recall(
        self, eid: str, user_id: str, agent_id: str, session_id: str,
        content: str, importance: float, tags: list,
    ) -> None:
        col = await _ensure_chroma_col()
        if col is None:
            return
        try:
            embed_result = await _embed(content)
            meta = {
                "user_id": user_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "importance": importance,
                "tags": json.dumps(tags),
                "created_at": time.time(),
            }
            kwargs: dict = {"ids": [eid], "documents": [content[:4000]], "metadatas": [meta]}
            if embed_result:
                col_dim = _stored_embed_dim(col) or embed_result.dim
                if embed_result.dim == col_dim:
                    kwargs["embeddings"] = [embed_result.vector.tolist()]
                else:
                    LOG.warning(
                        "[cortex] skip vector upsert: embed dim %d != collection %d",
                        embed_result.dim,
                        col_dim,
                    )
            col.upsert(**kwargs)
        except Exception as e:
            LOG.error("[cortex] recall write failed: %s", e)

    async def _search_recall(
        self, user_id: str, query: str, agent_id: str, k: int
    ) -> list[dict]:
        col = await _ensure_chroma_col()
        if col is None:
            return []
        try:
            total = col.count()
            if total == 0:
                return []
            embed_result = await _embed(query)
            n = min(k, total)
            where = {"user_id": {"$eq": user_id}}
            kwargs: dict = {"n_results": n, "where": where}
            use_vectors = False
            if embed_result:
                col_dim = _stored_embed_dim(col)
                if col_dim is None or embed_result.dim == col_dim:
                    kwargs["query_embeddings"] = [embed_result.vector.tolist()]
                    use_vectors = True
                else:
                    LOG.warning(
                        "[cortex] query embed dim %d != collection %d; using query_texts",
                        embed_result.dim,
                        col_dim,
                    )
            if not use_vectors:
                kwargs["query_texts"] = [query]
            try:
                results = col.query(**kwargs)
            except Exception as e:
                if use_vectors and _is_dim_mismatch_error(e):
                    LOG.warning("[cortex] ChromaDB dim error, retrying with query_texts: %s", e)
                    kwargs.pop("query_embeddings", None)
                    kwargs["query_texts"] = [query]
                    results = col.query(**kwargs)
                else:
                    raise
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            return [
                {"content": d, "agent_id": m.get("agent_id", ""),
                 "importance": float(m.get("importance", 0.5)), "distance": dist}
                for d, m, dist in zip(docs, metas, dists)
                if dist < 0.85  # filter low-relevance results
            ]
        except Exception as e:
            LOG.debug("[cortex] recall search failed: %s", e)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_cortex: MemoryCortex | None = None


def get_cortex() -> MemoryCortex:
    global _cortex
    if _cortex is None:
        _cortex = MemoryCortex()
    return _cortex


# ─────────────────────────────────────────────────────────────────────────────
# Context injection helper — drop-in for agent prompts
# ─────────────────────────────────────────────────────────────────────────────

async def inject_memory(
    system_prompt: str,
    user_id: str,
    query: str,
    agent_id: str = "",
    session_id: str = "",
) -> str:
    """
    Prepend relevant user memory to a system prompt.
    Use this in every agent call:

        system = await inject_memory(BASE_SYSTEM_PROMPT, user_id, user_message, agent_id)
    """
    if not user_id:
        return system_prompt
    cortex = get_cortex()
    ctx = await cortex.read(user_id, query, agent_id, session_id)
    if not ctx:
        return system_prompt
    return ctx + system_prompt


async def commit_exchange(
    user_id: str,
    user_message: str,
    agent_response: str,
    agent_id: str = "",
    session_id: str = "",
    importance: float = 0.5,
) -> None:
    """
    After an agent exchange, extract and store the key fact in memory.
    Runs async — call with asyncio.create_task() to not block the response.
    """
    cortex = get_cortex()
    summary = f"[{agent_id}] User asked: {user_message[:200]} | Key insight: {agent_response[:300]}"
    await cortex.write(user_id, summary, agent_id, session_id, importance=importance)
