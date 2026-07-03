"""RAG vector store for Aethyro site knowledge using ChromaDB + Ollama embeddings."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

LOG = logging.getLogger("site_agents.rag")

CHROMA_PATH = Path(__file__).parent.parent / "data" / "site_chroma"
OLLAMA_URL  = os.environ.get("OLLAMA_GATEWAY_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
COLLECTION  = "aethyro_site"

_client = None
_collection = None


def _chroma():
    global _client, _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        LOG.info("[rag] ChromaDB collection '%s' ready (%d docs)", COLLECTION, _collection.count())
        return _collection
    except Exception as e:
        LOG.error("[rag] ChromaDB init failed: %s", e)
        return None


async def embed_text(text: str) -> list[float] | None:
    """Get embedding from Ollama. Falls back to None (disables vector search)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
            )
            if r.status_code == 200:
                data = r.json()
                embeddings = data.get("embeddings") or data.get("embedding")
                if embeddings:
                    return embeddings[0] if isinstance(embeddings[0], list) else embeddings
    except Exception as e:
        LOG.debug("[rag] embed failed: %s", e)
    return None


async def upsert_page(page_id: str, text: str, metadata: dict) -> bool:
    """Store a page in the vector store."""
    col = _chroma()
    if col is None:
        return False
    try:
        embedding = await embed_text(text[:4000])
        doc_id = f"page_{page_id}"
        meta = {k: str(v)[:500] for k, v in metadata.items() if v is not None}
        meta["updated_at"] = str(time.time())

        if embedding:
            col.upsert(
                ids=[doc_id],
                documents=[text[:8000]],
                embeddings=[embedding],
                metadatas=[meta],
            )
        else:
            col.upsert(
                ids=[doc_id],
                documents=[text[:8000]],
                metadatas=[meta],
            )
        return True
    except Exception as e:
        LOG.error("[rag] upsert failed: %s", e)
        return False


async def query(q: str, n: int = 5, where: dict | None = None) -> list[dict]:
    """Semantic search over stored site knowledge."""
    col = _chroma()
    if col is None:
        return []
    try:
        embedding = await embed_text(q)
        kwargs = {"query_texts": [q], "n_results": min(n, max(1, col.count()))}
        if embedding:
            kwargs = {"query_embeddings": [embedding], "n_results": kwargs["n_results"]}
        if where:
            kwargs["where"] = where

        results = col.query(**kwargs)
        out = []
        docs   = results.get("documents", [[]])[0]
        metas  = results.get("metadatas", [[]])[0]
        dists  = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({"text": doc, "metadata": meta, "distance": dist})
        return out
    except Exception as e:
        LOG.error("[rag] query failed: %s", e)
        return []


async def store_knowledge(
    doc_id: str,
    text: str,
    category: str,
    tags: list[str] | None = None,
) -> bool:
    """Store any site knowledge (analysis, recommendations, insights)."""
    col = _chroma()
    if col is None:
        return False
    meta = {
        "category": category,
        "tags": json.dumps(tags or []),
        "stored_at": str(time.time()),
    }
    embedding = await embed_text(text[:4000])
    try:
        if embedding:
            col.upsert(ids=[doc_id], documents=[text[:8000]], embeddings=[embedding], metadatas=[meta])
        else:
            col.upsert(ids=[doc_id], documents=[text[:8000]], metadatas=[meta])
        return True
    except Exception as e:
        LOG.error("[rag] store_knowledge failed: %s", e)
        return False


def stats() -> dict:
    col = _chroma()
    if col is None:
        return {"available": False}
    return {"available": True, "total_docs": col.count(), "collection": COLLECTION}
