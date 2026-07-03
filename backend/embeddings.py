"""Hybrid semantic embeddings for GH05T3 memory retrieval.

Priority:
  1. Local MiniLM via fastembed (384-dim, offline, ships with LOQ build)
  2. Google text-embedding-004 via user's Google AI key (768-dim)
  3. SHA-seeded deterministic fallback (384-dim, zero-cost, non-semantic)

All callers see a normalized float32 vector. Callers MUST also record the
returned `EmbedResult.mode` + `dim` so retrieval can skip vectors from a
different embedding space.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

import httpx
import numpy as np

LOG = logging.getLogger("ghost.embed")

LOCAL_DIM = 384      # MiniLM-L6-v2
GOOGLE_DIM = 768     # text-embedding-004
FALLBACK_DIM = 384

_LOCAL_MODEL = None  # fastembed.TextEmbedding | False | None (sentinel)


@dataclass
class EmbedResult:
    vector: np.ndarray
    mode: str          # "local:minilm" | "google:text-embed-004" | "sha:fallback"
    dim: int


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return (v / n).astype(np.float32)


def _sha_vector(text: str, dim: int) -> np.ndarray:
    """Deterministic unit vector. Same text → same vector. Non-semantic."""
    norm = " ".join(text.lower().split())
    h = hashlib.sha256(norm.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype(np.float32)
    tokens = [t for t in norm.split() if len(t) > 2]
    for tok in tokens[:64]:
        tseed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
        trng = np.random.default_rng(tseed)
        base += trng.standard_normal(dim).astype(np.float32) * 0.05
    return _normalize(base)


def _load_local():
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL
    try:
        from fastembed import TextEmbedding
        _LOCAL_MODEL = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        LOG.info("embeddings: local MiniLM loaded (fastembed)")
    except Exception as e:  # noqa: BLE001
        LOG.warning("embeddings: local MiniLM unavailable — %s", e)
        _LOCAL_MODEL = False
    return _LOCAL_MODEL


async def _google_embed(text: str, key: str) -> np.ndarray | None:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "text-embedding-004:embedContent?key=" + key
    )
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(url, json={"content": {"parts": [{"text": text}]}})
            r.raise_for_status()
            v = np.array(r.json()["embedding"]["values"], dtype=np.float32)
            return _normalize(v)
    except Exception as e:  # noqa: BLE001
        LOG.warning("google embedding failed: %s", e)
        return None


async def embed_semantic(text: str, google_key: str | None = None) -> EmbedResult:
    """Hybrid embed. Local MiniLM → Google → SHA fallback."""
    if not text.strip():
        return EmbedResult(_sha_vector("", FALLBACK_DIM), "sha:fallback", FALLBACK_DIM)

    local = _load_local()
    if local:
        try:
            vecs = list(local.embed([text]))
            v = np.array(vecs[0], dtype=np.float32)
            return EmbedResult(_normalize(v), "local:minilm", LOCAL_DIM)
        except Exception as e:  # noqa: BLE001
            LOG.warning("local minilm embed failed: %s", e)

    key = google_key or os.environ.get("GOOGLE_AI_KEY")
    if key:
        v = await _google_embed(text, key)
        if v is not None:
            return EmbedResult(v, "google:text-embed-004", GOOGLE_DIM)

    return EmbedResult(_sha_vector(text, FALLBACK_DIM), "sha:fallback", FALLBACK_DIM)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


async def embed_status() -> dict:
    local_ok = _load_local() is not False
    return {
        "local_loaded": bool(local_ok),
        "local_model": "sentence-transformers/all-MiniLM-L6-v2" if local_ok else None,
        "local_dim": LOCAL_DIM if local_ok else None,
        "google_env_key": bool(os.environ.get("GOOGLE_AI_KEY")),
    }
