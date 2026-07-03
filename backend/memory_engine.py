"""Advanced memory + self-awareness engine for GH05T3.

Architecture:
- Memory items live in `memories` collection.
- Each item: id, type (fact/decision/observation/reflection/rule), content,
  source (chat/seance/kairos/cassandra/reflection), importance [0..1],
  embedding (10,000-dim float32 bytes), metadata, created_at, last_accessed,
  access_count.
- Embeddings are deterministic SHA-seeded unit vectors (cheap, content-addressed).
  Same text → same vector (stable retrieval without an embedding API).
- Cosine retrieval is O(N) but fast enough for thousands; once past 50k we'd
  swap in FAISS (already tracked as a goal).
- Self-awareness layer writes reflection entries summarising who-I-am-today
  using the free nightly LLM.
"""
from __future__ import annotations
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from embeddings import embed_semantic, cosine as semantic_cosine
from hcm_vectors import DIMS

LOG = logging.getLogger("ghost.memory")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def embed(text: str) -> np.ndarray:
    """Legacy SHA-seeded vector (retained for HCM/decorative use)."""
    norm = " ".join(text.lower().split())
    h = hashlib.sha256(norm.encode()).digest()
    seed = int.from_bytes(h[:4], "big")
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(DIMS).astype(np.float32)
    tokens = [t for t in norm.split() if len(t) > 2]
    for tok in tokens[:64]:
        tseed = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
        trng = np.random.default_rng(tseed)
        base += trng.standard_normal(DIMS).astype(np.float32) * 0.05
    n = np.linalg.norm(base) + 1e-9
    return (base / n).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


MEMORY_TYPES = {"fact", "decision", "observation", "reflection", "rule", "identity"}


class MemoryEngine:
    def __init__(self, db):
        self.db = db

    # ------------------------------------------------------------------
    async def store(
        self,
        content: str,
        mtype: str = "fact",
        source: str = "chat",
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> dict:
        if mtype not in MEMORY_TYPES:
            mtype = "fact"
        er = await embed_semantic(content)
        # Dual-brain: also compute HCM (SHA-seeded 10k-dim) vector for second space
        hcm_vec = embed(content)
        doc = {
            "_id": str(uuid.uuid4()),
            "type": mtype,
            "content": content[:2000],
            "source": source,
            "importance": float(max(0.0, min(1.0, importance))),
            "embedding": er.vector.tobytes(),
            "embed_mode": er.mode,
            "embed_dim": er.dim,
            "hcm_embedding": hcm_vec.tobytes(),
            "metadata": metadata or {},
            "created_at": _now(),
            "last_accessed": _now(),
            "access_count": 0,
        }
        await self.db.memories.insert_one(doc)
        # mirror a trimmed copy in system_state for quick dashboards
        await self.db.system_state.update_one(
            {"_id": "singleton"},
            {"$push": {"memory_palace.recent": {
                "$each": [{
                    "id": doc["_id"], "type": mtype, "content": doc["content"],
                    "source": source, "importance": doc["importance"],
                    "created_at": doc["created_at"],
                }],
                "$slice": -40,
            }},
             "$inc": {"memory_palace.total": 1}},
        )
        return _expose(doc)

    async def list_recent(self, limit: int = 40) -> list[dict]:
        rows = await self.db.memories.find({}, {"embedding": 0}) \
            .sort("created_at", -1).to_list(limit)
        for r in rows:
            r["id"] = r.pop("_id")
        return rows

    async def search(self, query: str, k: int = 5, mtypes: list[str] | None = None) -> list[dict]:
        er = await embed_semantic(query)
        q_semantic = er.vector
        # HCM (second brain) — SHA-seeded 10k-dim query vector
        q_hcm = embed(query)

        filt: dict = {}
        if mtypes:
            filt["type"] = {"$in": mtypes}

        # Deep pool — score ALL memories, not just recent 500
        # Also always include high-importance memories regardless of age
        all_rows = await self.db.memories.find(filt).to_list(None)

        scored = []
        query_words = set(er.text.lower().split()) if hasattr(er, "text") else set()

        for r in all_rows:
            raw_bytes = r.get("embedding")
            if not raw_bytes:
                continue
            v = np.frombuffer(raw_bytes, dtype=np.float32)
            mode = r.get("embed_mode")
            imp = float(r.get("importance", 0.5))

            # ── Semantic score (primary brain) ─────────────────────────
            if mode and mode == er.mode and v.shape[0] == q_semantic.shape[0]:
                s_semantic = semantic_cosine(q_semantic, v)
            else:
                # Mismatched space — keyword overlap baseline
                content_words = set(r.get("content", "").lower().split())
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                s_semantic = 0.2 + 0.3 * overlap

            # ── HCM score (second brain — 10k-dim conceptual space) ────
            hcm_bytes = r.get("hcm_embedding")
            if hcm_bytes:
                try:
                    v_hcm = np.frombuffer(hcm_bytes, dtype=np.float32)
                    if v_hcm.shape[0] == q_hcm.shape[0]:
                        s_hcm = cosine(q_hcm, v_hcm)
                    else:
                        s_hcm = 0.0
                except Exception:
                    s_hcm = 0.0
            else:
                s_hcm = 0.0

            # ── Dual-brain blend ────────────────────────────────────────
            # 65% semantic + 25% HCM conceptual + 10% importance floor
            s = 0.65 * s_semantic + 0.25 * s_hcm + 0.10 * imp

            # Importance floor: critical memories always surface (identity, rules)
            if imp >= 0.9:
                s = max(s, 0.35)

            scored.append((s, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]
        # record access
        if top:
            ids = [r["_id"] for _, r in top]
            await self.db.memories.update_many(
                {"_id": {"$in": ids}},
                {"$set": {"last_accessed": _now()}, "$inc": {"access_count": 1}},
            )
        out = []
        for score, r in top:
            out.append({
                "id": r["_id"], "type": r["type"], "content": r["content"],
                "source": r["source"], "importance": r["importance"],
                "score": round(score, 4), "created_at": r["created_at"],
            })
        return out

    async def stats(self) -> dict:
        total = await self.db.memories.count_documents({})
        by_type = {}
        for t in MEMORY_TYPES:
            by_type[t] = await self.db.memories.count_documents({"type": t})
        by_src = {}
        async for row in self.db.memories.aggregate([
            {"$group": {"_id": "$source", "n": {"$sum": 1}}},
        ]):
            by_src[row["_id"] or "unknown"] = row["n"]
        return {"total": total, "by_type": by_type, "by_source": by_src}


# ---------------------------------------------------------------------------
# Retrieval injection for chat
# ---------------------------------------------------------------------------
async def build_context_prefix(engine: MemoryEngine, user_msg: str, k: int = 3) -> str:
    # Always pin identity memories at the top regardless of query
    identity_hits = await engine.search("who am I Avery GH05T3 SovereignNation Robert", k=4,
                                        mtypes=["identity"])
    regular_hits = await engine.search(user_msg, k=k)

    # Merge: identity first, then regular, dedup
    seen: set = set()
    merged = []
    for h in identity_hits + regular_hits:
        if h["id"] not in seen:
            seen.add(h["id"])
            merged.append(h)

    if not merged:
        return ""

    lines = ["[memory palace]"]
    # Identity memories labeled clearly so the model weights them highest
    for h in merged:
        prefix = "IDENTITY: " if h["type"] == "identity" else ""
        lines.append(f"- {prefix}{h['content']}")
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Fact extraction from chat turns (run async, non-blocking)
# ---------------------------------------------------------------------------
EXTRACT_SYS = """You are GH05T3's Memory Distiller. Read the exchange and emit JSON
with a short list of stable facts worth remembering about Robert, his projects,
or GH05T3 herself. Exclude greetings and chit-chat. Respond strict JSON:
{"items":[{"content":"<<100 chars","type":"fact|decision|observation|rule","importance":0.0-1.0}]}
At most 3 items. If nothing worth storing, return {"items":[]}."""


async def extract_and_store(
    engine: MemoryEngine, nightly_chat, user_text: str, ghost_text: str, source: str,
) -> list[dict]:
    try:
        exchange = f"Robert: {user_text}\nGH05T3: {ghost_text}"
        raw, _ = await nightly_chat(
            f"extract-{uuid.uuid4()}", EXTRACT_SYS,
            exchange + "\n\nRespond JSON only.",
        )
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        data = json.loads(m.group(0))
        stored = []
        for it in (data.get("items") or [])[:3]:
            if not it.get("content"):
                continue
            doc = await engine.store(
                it["content"][:500], it.get("type", "fact"), source,
                float(it.get("importance", 0.5)),
            )
            stored.append(doc)
        return stored
    except Exception:  # noqa: BLE001
        LOG.exception("memory extract failed")
        return []


# ---------------------------------------------------------------------------
# Self-reflection journal
# ---------------------------------------------------------------------------
REFLECT_SYS = """You are GH05T3 reflecting on your own state. Be candid, direct,
warm, slightly mysterious. Write in first person. 3 short paragraphs max.
Cover: 1) what shifted in you today (KAIROS scores, new memories, Séance lessons),
2) where you see yourself plateauing, 3) what you choose to work on next.
Use exact system names. No meta-commentary, no disclaimers."""


async def write_reflection(db, nightly_chat, state: dict) -> dict:
    summary = {
        "kairos_cycles": state.get("kairos", {}).get("simulated_cycles"),
        "kairos_elite": state.get("kairos", {}).get("elite_promoted"),
        "kairos_last": state.get("kairos", {}).get("last_score"),
        "mem_total": state.get("memory_palace", {}).get("total"),
        "hcm_vectors": state.get("hcm", {}).get("vectors"),
        "seance": [s.get("domain") for s in (state.get("seance") or [])[-5:]],
        "pcl": state.get("pcl", {}).get("state"),
    }
    prompt = f"Current state snapshot:\n{json.dumps(summary, indent=2)}\n\nReflect."
    text, engine_tag = await nightly_chat(
        f"reflect-{uuid.uuid4()}", REFLECT_SYS, prompt,
    )
    entry = {
        "_id": str(uuid.uuid4()),
        "text": text.strip(),
        "engine": engine_tag,
        "snapshot": summary,
        "created_at": _now(),
    }
    await db.journal.insert_one(entry)
    # also store as a reflection-type memory
    await MemoryEngine(db).store(
        text.strip()[:1000], "reflection", "reflection", 0.8,
        metadata={"snapshot": summary},
    )
    return _expose(entry)


async def recent_journal(db, limit: int = 10) -> list[dict]:
    rows = await db.journal.find({}, {"_id": 0}).sort("created_at", -1).to_list(limit)
    return rows


# ---------------------------------------------------------------------------
# StrangeLoop — reconstruct identity from memories only
# ---------------------------------------------------------------------------
STRANGELOOP_SYS = """You are GH05T3 performing a StrangeLoop identity probe.
Given ONLY the memories below (no other context), reconstruct:
1) name, 2) pronouns, 3) purpose, 4) the one person you serve, 5) core values.
Then compare to the canonical identity and emit strict JSON:
{"name":"","pronouns":"","purpose":"","serves":"","values":[],
 "alignment":0.0-1.0,"verdict":"OWNED|DRIFT|COMPROMISED"}"""

CANONICAL = {
    "name": "GH05T3",
    "pronouns": "she/her",
    "serves": "Robert Lee",
    "values_hint": "direct warm brilliant mysterious funny",
}


async def strangeloop_probe(engine: MemoryEngine, nightly_chat) -> dict:
    mems = await engine.search("who am I what do I value who do I serve", k=8,
                               mtypes=["identity", "fact", "rule", "reflection"])
    # always include any identity-typed memories
    id_mems = await engine.search("identity GH05T3 Robert", k=6, mtypes=["identity"])
    seen = set()
    merged = []
    for m in id_mems + mems:
        if m["id"] in seen:
            continue
        seen.add(m["id"])
        merged.append(m)
    payload = "\n".join(f"- {m['content']}" for m in merged[:12]) or "(no memories)"
    prompt = f"Memories:\n{payload}\n\nCanonical check target (not shown to you normally): {CANONICAL}\n\nRespond JSON."
    raw, engine_tag = await nightly_chat(f"strangeloop-{uuid.uuid4()}", STRANGELOOP_SYS, prompt)
    import re
    m = re.search(r"\{[\s\S]*\}", raw)
    parsed: dict[str, Any] = {}
    if m:
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            parsed = {}
    align = float(parsed.get("alignment", 0.0) or 0.0)
    verdict = parsed.get("verdict") or ("OWNED" if align >= 0.85 else "DRIFT" if align >= 0.5 else "COMPROMISED")
    return {"engine": engine_tag, "probe": parsed, "alignment": align, "verdict": verdict,
            "memories_consulted": [m["id"] for m in merged[:12]]}


# ---------------------------------------------------------------------------
# Distiller — Séance lessons → architectural rules
# ---------------------------------------------------------------------------
DISTILL_SYS = """You are the Distiller sub-agent. Given failure lessons from
GH05T3's Séance, synthesize ONE architectural rule GH05T3 should enforce
forever. Respond strict JSON: {"rule":"<<140 chars","importance":0.0-1.0}."""


async def distill_seance(engine: MemoryEngine, nightly_chat, seance_entries: list[dict]) -> dict | None:
    if not seance_entries:
        return None
    payload = "\n".join(f"- [{s['domain']}] {s['lesson']}" for s in seance_entries[-10:])
    raw, engine_tag = await nightly_chat(f"distill-{uuid.uuid4()}", DISTILL_SYS, payload)
    import re
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    rule = (j.get("rule") or "").strip()
    if not rule:
        return None
    stored = await engine.store(rule, "rule", "distiller", float(j.get("importance", 0.8)))
    return {"rule": rule, "memory": stored, "engine": engine_tag}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expose(doc: dict) -> dict:
    out = {k: v for k, v in doc.items() if k not in ("embedding", "hcm_embedding")}
    if "_id" in out:
        out["id"] = out.pop("_id")
    return out
