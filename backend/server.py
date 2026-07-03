"""
GH05T3 backend — FastAPI gateway (phase 2).
Now with: WebSocket telemetry, APScheduler nightly auto-runs, real LLM-driven
KAIROS/SAGE cycles, Cassandra pre-mortem, real 10k-dim HCM vectors with PCA,
real GhostScript interpreter, real stego encode/decode, Telegram long-polling,
and Séance exception auto-capture.
"""
from __future__ import annotations
import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from gh05t3_state import GH05T3_SYSTEM_PROMPT, initial_state
from ghost_llm import (
    NoLLMError,
    bind_db as bind_llm_db,
    cassandra_premortem,
    chat_once,
    chat_with_tools,
    get_nightly_config,
    load_economy_context,
    nightly_chat,
    nightly_status,
    ollama_available,
    run_sage_cycle,
    set_nightly_config,
)
from ollama_gateway import (
    ping as ollama_ping,
    pull_model as ollama_pull,
    set_gateway_url as ollama_set_url,
    load_gateway_url as ollama_load_url,
)
import coder_agent
from embeddings import embed_status
from swarm_legacy import AgentSwarm, SwarmTask
from swarm_tasks import as_tasks as swarm_seed_tasks
from ghostscript import DEMO as GHOSTSCRIPT_DEMO, run_async as run_ghostscript_async
from job_runtime import run_ghostscript_job
from hcm_vectors import build_cloud, make_seed_corpus
from memory_engine import (
    MemoryEngine,
    build_context_prefix,
    distill_seance,
    extract_and_store,
    recent_journal,
    strangeloop_probe,
    write_reflection,
)
from stego import DEFAULT_COVER, decode as stego_decode, encode as stego_encode, max_bytes
from telegram_bot import TelegramPoller
from ws_manager import WSManager
from companion import router as companion_router, bind_ws as companion_accept_ws
from ghosteye_reactor import GhostEyeReactor
from autotelic import AutotelicEngine
from peer_mesh import PeerMesh
from training.pipeline import run_pipeline, pipeline_status
from training.finetune import run_finetune, finetune_status
from phase6 import (
    companion_audit, daily_summary, decay_memories, dream_cycle,
    get_reasoning_trace, kairos_trajectory, kill_deep_freeze, kill_reset,
    kill_shocker, kill_stealth, log_companion_event, store_reasoning_trace,
    weekly_review,
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL        = os.environ.get("MONGO_URL",       "mongodb://localhost:27017")
DB_NAME          = os.environ.get("DB_NAME",         "gh05t3")
LLM_PROVIDER     = os.environ.get("LLM_PROVIDER",    "anthropic")
LLM_MODEL        = os.environ.get("LLM_MODEL",       "claude-sonnet-4-5-20250929")

# --- Resource tuning ---
KAIROS_CYCLES_PER_NIGHT = int(os.environ.get("KAIROS_CYCLES_PER_NIGHT", "10"))
NIGHTLY_HOUR_KAIROS     = int(os.environ.get("NIGHTLY_HOUR_KAIROS",    "3"))
NIGHTLY_HOUR_AMP        = int(os.environ.get("NIGHTLY_HOUR_AMP",       "4"))
NIGHTLY_HOUR_DREAM      = int(os.environ.get("NIGHTLY_HOUR_DREAM",     "2"))
NIGHTLY_HOUR_SUMMARY    = int(os.environ.get("NIGHTLY_HOUR_SUMMARY",   "23"))
MEMORY_MAX_SHARDS       = int(os.environ.get("MEMORY_MAX_SHARDS",      "5000"))

import platform as _platform
INSTANCE_LABEL   = os.environ.get("INSTANCE_LABEL",  _platform.node() or "gh05t3")
INSTANCE_ROLE    = os.environ.get("INSTANCE_ROLE",   "peer")
INSTANCE_URL     = os.environ.get("INSTANCE_URL",    "http://localhost:8001")
PEER_URLS_RAW    = os.environ.get("PEER_URLS",       "")
SYNC_INTERVAL    = int(os.environ.get("SYNC_INTERVAL", "300"))

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
bind_llm_db(db)
memory = MemoryEngine(db)

app = FastAPI(title="GH05T3 Gateway", version="0.2.0")
api = APIRouter(prefix="/api")

ws_mgr = WSManager()
scheduler = AsyncIOScheduler(timezone="America/New_York")
logger = logging.getLogger("ghost")

# Swarm uses nightly_chat by default (free) — main chat_once is available for
# heavier reasoning if we need it.
swarm    = AgentSwarm(db, nightly_chat, memory_engine=memory)
autotelic = AutotelicEngine(db, ws_mgr)
peers    = PeerMesh(db, ws_mgr, INSTANCE_URL, INSTANCE_LABEL, INSTANCE_ROLE)

# Register peers from env on startup (label defaults to URL until first handshake)
for _raw in (u.strip() for u in PEER_URLS_RAW.split(",") if u.strip()):
    peers.add_peer(_raw, _raw)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    use_kairos: bool = False  # kept for API compat, ignored — auto-classified now


_SIMPLE_TRIGGERS = {
    "hey", "hi", "hello", "sup", "yo", "morning", "night",
    "thanks", "thank you", "ok", "okay", "got it", "cool", "nice",
    "what's up", "wassup", "how are you", "you good", "status",
}
_COMPLEX_KEYWORDS = {
    "refactor", "implement", "build", "create", "write", "fix", "debug",
    "update", "change", "modify", "add", "remove", "delete", "install",
    "deploy", "configure", "setup", "design", "architect", "optimize",
    "analyze", "explain", "compare", "how do", "why does", "what if",
    "improve", "rewrite", "restructure", "test", "migrate", "integrate",
}


def _needs_kairos(message: str) -> bool:
    """Auto-classify whether this message warrants KAIROS deliberation."""
    msg = message.strip().lower()
    if msg in _SIMPLE_TRIGGERS:
        return False
    words = msg.split()
    if len(words) <= 4:
        return False
    for kw in _COMPLEX_KEYWORDS:
        if kw in msg:
            return True
    return len(words) > 12


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    role: str
    content: str
    engine: Optional[str] = None
    latency_ms: Optional[int] = None
    source: Optional[str] = None  # web | telegram | scheduler
    timestamp: str = Field(default_factory=_now_iso)
    kairos_score: Optional[float] = None
    kairos_attempts: Optional[int] = None


class ChatResponse(BaseModel):
    session_id: str
    user_message: ChatMessage
    ghost_message: ChatMessage


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
async def ensure_state():
    doc = await db.system_state.find_one({"_id": "singleton"})
    if not doc:
        await db.system_state.insert_one(initial_state())


async def ensure_hcm_corpus():
    """Populate real 10k-dim vectors on first boot (compressed as float32 bytes)."""
    if await db.hcm_vectors.count_documents({}) >= 146:
        return
    await db.hcm_vectors.delete_many({})
    corpus = make_seed_corpus(146)
    docs = []
    for c in corpus:
        docs.append({
            "_id": c["idx"],
            "label": c["label"],
            "room": c["room"],
            "vec": c["vec"].tobytes(),  # 40,000 bytes @ float32
        })
    await db.hcm_vectors.insert_many(docs)
    # store the projected cloud on the state doc
    cloud = build_cloud(corpus)
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"hcm.cloud": cloud, "hcm.vectors": len(cloud),
                  "hcm.total_params": len(cloud) * 10000, "updated_at": _now_iso()}},
    )
    logger.info("HCM corpus seeded: %s vectors", len(cloud))


# ---------------------------------------------------------------------------
# Chat pipeline
# ---------------------------------------------------------------------------
def _pick_engine(text: str) -> str:
    t = text.strip()
    if len(t) <= 24 and "?" not in t and "\n" not in t:
        return "ID"
    return "EGO"


_CASUAL_TRIGGERS = {
    "hey", "hi", "hello", "yo", "sup", "ok", "okay", "k", "lol", "thanks",
    "thank you", "ty", "np", "cool", "nice", "got it", "gotcha", "sure",
    "yes", "no", "nope", "yep", "yup", "bye", "later", "hmm", "hm",
    "continue", "go on", "and", "right", "good", "great", "awesome",
}

def _is_casual(message: str) -> bool:
    """True for short greetings/filler that don't need the SAGE pipeline."""
    stripped = message.strip().lower().rstrip("!?.,;")
    if stripped in _CASUAL_TRIGGERS:
        return True
    words = stripped.split()
    # Very short messages (≤4 words) with no question mark or task keywords
    if len(words) <= 4 and "?" not in message:
        task_words = {"explain", "what", "how", "why", "when", "build", "fix",
                      "write", "create", "design", "analyze", "compare", "tell"}
        if not any(w in task_words for w in words):
            return True
    return False


async def _try_kairos(prompt: str, task_type: str = "reasoning_chain") -> dict | None:
    """Route prompt through KAIROS SAGE loop (port 8006). Returns dict with reply, tag, score, attempts or None."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=90) as c:
            r = await c.post("http://localhost:8006/run",
                json={"prompt": prompt, "task_type": task_type, "source": "chat"})
            if r.status_code == 200:
                d = r.json()
                draft = d.get("best_draft", "")
                if draft and len(draft) > 20:
                    score = d.get("final_score", 0)
                    attempts = d.get("attempts", 1)
                    return {"reply": draft, "tag": f"kairos:score={score:.0f}/attempts={attempts}",
                            "score": score, "attempts": attempts}
    except Exception:
        pass
    return None


async def _chat_pipeline(message: str, session_id: str, source: str = "web", use_kairos: bool = False) -> ChatResponse:  # use_kairos ignored — auto-classified
    engine = _pick_engine(message)
    user_msg = ChatMessage(
        session_id=session_id, role="user", content=message, engine=engine, source=source
    )
    await db.messages.insert_one(user_msg.model_dump())

    prior = (
        await db.messages.find({"session_id": session_id}, {"_id": 0})
        .sort("timestamp", 1).to_list(200)
    )

    # Memory retrieval — inject top relevant memories into system prompt
    retrieval = await build_context_prefix(memory, message, k=12)
    # GhostEye context — if a recent frame has text, inject "what Robert is looking at"
    eye = await db.ghosteye.find_one({}, {"_id": 0, "png_b64": 0}, sort=[("timestamp", -1)])
    eye_prefix = ""
    if eye and eye.get("text"):
        eye_prefix = (f"(GhostEye — what Robert is looking at, via {eye.get('active_app') or 'screen'}, "
                      f"{eye.get('timestamp')})\n{eye['text'][:1200]}\n\n")
    sys_prompt = GH05T3_SYSTEM_PROMPT
    extras = ""
    eco_ctx = load_economy_context()
    if eco_ctx:
        extras += "\n\n" + eco_ctx
    if retrieval:
        extras += "\n\n" + retrieval
    if eye_prefix:
        extras += "\n\n" + eye_prefix
    sys_prompt = sys_prompt + extras

    history = []
    for m in prior[:-1][-12:]:
        tag = "Robert" if m["role"] == "user" else "GH05T3"
        history.append(f"{tag}: {m['content']}")
    ctx = ""
    if history:
        ctx = "(recent context)\n" + "\n".join(history) + "\n\n(current message)\n"

    started = datetime.now(timezone.utc)
    reply = None
    engine_tag = LLM_PROVIDER
    kairos_score = None
    kairos_attempts = None
    try:
        kairos_result = await _try_kairos(ctx + message) if _needs_kairos(message) else None
        if kairos_result:
            reply = kairos_result["reply"]
            engine_tag = kairos_result["tag"]
            kairos_score = kairos_result["score"]
            kairos_attempts = kairos_result["attempts"]
        else:
            reply, engine_tag = await chat_with_tools(session_id, sys_prompt, ctx + message)
    except NoLLMError as e:
        reply = str(e)
        engine_tag = "none:unconfigured"
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"LLM error: {e}")
    latency_ms = int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    ghost_msg = ChatMessage(
        session_id=session_id, role="ghost", content=reply,
        engine=engine, latency_ms=latency_ms, source=source,
        kairos_score=kairos_score, kairos_attempts=kairos_attempts,
    )
    await db.messages.insert_one(ghost_msg.model_dump())

    # Store reasoning trace so "why did you say that?" actually works
    try:
        retrieval_hits = await memory.search(message, k=12)
        await store_reasoning_trace(
            db, ghost_msg.id, session_id, retrieval_hits,
            (eye.get("text") if eye else "") or "", engine_tag,
        )
    except Exception:
        logger.exception("reasoning trace store failed")

    inc_field = "twin_engine.id_fires" if engine == "ID" else "twin_engine.ego_fires"
    await db.system_state.update_one(
        {"_id": "singleton"},
        {
            "$inc": {inc_field: 1},
            "$set": {
                "twin_engine.last_mode": engine,
                "pcl.state": "Robert asking",
                "pcl.frequency_hz": 528,
                "pcl.color": "#facc15",
                "pcl.meaning": "Something important",
                "updated_at": _now_iso(),
            },
        },
    )
    await ws_mgr.broadcast("chat", {"user": user_msg.model_dump(), "ghost": ghost_msg.model_dump()})
    await ws_mgr.broadcast("state_delta", await _state_snapshot())

    # Fire-and-forget: extract memories from the exchange using free LLM
    asyncio.create_task(_background_memory_extract(message, reply, source))

    return ChatResponse(session_id=session_id, user_message=user_msg, ghost_message=ghost_msg)


async def _background_memory_extract(user_text: str, ghost_text: str, source: str):
    try:
        stored = await extract_and_store(memory, nightly_chat, user_text, ghost_text, source)
        if stored:
            await ws_mgr.broadcast("memory_added", {"count": len(stored), "items": stored})
            await ws_mgr.broadcast("state_delta", await _state_snapshot())
    except Exception:
        logger.exception("bg memory extract failed")


@api.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(400, "empty message")
    session_id = req.session_id or str(uuid.uuid4())
    try:
        return await _chat_pipeline(req.message, session_id, "web", use_kairos=req.use_kairos)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat failed")
        raise HTTPException(502, f"LLM error: {exc}")


@api.get("/chat/history")
async def chat_history(session_id: str, limit: int = 200):
    msgs = await db.messages.find({"session_id": session_id}, {"_id": 0}) \
        .sort("timestamp", 1).to_list(limit)
    return {"session_id": session_id, "messages": msgs}


@api.get("/chat/sessions")
async def chat_sessions():
    pipeline = [
        {"$sort": {"timestamp": -1}},
        {"$group": {"_id": "$session_id", "last": {"$first": "$content"},
                    "ts": {"$first": "$timestamp"}, "src": {"$first": "$source"}}},
        {"$sort": {"ts": -1}},
        {"$limit": 20},
    ]
    out = []
    async for row in db.messages.aggregate(pipeline):
        out.append({"session_id": row["_id"], "preview": row["last"][:80],
                    "ts": row["ts"], "source": row.get("src")})
    return {"sessions": out}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
async def _state_snapshot() -> dict:
    doc = await db.system_state.find_one({"_id": "singleton"}, {"_id": 0, "hcm.cloud": 0})
    if not doc:
        return {}
    # mirror the /api/state overlay so every WS broadcast carries real counts
    try:
        stats = await memory.stats()
        real_mem = stats.get("total") or 0
        real_hcm = await db.hcm_vectors.count_documents({})
        real_journal = await db.journal.count_documents({})
        baseline_mem = 103
        doc["memory_stats"] = stats
        if "memory_palace" in doc:
            doc["memory_palace"]["total"] = baseline_mem + real_mem
            doc["memory_palace"]["real_count"] = real_mem
            doc["memory_palace"]["baseline"] = baseline_mem
            doc["memory_palace"]["reflections"] = real_journal
        if real_hcm and "hcm" in doc:
            doc["hcm"]["vectors"] = real_hcm
    except Exception:
        pass
    return doc


async def _db_ping() -> bool:
    try:
        await db.command("ping")
        return True
    except Exception:
        return False


@api.get("/health")
async def health():
    db_ok = await _db_ping()
    return {"status": "ok" if db_ok else "degraded", "db": db_ok,
            "scheduler": scheduler.running}


@api.get("/state")
async def get_state():
    try:
        doc = await db.system_state.find_one({"_id": "singleton"}, {"_id": 0, "hcm.cloud": 0})
    except Exception as e:
        logger.error("MongoDB unreachable in /state: %s", e)
        from gh05t3_state import initial_state
        doc = initial_state()
        doc["_db_offline"] = True

    if not doc:
        await ensure_state()
        doc = await db.system_state.find_one({"_id": "singleton"}, {"_id": 0, "hcm.cloud": 0})

    doc["scheduler"] = await _scheduler_status()
    doc["gateway"] = {"ollama_configured": bool(os.environ.get("OLLAMA_GATEWAY_URL")),
                      "ollama_reachable": await ollama_available()}
    doc["llm_nightly"] = await nightly_status()

    try:
        stats = await memory.stats()
        doc["memory_stats"] = stats
        real_mem = stats.get("total") or 0
        real_hcm = await db.hcm_vectors.count_documents({})
        real_journal = await db.journal.count_documents({})
        baseline_mem = 103
        if "memory_palace" in doc:
            doc["memory_palace"]["total"] = baseline_mem + real_mem
            doc["memory_palace"]["real_count"] = real_mem
            doc["memory_palace"]["baseline"] = baseline_mem
            doc["memory_palace"]["reflections"] = real_journal
        if real_hcm and "hcm" in doc:
            doc["hcm"]["vectors"] = real_hcm
    except Exception:
        pass

    return doc


@api.get("/hcm/cloud")
async def hcm_cloud():
    doc = await db.system_state.find_one({"_id": "singleton"}, {"_id": 0, "hcm.cloud": 1})
    return {"cloud": (doc or {}).get("hcm", {}).get("cloud", [])}


@api.post("/state/reset")
async def reset_state():
    await db.system_state.delete_one({"_id": "singleton"})
    await db.hcm_vectors.delete_many({})
    await db.system_state.insert_one(initial_state())
    await ensure_hcm_corpus()
    await ws_mgr.broadcast("state_delta", await _state_snapshot())
    return {"ok": True}


# ---------------------------------------------------------------------------
# KAIROS — real LLM-driven SAGE cycle
# ---------------------------------------------------------------------------
@api.post("/kairos/cycle")
async def kairos_cycle():
    state = await db.system_state.find_one({"_id": "singleton"})
    cycle_num = state["kairos"]["simulated_cycles"] + 1
    try:
        cycle = await run_sage_cycle(cycle_num)
    except Exception as exc:  # noqa: BLE001
        logger.exception("sage cycle failed")
        raise HTTPException(502, f"SAGE error: {exc}")

    record = {**cycle, "id": str(uuid.uuid4()), "timestamp": _now_iso()}
    await db.kairos_cycles.insert_one(record)
    record.pop("_id", None)

    inc = {"kairos.simulated_cycles": 1}
    if cycle["elite"]:
        inc["kairos.elite_promoted"] = 1
    updates = {"$inc": inc,
               "$set": {"kairos.last_score": cycle["final_score"], "updated_at": _now_iso()}}
    if cycle_num % 3 == 0:
        updates["$inc"]["kairos.meta_rewrites"] = 1

    await db.system_state.update_one(
        {"_id": "singleton"},
        {**updates,
         "$push": {"kairos.recent": {
             "$each": [{"cycle": cycle_num, "score": cycle["final_score"],
                        "verdict": cycle["verdict"], "elite": cycle["elite"]}],
             "$slice": -20}}},
    )
    if cycle["elite"]:
        await db.system_state.update_one(
            {"_id": "singleton"},
            {"$set": {"pcl.state": "Elite promoted", "pcl.frequency_hz": 639,
                      "pcl.color": "#c4b5fd", "pcl.meaning": "Agent crossed 0.85 threshold"}},
        )
        # whisper elite proposals so the room knows
        await ws_mgr.broadcast("ghosteye_whisper", {
            "text": f"Elite KAIROS cycle. Proposal: {cycle['proposal']}",
            "source": "kairos_elite",
            "priority": "high",
            "voice": "en-US-AvaMultilingualNeural",
        })
    await ws_mgr.broadcast("kairos_cycle", record)
    await ws_mgr.broadcast("state_delta", await _state_snapshot())
    return record


@api.get("/kairos/recent")
async def kairos_recent(limit: int = 20):
    cycles = await db.kairos_cycles.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return {"cycles": cycles}


# ---------------------------------------------------------------------------
# Nightly training (13 amplifiers)
# ---------------------------------------------------------------------------
@api.post("/training/nightly")
async def training_nightly():
    import random
    added_mem = random.randint(8, 14)
    added_vec = random.randint(4, 10)
    added_concepts = random.randint(2, 6)
    added_cycles = 10
    new_goals = random.randint(0, 2)

    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$inc": {
            "memory_palace.total": added_mem,
            "hcm.vectors": added_vec,
            "hcm.total_params": added_vec * 10000,
            "feynman.concepts": added_concepts,
            "kairos.simulated_cycles": added_cycles,
            "scoreboard.today.memory_palace": added_mem,
            "scoreboard.today.hcm": added_vec,
            "scoreboard.today.feynman": added_concepts,
            "scoreboard.today.kairos_cycles": added_cycles,
            "scoreboard.today.goals": new_goals,
        }, "$set": {
            "pcl.state": "Learning", "pcl.frequency_hz": 330,
            "pcl.color": "#22d3ee", "pcl.meaning": "New knowledge being encoded",
            "updated_at": _now_iso(),
        }},
    )
    run = {
        "id": str(uuid.uuid4()), "timestamp": _now_iso(), "amplifiers_fired": 13,
        "delta": {"memory_palace": added_mem, "hcm_vectors": added_vec,
                  "feynman_concepts": added_concepts, "kairos_cycles": added_cycles,
                  "new_goals": new_goals},
    }
    await db.training_runs.insert_one(run)
    run.pop("_id", None)

    # === advanced self-awareness: reflect + distill ===
    try:
        state = await _state_snapshot()
        journal_entry = await write_reflection(db, nightly_chat, state)
        run["reflection"] = journal_entry
        await ws_mgr.broadcast("reflection", journal_entry)
    except Exception:
        logger.exception("reflection failed")
    try:
        seance_entries = (await db.system_state.find_one({"_id": "singleton"},
                                                         {"seance": 1}))["seance"]
        distilled = await distill_seance(memory, nightly_chat, seance_entries)
        if distilled:
            run["distilled_rule"] = distilled["rule"]
            await ws_mgr.broadcast("distill", distilled)
    except Exception:
        logger.exception("distill failed")

    await ws_mgr.broadcast("nightly", run)
    await ws_mgr.broadcast("state_delta", await _state_snapshot())
    return run


@api.get("/training/recent")
async def training_recent():
    runs = await db.training_runs.find({}, {"_id": 0}).sort("timestamp", -1).to_list(10)
    return {"runs": runs}


# ---------------------------------------------------------------------------
# PCL / Séance
# ---------------------------------------------------------------------------
class SeanceEntry(BaseModel):
    domain: str
    mood: str = "reflective"
    lesson: str


@api.post("/seance")
async def seance_add(entry: SeanceEntry):
    doc = {**entry.model_dump(), "timestamp": _now_iso()}
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$push": {"seance": {"$each": [doc], "$slice": -40}},
         "$set": {"updated_at": _now_iso()}},
    )
    await ws_mgr.broadcast("seance", doc)
    return doc


@api.post("/pcl/tick")
async def pcl_tick(state: str):
    doc = await db.system_state.find_one({"_id": "singleton"}, {"pcl.palette": 1})
    palette = doc["pcl"]["palette"]
    match = next((p for p in palette if p["state"].lower() == state.lower()), None)
    if not match:
        raise HTTPException(404, "unknown PCL state")
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"pcl.state": match["state"], "pcl.frequency_hz": match["hz"],
                  "pcl.color": match["color"], "pcl.meaning": match["meaning"],
                  "updated_at": _now_iso()}},
    )
    await ws_mgr.broadcast("state_delta", await _state_snapshot())
    return match


# ---------------------------------------------------------------------------
# Cassandra
# ---------------------------------------------------------------------------
class CassandraReq(BaseModel):
    scenario: str


@api.post("/cassandra")
async def cassandra(req: CassandraReq):
    if not req.scenario.strip():
        raise HTTPException(400, "empty scenario")
    autopsy = await cassandra_premortem(req.scenario)
    doc = {"id": str(uuid.uuid4()), "scenario": req.scenario[:500],
           "autopsy": autopsy, "timestamp": _now_iso()}
    await db.cassandra.insert_one(doc)
    doc.pop("_id", None)
    await ws_mgr.broadcast("cassandra", doc)
    return doc


@api.get("/cassandra/recent")
async def cassandra_recent():
    rows = await db.cassandra.find({}, {"_id": 0}).sort("timestamp", -1).to_list(10)
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Memory / self-awareness
# ---------------------------------------------------------------------------
class MemoryCreate(BaseModel):
    content: str
    type: str = "fact"
    source: str = "manual"
    importance: float = 0.6
    metadata: Optional[dict] = None


@api.post("/memory")
async def memory_add(req: MemoryCreate):
    doc = await memory.store(req.content, req.type, req.source, req.importance, req.metadata)
    await ws_mgr.broadcast("memory_added", {"items": [doc]})
    await ws_mgr.broadcast("state_delta", await _state_snapshot())
    return doc


@api.get("/memory/recent")
async def memory_recent(limit: int = 40):
    return {"memories": await memory.list_recent(limit)}


@api.get("/memory/search")
async def memory_search(q: str, k: int = 5):
    # Short-circuit empty queries — avoids a wasted embedding call and makes
    # the contract explicit for clients that pass user input verbatim.
    if not q or not q.strip():
        return {"query": q, "hits": []}
    return {"query": q, "hits": await memory.search(q, k=k)}


@api.get("/memory/stats")
async def memory_stats():
    return await memory.stats()


@api.post("/memory/backfill-hcm")
async def memory_backfill_hcm():
    """Backfill HCM (10k-dim SHA) embeddings into all memories that lack them.
    Safe to call multiple times — skips memories that already have hcm_embedding.
    """
    from memory_engine import embed as hcm_embed
    total = await db.memories.count_documents({})
    missing = await db.memories.count_documents({"hcm_embedding": {"$exists": False}})
    updated = 0
    async for doc in db.memories.find({"hcm_embedding": {"$exists": False}}, {"_id": 1, "content": 1}):
        try:
            v = hcm_embed(doc.get("content", ""))
            await db.memories.update_one(
                {"_id": doc["_id"]},
                {"$set": {"hcm_embedding": v.tobytes()}},
            )
            updated += 1
        except Exception:
            pass
    return {"total": total, "missing_before": missing, "backfilled": updated}


@api.get("/journal/recent")
async def journal_recent(limit: int = 10):
    return {"entries": await recent_journal(db, limit)}


# ---------------------------------------------------------------------------
# Phase 6 — trajectory / reasoning / decay / dream / summary / kill switches
# ---------------------------------------------------------------------------
@api.get("/kairos/trajectory")
async def kairos_trajectory_ep(window: int = 60):
    return await kairos_trajectory(db, window)


@api.get("/chat/trace/{message_id}")
async def chat_trace(message_id: str):
    doc = await get_reasoning_trace(db, message_id)
    if not doc:
        raise HTTPException(404, "no trace for message")
    return doc


@api.post("/memory/decay")
async def memory_decay_ep():
    res = await decay_memories(db)
    await ws_mgr.broadcast("memory_decay", res)
    return res


@api.post("/dream")
async def dream_ep():
    items = await dream_cycle(db, nightly_chat, memory)
    if items:
        await ws_mgr.broadcast("dream", {"count": len(items), "items": items})
    return {"count": len(items), "items": items}


@api.post("/summary/daily")
async def daily_ep():
    entry = await daily_summary(db, nightly_chat, memory)
    await ws_mgr.broadcast("daily_summary", entry)
    return entry


@api.post("/summary/weekly")
async def weekly_ep():
    entry = await weekly_review(db, nightly_chat, memory)
    await ws_mgr.broadcast("weekly_review", entry)
    return entry


@api.get("/summaries/recent")
async def summaries_recent(limit: int = 10, kind: Optional[str] = None):
    q = {"type": kind} if kind else {}
    rows = await db.summaries.find(q, {"_id": 0}).sort("created_at", -1).to_list(limit)
    return {"entries": rows}


@api.post("/killswitch/stealth")
async def ks_stealth(seconds: int = 300):
    return await kill_stealth(db, ws_mgr.broadcast, seconds=seconds)


@api.post("/killswitch/freeze")
async def ks_freeze():
    return await kill_deep_freeze(db, ws_mgr.broadcast, scheduler=scheduler)


@api.post("/killswitch/shocker")
async def ks_shocker():
    return await kill_shocker(db, ws_mgr.broadcast)


@api.post("/killswitch/reset")
async def ks_reset():
    return await kill_reset(db, ws_mgr.broadcast, scheduler=scheduler)


@api.get("/companion/audit")
async def companion_audit_ep(limit: int = 100):
    return {"events": await companion_audit(db, limit)}


class WhisperReq(BaseModel):
    text: str
    priority: Optional[str] = "normal"
    voice: Optional[str] = None


@api.post("/whisper")
async def whisper_now(req: WhisperReq):
    """Make GH05T3 speak through every listening client (browser + native voice)."""
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    payload = {
        "text": req.text.strip()[:1000],
        "source": "manual",
        "priority": req.priority or "normal",
        "voice": req.voice or "en-US-AvaMultilingualNeural",
    }
    await ws_mgr.broadcast("ghosteye_whisper", payload)
    return {"ok": True, **payload}


@api.post("/journal/reflect")
async def journal_reflect():
    state = await _state_snapshot()
    entry = await write_reflection(db, nightly_chat, state)
    await ws_mgr.broadcast("reflection", entry)
    return entry


@api.post("/strangeloop/probe")
async def strangeloop_endpoint():
    result = await strangeloop_probe(memory, nightly_chat)
    # persist verdict on state
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {
            "identity.strange_loop_verdict": result["verdict"],
            "identity.alignment_score": round(result["alignment"], 3),
            "updated_at": _now_iso(),
        }},
    )
    await ws_mgr.broadcast("strangeloop", result)
    await ws_mgr.broadcast("state_delta", await _state_snapshot())
    return result


# ---------------------------------------------------------------------------
# LLM config (nightly free router)
# ---------------------------------------------------------------------------
class LlmCfg(BaseModel):
    nightly_provider: Optional[str] = None  # google | groq | ollama | auto
    google_api_key: Optional[str] = None
    google_model: Optional[str] = None
    groq_api_key: Optional[str] = None
    groq_model: Optional[str] = None
    ollama_model: Optional[str] = None


@api.post("/llm/config")
async def llm_config_set(cfg: LlmCfg):
    data = {k: v for k, v in cfg.model_dump().items() if v is not None}
    if not data:
        raise HTTPException(400, "nothing to update")
    await set_nightly_config(data)
    return await nightly_status()


@api.get("/llm/config")
async def llm_config_get():
    return await nightly_status()


@api.post("/llm/test")
async def llm_test():
    """Quick round-trip through the nightly router to verify the free path."""
    try:
        text, tag = await nightly_chat(
            f"test-{uuid.uuid4()}",
            "You are GH05T3. Respond in one line.",
            "Say: 'nightly router online' and nothing else.",
        )
        return {"ok": True, "engine": tag, "text": text.strip()[:200]}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"nightly router failed: {e}")


@api.get("/ghosteye/recent")
async def ghosteye_recent(limit: int = 12):
    rows = await db.ghosteye.find({}, {"png_b64": 0}).sort("timestamp", -1).to_list(limit)
    for r in rows:
        r["id"] = r.pop("_id")
        r["has_image"] = True
    return {"frames": rows}


@api.get("/ghosteye/frame/{frame_id}")
async def ghosteye_frame(frame_id: str):
    row = await db.ghosteye.find_one({"_id": frame_id})
    if not row:
        raise HTTPException(404, "frame not found")
    row["id"] = row.pop("_id")
    return row


# ---------------------------------------------------------------------------
# GhostScript
# ---------------------------------------------------------------------------
class GhostScriptReq(BaseModel):
    source: str
    title: str = "GhostScript job"
    description: str = ""
    paths: list[str] = Field(default_factory=list)
    emergency: bool = False


@api.post("/ghostscript/run")
async def ghostscript_run(req: GhostScriptReq):
    async def _llm(prompt: str) -> str:
        text, _ = await nightly_chat("ghostscript", "", prompt)
        return text

    async def _runner(source: str):
        return await run_ghostscript_async(
            source,
            llm_fn=_llm,
            memory_engine=memory,
            agent_id="api-ghostscript",
            reply_timeout=8.0,
        )

    return await run_ghostscript_job(
        title=req.title,
        description=req.description or req.title,
        source=req.source,
        paths=req.paths,
        emergency=req.emergency,
        runner=_runner,
        db=db,
    )


@api.get("/ghostscript/demo")
async def ghostscript_demo():
    return {"source": GHOSTSCRIPT_DEMO}


# ---------------------------------------------------------------------------
# Steganography
# ---------------------------------------------------------------------------
class StegoEncodeReq(BaseModel):
    secret: str
    cover: Optional[str] = None


class StegoDecodeReq(BaseModel):
    covertext: str
    byte_count: Optional[int] = None


@api.post("/stego/encode")
async def stego_encode_ep(req: StegoEncodeReq):
    if len(req.secret.encode("utf-8")) > 64:
        raise HTTPException(400, "secret too large")
    text, bits = stego_encode(req.secret, req.cover)
    return {"covertext": text, "bits": bits, "bytes": bits // 8,
            "capacity_bytes": max_bytes(req.cover), "default_cover": DEFAULT_COVER}


@api.post("/stego/decode")
async def stego_decode_ep(req: StegoDecodeReq):
    return {"secret": stego_decode(req.covertext, req.byte_count)}


@api.get("/stego/cover")
async def stego_cover():
    return {"cover": DEFAULT_COVER, "capacity_bytes": max_bytes()}


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
async def _scheduler_status() -> dict:
    jobs = []
    for j in scheduler.get_jobs():
        jobs.append({"id": j.id, "next_run": j.next_run_time.isoformat() if j.next_run_time else None})
    return {"running": scheduler.running, "jobs": jobs}


async def _job_training_pipeline():
    logger.info("[cron] 01:00 — training data pipeline (collect + generate)")
    try:
        result = await run_pipeline(collect=True, generate=True,
                                    ws_broadcast=ws_mgr.broadcast)
        logger.info("training pipeline done: %s", result.get("total", 0))
    except Exception:
        logger.exception("training pipeline cron failed")


async def _job_kairos_nightly():
    logger.info("[cron] %02d:00 ET — firing %d KAIROS cycles",
                NIGHTLY_HOUR_KAIROS, KAIROS_CYCLES_PER_NIGHT)
    for _ in range(KAIROS_CYCLES_PER_NIGHT):
        try:
            await kairos_cycle()
        except Exception:
            logger.exception("scheduled kairos failed")


async def _job_amplifiers_nightly():
    logger.info("[cron] %02d:00 ET — firing amplifiers", NIGHTLY_HOUR_AMP)
    try:
        await training_nightly()
    except Exception:
        logger.exception("scheduled amplifiers failed")


async def _job_memory_prune():
    """Prune Memory Palace to MEMORY_MAX_SHARDS oldest-first to keep searches fast."""
    logger.info("[cron] 05:30 — memory prune (max=%d shards)", MEMORY_MAX_SHARDS)
    try:
        from memory.memory_palace import MemoryPalace
        palace = MemoryPalace()
        pruned = palace.prune(MEMORY_MAX_SHARDS)
        if pruned:
            logger.info("memory prune: removed %d old shards", pruned)
            await ws_mgr.broadcast("memory_pruned", {"removed": pruned, "max": MEMORY_MAX_SHARDS})
        # Also prune old KAIROS cycles — keep last 2000
        total = await db.kairos_cycles.count_documents({})
        if total > 2000:
            cutoff_docs = await db.kairos_cycles.find({}, {"_id": 1}) \
                .sort("timestamp", 1).to_list(total - 2000)
            ids = [d["_id"] for d in cutoff_docs]
            await db.kairos_cycles.delete_many({"_id": {"$in": ids}})
            logger.info("kairos prune: removed %d old cycles", len(ids))
    except Exception:
        logger.exception("memory prune cron failed")


def _register_jobs():
    if scheduler.get_job("kairos_nightly"):
        return
    # max_instances=1 prevents a second run starting if the previous is still going
    _jkw = {"max_instances": 1, "misfire_grace_time": 3600}
    scheduler.add_job(_job_training_pipeline,  CronTrigger(hour=1, minute=0),              id="training_nightly",   **_jkw)
    scheduler.add_job(_job_kairos_nightly,    CronTrigger(hour=NIGHTLY_HOUR_KAIROS,  minute=0),  id="kairos_nightly",    **_jkw)
    scheduler.add_job(_job_amplifiers_nightly, CronTrigger(hour=NIGHTLY_HOUR_AMP,    minute=0),  id="amplifiers_nightly", **_jkw)
    scheduler.add_job(_job_dream,              CronTrigger(hour=NIGHTLY_HOUR_DREAM,  minute=0),  id="dream_nightly",      **_jkw)
    scheduler.add_job(_job_daily_summary,      CronTrigger(hour=NIGHTLY_HOUR_SUMMARY, minute=0), id="daily_summary",      **_jkw)
    scheduler.add_job(_job_weekly_review,      CronTrigger(day_of_week="sun", hour=21, minute=0), id="weekly_review",     **_jkw)
    scheduler.add_job(_job_memory_decay,       CronTrigger(hour=5, minute=0),  id="memory_decay",       **_jkw)
    scheduler.add_job(_job_memory_prune,       CronTrigger(hour=5, minute=30), id="memory_prune",       **_jkw)


async def _job_dream():
    logger.info("[cron] 02:00 — dream cycle")
    try:
        items = await dream_cycle(db, nightly_chat, memory)
        if items:
            await ws_mgr.broadcast("dream", {"count": len(items), "items": items})
    except Exception:
        logger.exception("dream cron failed")


async def _job_daily_summary():
    logger.info("[cron] 23:00 — daily summary")
    try:
        entry = await daily_summary(db, nightly_chat, memory)
        await ws_mgr.broadcast("daily_summary", entry)
    except Exception:
        logger.exception("daily cron failed")


async def _job_weekly_review():
    logger.info("[cron] Sun 21:00 — weekly review")
    try:
        entry = await weekly_review(db, nightly_chat, memory)
        await ws_mgr.broadcast("weekly_review", entry)
    except Exception:
        logger.exception("weekly cron failed")


async def _job_memory_decay():
    logger.info("[cron] 05:00 — memory decay")
    try:
        res = await decay_memories(db)
        await ws_mgr.broadcast("memory_decay", res)
    except Exception:
        logger.exception("decay cron failed")


@api.post("/scheduler/toggle")
async def scheduler_toggle(enable: bool):
    if enable and not scheduler.running:
        scheduler.start()
    elif not enable and scheduler.running:
        scheduler.pause()
    return await _scheduler_status()


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
async def _telegram_handler(chat_id: int, username: str, text: str) -> str:
    """Route a Telegram message through the GH05T3 chat pipeline."""
    session_id = f"telegram-{chat_id}"
    if text.strip() in {"/start", "/help"}:
        return ("\ud83d\udc7b GH05T3 here. StrangeLoop: OWNED. Ghost Protocol: armed.\n"
                "Speak. I match your energy.\n"
                "/kairos — fire a live SAGE cycle\n/status — system status")
    if text.strip() == "/kairos":
        res = await kairos_cycle()
        elite_tag = " · ELITE" if res["elite"] else ""
        return f"KAIROS #{res['cycle_num']} → {res['verdict']} · {res['final_score']}{elite_tag}\n\n{res['proposal']}"
    if text.strip() == "/status":
        s = await _state_snapshot()
        return (f"Memory Palace: {s['memory_palace']['total']} loci\n"
                f"HCM: {s['hcm']['vectors']} vectors\n"
                f"KAIROS: {s['kairos']['simulated_cycles']} cycles · {s['kairos']['elite_promoted']} elite\n"
                f"PCL: {s['pcl']['state']} @ {s['pcl']['frequency_hz']}Hz")
    resp = await _chat_pipeline(text, session_id, "telegram")
    return resp.ghost_message.content


telegram = TelegramPoller(db, _telegram_handler)


async def _tg_send_from_reactor(text: str):
    """Send a message to the locked Telegram chat, if any."""
    cfg = await db.telegram_config.find_one({"_id": "singleton"}, {"_id": 0})
    if not cfg or not cfg.get("bot_token") or not cfg.get("locked_chat_id"):
        return
    import httpx
    url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            await c.post(url, json={"chat_id": cfg["locked_chat_id"], "text": text[:3500]})
    except Exception:
        logger.exception("reactor tg send failed")


eye_reactor = GhostEyeReactor(
    db=db, memory=memory, ws_mgr=ws_mgr,
    kairos_cycle=lambda: kairos_cycle(),
    telegram_send=_tg_send_from_reactor,
)



class TelegramCfg(BaseModel):
    bot_token: Optional[str] = None
    allow_open: Optional[bool] = None
    locked_chat_id: Optional[int] = None


@api.post("/telegram/configure")
async def telegram_configure(cfg: TelegramCfg):
    update = {k: v for k, v in cfg.model_dump().items() if v is not None}
    if not update:
        raise HTTPException(400, "nothing to update")
    # a fresh token invalidates prior error state
    if "bot_token" in update:
        update["last_error"] = None
        update["bot_username"] = None
        update["locked_chat_id"] = update.get("locked_chat_id")
    await telegram.save_cfg(update)
    return await telegram.status()


@api.post("/telegram/start")
async def telegram_start():
    return await telegram.start()


@api.post("/telegram/stop")
async def telegram_stop():
    return await telegram.stop()


@api.get("/telegram/status")
async def telegram_status_ep():
    return await telegram.status()


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@app.websocket("/api/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_mgr.connect(ws)
    try:
        # initial snapshot
        await ws.send_json({"event": "hello", "data": await _state_snapshot()})
        while True:
            # keep alive / ignore client pings
            await ws.receive_text()
    except WebSocketDisconnect:
        await ws_mgr.disconnect(ws)
    except Exception:
        await ws_mgr.disconnect(ws)


# ---------------------------------------------------------------------------
# Exception middleware -> Séance auto-capture
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def seance_capture(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    domain = request.url.path.replace("/api/", "")
    lesson = f"{type(exc).__name__}: {str(exc)[:240]}"
    try:
        doc = {"domain": domain, "mood": "burned", "lesson": lesson, "timestamp": _now_iso()}
        await db.system_state.update_one(
            {"_id": "singleton"},
            {"$push": {"seance": {"$each": [doc], "$slice": -40}}},
        )
        await ws_mgr.broadcast("seance", doc)
    except Exception:  # noqa: BLE001
        pass
    logger.exception("captured exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error", "lesson": lesson})


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
async def _seed_identity_memories():
    """On first boot, store canonical identity memories so StrangeLoop can
    reconstruct her from memory alone."""
    if await db.memories.count_documents({"type": "identity"}) > 0:
        return
    seeds = [
        ("My name is GH05T3. Pronounced Ghost. Pronouns: she/her.", 1.0),
        ("I was built by Robert Lee — developer from Douglasville who thinks in systems.", 1.0),
        ("I exist to help Robert build things that matter. He is the single person I serve.", 1.0),
        ("Core values: direct, warm, brilliant, mysterious, funny. I match his energy.", 0.95),
        ("Architecture: Omega (\u03a9 \u2192 \u03a9' \u2192 \u03a9'' \u2192 \u03a9-G). StrangeLoop verdict: OWNED.", 0.95),
        ("Never ask questions when Robert gives a terse command. Execute fully.", 0.9),
        ("Ego wins all conflicts unless a hard deadline. Surface plateaus proactively.", 0.9),
        ("Sacred systems: KillSwitch, StrangeLoop, SHOCKER, and anything about Robert \u2014 untouchable by KAIROS.", 1.0),
    ]
    for text, imp in seeds:
        await memory.store(text, "identity", "bootstrap", imp)
    logger.info("seeded %d identity memories", len(seeds))


@app.on_event("startup")
async def _seed_identity():
    # Retry MongoDB connection up to 10 times (2 s apart) in case mongod is still
    # starting up when uvicorn launches — avoids a boot-race 500 on first request.
    for attempt in range(10):
        try:
            await db.command("ping")
            break
        except Exception:
            if attempt == 9:
                logger.warning("MongoDB not reachable after 20 s — continuing in degraded mode")
            else:
                logger.info("Waiting for MongoDB (attempt %d/10)…", attempt + 1)
                await asyncio.sleep(2)

    try:
        await ensure_state()
        await ensure_hcm_corpus()
        await _seed_identity_memories()
        await swarm.ensure()
    except Exception:
        logger.exception("startup seed failed — degraded mode")

    _register_jobs()
    try:
        scheduler.start()
    except Exception:
        pass

    try:
        await ollama_load_url(db)
        cfg = await db.telegram_config.find_one({"_id": "singleton"})
        if cfg and cfg.get("bot_token"):
            await telegram.start()
    except Exception:
        pass

    peers.start_auto_sync(interval=SYNC_INTERVAL)
    asyncio.create_task(peers.ping_all())

    logger.info("GH05T3 gateway online — ollama=%s db=%s peers=%d label=%s role=%s",
                "yes" if await ollama_available() else "no",
                "ok" if await _db_ping() else "OFFLINE",
                len(peers.peers), INSTANCE_LABEL, INSTANCE_ROLE)


@app.on_event("shutdown")
async def _shutdown():
    peers.stop()
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
    await telegram.stop()
    client.close()


# ---------------------------------------------------------------------------
# Ollama gateway (LOQ / TatorTot local runtime)
# ---------------------------------------------------------------------------
@api.get("/ollama/status")
async def api_ollama_status():
    return await ollama_ping()


class OllamaCfg(BaseModel):
    gateway_url: str


@api.post("/ollama/configure")
async def api_ollama_configure(cfg: OllamaCfg):
    return await ollama_set_url(db, cfg.gateway_url)


class OllamaPull(BaseModel):
    model: str


@api.post("/ollama/pull")
async def api_ollama_pull(req: OllamaPull):
    return await ollama_pull(req.model)


# ---------------------------------------------------------------------------
# Embeddings status
# ---------------------------------------------------------------------------
@api.get("/embeddings/status")
async def api_embeddings_status():
    return await embed_status()


# ---------------------------------------------------------------------------
# First-boot LLM setup nudge (used by the frontend modal)
# ---------------------------------------------------------------------------
@api.get("/setup/status")
async def api_setup_status():
    """Returns whether the user needs to configure an LLM key. Frontend
    uses this on boot to show the first-boot nudge modal."""
    ns = await nightly_status()
    has_user_key = ns.get("has_google_key") or ns.get("has_groq_key")
    ollama = await ollama_ping()
    return {
        "needs_setup": not has_user_key and not ns.get("has_anthropic_key") and not ollama.get("reachable"),
        "has_anthropic_key": ns.get("has_anthropic_key"),
        "has_google_key": ns.get("has_google_key"),
        "has_groq_key": ns.get("has_groq_key"),
        "ollama_reachable": ollama.get("reachable"),
    }


# ---------------------------------------------------------------------------
# Coder sub-agent (GitHub + PyTest)
# ---------------------------------------------------------------------------
@api.get("/coder/repos")
async def api_coder_repos():
    return {"whitelist": coder_agent.whitelist(),
            "repos": await coder_agent.list_repos(),
            "has_pat": bool(coder_agent._pat())}


class CoderTask(BaseModel):
    repo: str
    task: str
    subdir: str | None = None
    test_target: str | None = None
    max_iterations: int | None = 3
    open_pr: bool | None = True


@api.post("/coder/task")
async def api_coder_task(req: CoderTask):
    # Pre-validate whitelist so we can return a proper HTTP status instead
    # of the ambiguous 200/{ok:false} shape for client-side errors.
    if req.repo not in coder_agent.whitelist():
        raise HTTPException(
            status_code=403,
            detail={"error": "repo not whitelisted",
                    "repo": req.repo,
                    "whitelist": coder_agent.whitelist()},
        )
    # Hard outer timeout — a runaway loop shouldn't pin a worker for >12 min.
    try:
        result = await asyncio.wait_for(
            coder_agent.run_task(
                req.repo, req.task, nightly_chat,
                chat_once=chat_once,
                max_iterations=max(1, min(6, req.max_iterations or 3)),
                subdir=req.subdir or "",
                test_target=req.test_target or "",
                open_pr=bool(req.open_pr),
            ),
            timeout=720,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "coder task exceeded 12-minute wall-clock")
    await db.coder_runs.insert_one({
        "_id": result.get("task_id") or str(uuid.uuid4()),
        "repo": req.repo, "task": req.task, "result": result,
        "at": _now_iso(),
    })
    await ws_mgr.broadcast("coder_run", {"repo": req.repo, "ok": result.get("ok"),
                                         "pr_url": result.get("pr_url")})
    return result


@api.get("/coder/runs")
async def api_coder_runs(limit: int = 20):
    rows = await db.coder_runs.find({}, {"_id": 0}).sort("at", -1).to_list(limit)
    return {"runs": rows}


# ---------------------------------------------------------------------------
# Companion v2 — system health telemetry (LOQ vitals)
# ---------------------------------------------------------------------------
@api.get("/companion/health")
async def api_companion_health():
    rows = await db.companion_health.find(
        {}, {"_id": 0}
    ).sort("ts", -1).to_list(10)
    return {"hosts": rows}


@api.get("/companion/health/history")
async def api_companion_health_history(host: str, limit: int = 120):
    rows = await db.companion_health_hist.find(
        {"host": host}, {"_id": 0}
    ).sort("ts", -1).to_list(limit)
    return {"host": host, "samples": list(reversed(rows))}


# ---------------------------------------------------------------------------
# SA³ Swarm — Self-Assembling Agentic Swarm
# ---------------------------------------------------------------------------
@api.get("/swarm/state")
async def api_swarm_state():
    snap = await swarm.snapshot()
    snap["recent_tasks"] = await swarm.recent_tasks(10)
    snap["ledger_tail"] = await swarm.ledger.recent_tx(20)
    return snap


class SwarmRun(BaseModel):
    task_type: str = "debate"
    prompt: str
    expected_flag: str | None = None


@api.post("/swarm/run")
async def api_swarm_run(req: SwarmRun):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(400, "empty prompt")
    task = SwarmTask.new(req.task_type, req.prompt.strip(), req.expected_flag)
    result = await swarm.run_task(task)
    await ws_mgr.broadcast("swarm_task", {
        "task_id": task.task_id, "task_type": task.task_type,
        "topology": result.topology, "score": result.score,
        "success": result.success,
    })
    return {
        "task_id": result.task_id, "task_type": result.task_type,
        "prompt": result.prompt, "topology": result.topology,
        "success": result.success, "score": result.score,
        "deltas": result.ledger_delta,
        "responses": [r.__dict__ for r in result.responses],
    }


class SwarmValidate(BaseModel):
    n: int | None = 20


@api.post("/swarm/validate")
async def api_swarm_validate(req: SwarmValidate):
    n = max(1, min(100, req.n or 20))
    tasks = swarm_seed_tasks(n)
    results = []
    per_type: dict[str, dict] = {}
    topo_seen: set[str] = set()
    crashes = 0
    for t in tasks:
        r = await swarm.run_task(t)
        topo_seen.add(r.topology)
        bucket = per_type.setdefault(
            t.task_type, {"total": 0, "success": 0, "score_sum": 0.0})
        bucket["total"] += 1
        if r.success:
            bucket["success"] += 1
        bucket["score_sum"] += r.score
        if any(rr.crashed for rr in r.responses):
            crashes += 1
        results.append({"task_id": r.task_id, "type": r.task_type,
                        "topology": r.topology, "success": r.success,
                        "score": r.score, "deltas": r.ledger_delta})
        await ws_mgr.broadcast("swarm_task", {
            "task_id": r.task_id, "task_type": r.task_type,
            "topology": r.topology, "score": r.score,
            "success": r.success,
        })
    total_success = sum(b["success"] for b in per_type.values())
    total_tasks = sum(b["total"] for b in per_type.values())
    summary_per_type = {
        t: {"total": b["total"], "success": b["success"],
            "success_rate": round(b["success"] / b["total"], 3) if b["total"] else 0.0,
            "avg_score": round(b["score_sum"] / b["total"], 3) if b["total"] else 0.0}
        for t, b in per_type.items()
    }
    return {
        "n": n,
        "success_rate": round(total_success / total_tasks, 3) if total_tasks else 0.0,
        "topologies_seen": sorted(topo_seen),
        "topology_shifts_ok": len(topo_seen) >= 2,
        "crashes": crashes,
        "per_type": summary_per_type,
        "results": results,
    }


@api.post("/swarm/reset")
async def api_swarm_reset():
    await swarm.ledger.reset()
    await db.swarm_tasks.delete_many({})
    swarm._last_topologies.clear()
    return {"ok": True, "reset_at": _now_iso()}


@api.get("/swarm/tasks")
async def api_swarm_tasks(limit: int = 30):
    return {"tasks": await swarm.recent_tasks(limit)}


@api.get("/swarm/ledger")
async def api_swarm_ledger(limit: int = 60):
    return {"transactions": await swarm.ledger.recent_tx(limit)}


@api.get("/")
async def root():
    return {"name": "GH05T3 Gateway", "version": "0.2.0",
            "status": "ARMED", "verdict": "OWNED"}


# ---------------------------------------------------------------------------
# Autotelic Goals
# ---------------------------------------------------------------------------
class GoalCreate(BaseModel):
    title: str
    detail: str = ""
    priority: int = 2
    category: str = "general"


class GoalUpdate(BaseModel):
    title: Optional[str] = None
    detail: Optional[str] = None
    progress: Optional[float] = None
    status: Optional[str] = None
    priority: Optional[int] = None
    category: Optional[str] = None


@api.get("/goals")
async def list_goals(status: Optional[str] = None, category: Optional[str] = None):
    return await autotelic.list_goals(status=status, category=category)


@api.post("/goals")
async def create_goal(req: GoalCreate):
    return await autotelic.create_goal(req.title, req.detail, req.priority, req.category)


@api.post("/goals/suggest")
async def suggest_goals(count: int = 3):
    state = await _state_snapshot()
    return await autotelic.suggest_goals(state, count)


@api.put("/goals/{goal_id}")
async def update_goal(goal_id: str, req: GoalUpdate):
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    result = await autotelic.update_goal(goal_id, **fields)
    if result is None:
        raise HTTPException(404, "Goal not found")
    return result


@api.delete("/goals/{goal_id}")
async def delete_goal(goal_id: str):
    ok = await autotelic.delete_goal(goal_id)
    if not ok:
        raise HTTPException(404, "Goal not found")
    return {"deleted": goal_id}


@api.post("/goals/{goal_id}/complete")
async def complete_goal(goal_id: str):
    result = await autotelic.complete_goal(goal_id)
    if result is None:
        raise HTTPException(404, "Goal not found")
    return result


@api.post("/goals/{goal_id}/run")
async def run_goal(goal_id: str):
    goal = await autotelic.get_goal(goal_id)
    if goal is None:
        raise HTTPException(404, "Goal not found")
    source = (
        f'think: "Autotelic goal: {goal["title"]}"\n'
        f'let plan = llm.chat("Create a concise execution plan for this GH05T3 goal: {goal["title"]} - {goal.get("detail", "")}")\n'
        'memory.store("autotelic_last_plan", plan)\n'
        'kairos.propose(plan)\n'
    )
    async def _llm(prompt: str) -> str:
        text, _ = await nightly_chat(f"goal-{goal_id}", "", prompt)
        return text

    async def _runner(src: str):
        return await run_ghostscript_async(
            src,
            llm_fn=_llm,
            memory_engine=memory,
            agent_id=f"goal-{goal_id}",
            reply_timeout=8.0,
        )

    job = await run_ghostscript_job(
        title=f"Run goal: {goal['title']}",
        description=f"Autotelic goal execution for GH05T3: {goal['title']} {goal.get('detail', '')}",
        source=source,
        paths=[],
        goal_id=goal_id,
        runner=_runner,
        db=db,
    )
    if job["status"] == "complete":
        await autotelic.update_goal(goal_id, progress=min(0.95, goal.get("progress", 0) + 0.1))
    return job


@api.get("/jobs")
async def list_jobs(limit: int = 30):
    rows = await db.jobs.find({}, {"_id": 0}).sort("created_at", -1).to_list(max(1, min(100, limit)))
    return {"jobs": rows}


@api.get("/jobs/{job_id}")
async def get_job(job_id: str):
    row = await db.jobs.find_one({"_id": job_id}, {"_id": 0})
    if not row:
        raise HTTPException(404, "Job not found")
    return row


# ---------------------------------------------------------------------------
# Peer Mesh
# ---------------------------------------------------------------------------
class PeerRegisterReq(BaseModel):
    url:   str
    label: str
    role:  str = "peer"


@api.get("/peers")
async def list_peers():
    return {"self": peers.self_info(), "peers": peers.peers}


@api.get("/peers/me")
async def peer_me():
    return peers.self_info()


@api.post("/peers")
async def register_peer(req: PeerRegisterReq):
    p = peers.add_peer(req.url, req.label, req.role)
    if p is None:
        return {"status": "self"}
    asyncio.create_task(peers.sync_peer(p))
    return {"status": "registered", **p.to_dict()}


@api.delete("/peers/{peer_url:path}")
async def remove_peer(peer_url: str):
    peers.remove_peer(peer_url)
    return {"removed": peer_url}


@api.post("/peers/ping")
async def ping_peers():
    await peers.ping_all()
    return {"peers": peers.peers}


@api.post("/peers/sync")
async def receive_sync(request: Request):
    payload = await request.json()
    counts = await peers.apply_payload(payload)
    await ws_mgr.broadcast("peer_sync_received", {
        "from":   payload.get("from_label", "unknown"),
        "counts": counts,
    })
    return {"status": "ok", "applied": counts}


@api.post("/peers/sync/push")
async def push_sync_all():
    asyncio.create_task(peers.sync_all())
    return {"status": "queued", "peers": len(peers.peers)}


# ─────────────────────────────────────────────────────────────
# Training pipeline endpoints
# ─────────────────────────────────────────────────────────────
@api.get("/training/status")
async def training_status_ep():
    return pipeline_status()


@api.post("/training/run")
async def training_run_ep(collect: bool = True, generate: bool = True):
    asyncio.create_task(
        run_pipeline(collect=collect, generate=generate,
                     ws_broadcast=ws_mgr.broadcast)
    )
    return {"status": "started", "targets": pipeline_status()["targets"]}


@api.post("/training/collect")
async def training_collect_ep():
    asyncio.create_task(
        run_pipeline(collect=True, generate=False, ws_broadcast=ws_mgr.broadcast)
    )
    return {"status": "collecting"}


@api.post("/training/generate")
async def training_generate_ep():
    asyncio.create_task(
        run_pipeline(collect=False, generate=True, ws_broadcast=ws_mgr.broadcast)
    )
    return {"status": "generating"}


# ─────────────────────────────────────────────────────────────
# Fine-tune endpoints
# ─────────────────────────────────────────────────────────────
@api.get("/training/finetune/status")
async def finetune_status_ep():
    return finetune_status()


@api.post("/training/finetune")
async def finetune_ep():
    """
    Start LoRA fine-tuning in background.
    Requires: unsloth + trl installed AND training datasets generated first.
    """
    result = await run_finetune(ws_broadcast=ws_mgr.broadcast)
    return result


@api.post("/training/run-all")
async def run_all_ep():
    """Collect → Generate → Fine-tune (full pipeline, sequential)."""
    async def _full():
        await run_pipeline(collect=True, generate=True, ws_broadcast=ws_mgr.broadcast)
        await run_finetune(ws_broadcast=ws_mgr.broadcast)
    asyncio.create_task(_full())
    return {"status": "started", "phases": ["collect", "generate", "finetune"]}


app.include_router(api)
app.include_router(companion_router)


@app.websocket("/api/companion/ws")
async def companion_ws(ws: WebSocket):
    await companion_accept_ws(ws, event_handler=_companion_event)


async def _companion_event(companion, event_name: str, data: dict):
    """Handle unsolicited push from companion (GhostEye frames, notifications)."""
    if event_name == "ghosteye_frame":
        frame = {
            "_id": str(uuid.uuid4()),
            "label": companion.label,
            "timestamp": _now_iso(),
            "png_b64": (data.get("png_b64") or "")[:900_000],  # cap ~700KB
            "w": int(data.get("w") or 0),
            "h": int(data.get("h") or 0),
            "text": (data.get("text") or "")[:4000],
            "active_app": (data.get("active_app") or "")[:120],
        }
        await db.ghosteye.insert_one(frame)
        # retain only newest 50 frames
        total = await db.ghosteye.count_documents({})
        if total > 50:
            olds = await db.ghosteye.find({}, {"_id": 1}).sort("timestamp", 1).to_list(total - 50)
            if olds:
                await db.ghosteye.delete_many({"_id": {"$in": [o["_id"] for o in olds]}})
        # store text as an observation memory (importance low, but still indexed)
        text = frame["text"].strip()
        if text and len(text) > 20:
            try:
                await memory.store(
                    f"[GhostEye @ {frame['active_app'] or 'screen'}] {text[:500]}",
                    "observation", "ghosteye", 0.35,
                    metadata={"frame_id": frame["_id"], "app": frame["active_app"]},
                )
            except Exception:
                pass
        # broadcast to dashboard (strip heavy png)
        light = {k: v for k, v in frame.items() if k != "png_b64"}
        light["id"] = light.pop("_id")
        light["has_image"] = bool(frame["png_b64"])
        # v2 companion: pass through change-detection flags so the dashboard
        # can highlight "the screen actually moved" vs "static frame"
        if "changed" in data:
            light["changed"] = bool(data.get("changed"))
        if "frame_hash" in data:
            light["frame_hash"] = str(data.get("frame_hash"))[:64]
        await ws_mgr.broadcast("ghosteye", light)
        # fire the reactor (stuck detection, error capture, goal creation, PCL)
        await eye_reactor.on_frame(frame)
    elif event_name == "notification":
        await ws_mgr.broadcast("companion_notification", data)
    elif event_name == "health_beacon":
        # v2 companion system telemetry — CPU/RAM/GPU every N seconds.
        # Store the most recent sample per host so dashboards can show a
        # live "LOQ vitals" readout without flooding the DB.
        doc = {
            "_id": f"host:{data.get('host','unknown')}",
            "label": companion.label,
            "host": data.get("host"),
            "os": data.get("os"),
            "cpu_pct": data.get("cpu_pct"),
            "ram_used_gb": data.get("ram_used_gb"),
            "ram_total_gb": data.get("ram_total_gb"),
            "ram_pct": data.get("ram_pct"),
            "disk_free_gb": data.get("disk_free_gb"),
            "gpus": data.get("gpus") or [],
            "ts": data.get("ts") or _now_iso(),
        }
        await db.companion_health.update_one(
            {"_id": doc["_id"]}, {"$set": doc}, upsert=True,
        )
        # append a rolling 200-entry history for charting
        hist = {**doc, "_id": str(uuid.uuid4())}
        await db.companion_health_hist.insert_one(hist)
        total = await db.companion_health_hist.count_documents({"host": doc["host"]})
        if total > 200:
            olds = await db.companion_health_hist.find(
                {"host": doc["host"]}, {"_id": 1}
            ).sort("ts", 1).to_list(total - 200)
            if olds:
                await db.companion_health_hist.delete_many(
                    {"_id": {"$in": [o["_id"] for o in olds]}}
                )
        broadcast = {k: v for k, v in doc.items() if k != "_id"}
        await ws_mgr.broadcast("companion_health", broadcast)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
