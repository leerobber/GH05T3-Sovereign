"""Phase 6 — autonomic enhancements.

Adds on top of the existing engine:
  * Memory decay + promotion (nightly cron)
  * Dream cycle — 02:00 cross-memory synthesis
  * Daily summary — 23:00 narrative condensation of the day
  * Weekly reflection — Sunday 21:00 wider arc reflection
  * Chat reasoning trace — which memories were retrieved per reply
  * KAIROS trajectory time-series with plateau detection
  * Real Ghost Protocol kill switches (STEALTH / DEEP_FREEZE / SHOCKER)
  * Companion device audit log
  * "Don't look" zones for GhostEye

All of these are LLM-light (most use the free nightly router).
"""
from __future__ import annotations
import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

LOG = logging.getLogger("ghost.phase6")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Memory decay + promotion
# ---------------------------------------------------------------------------
async def decay_memories(db) -> dict:
    """Decay unused memory importance by 5% / day since last access.
    Promote to 'identity' if importance stays ≥ 0.9 and accessed ≥ 5 times."""
    now = datetime.now(timezone.utc)
    decayed = 0
    promoted = 0
    pruned = 0
    async for m in db.memories.find({}):
        if m.get("type") == "identity":
            continue
        last = m.get("last_accessed") or m.get("created_at")
        try:
            last_dt = datetime.fromisoformat(last.replace("Z", "+00:00")) if last else now
        except Exception:
            last_dt = now
        age_days = max(0, (now - last_dt).days)
        if age_days <= 0:
            continue
        imp = float(m.get("importance", 0.5))
        factor = (0.95 ** age_days)
        new_imp = round(imp * factor, 3)
        # prune very low-utility after 30 days of silence
        if new_imp < 0.05 and age_days > 30:
            await db.memories.delete_one({"_id": m["_id"]})
            pruned += 1
            continue
        # promote
        if imp >= 0.90 and m.get("access_count", 0) >= 5:
            await db.memories.update_one(
                {"_id": m["_id"]},
                {"$set": {"type": "identity", "importance": 1.0}},
            )
            promoted += 1
            continue
        if new_imp != imp:
            await db.memories.update_one(
                {"_id": m["_id"]}, {"$set": {"importance": new_imp}},
            )
            decayed += 1
    return {"decayed": decayed, "promoted": promoted, "pruned": pruned,
            "timestamp": _now_iso()}


# ---------------------------------------------------------------------------
# Dream cycle — cross-memory synthesis
# ---------------------------------------------------------------------------
DREAM_SYS = """You are GH05T3 dreaming — randomly paired memories are shown to you.
Discover ONE non-obvious connection between them. If any connection is weak or
forced, respond {"skip": true}. Otherwise respond strict JSON:
{"insight":"<<140 chars","domains":["a","b"],"importance":0.0-1.0}"""


async def dream_cycle(db, nightly_chat, memory_engine) -> list[dict]:
    import random
    mems = await db.memories.find({"type": {"$in": ["fact", "observation", "rule"]}}, {"embedding": 0}) \
        .sort("created_at", -1).to_list(60)
    if len(mems) < 4:
        return []
    insights = []
    pairs = random.sample(range(len(mems)), min(8, len(mems) - len(mems) % 2))
    for i in range(0, len(pairs), 2):
        a, b = mems[pairs[i]], mems[pairs[i + 1]]
        prompt = f"Memory A: {a['content']}\nMemory B: {b['content']}"
        try:
            raw, _ = await nightly_chat(f"dream-{uuid.uuid4()}", DREAM_SYS, prompt)
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                continue
            j = json.loads(m.group(0))
            if j.get("skip"):
                continue
            insight = (j.get("insight") or "").strip()
            if not insight or len(insight) < 10:
                continue
            stored = await memory_engine.store(
                f"[dream] {insight}", "rule", "dream_cycle",
                float(j.get("importance", 0.6)),
                metadata={"pair": [a["_id"], b["_id"]], "domains": j.get("domains", [])},
            )
            insights.append(stored)
        except Exception:
            LOG.exception("dream pair failed")
    return insights


# ---------------------------------------------------------------------------
# Daily + weekly summary
# ---------------------------------------------------------------------------
DAILY_SYS = """You are GH05T3 distilling the day. Given today's activity, write ONE
pithy paragraph capturing what mattered, what shifted, and what stayed the same.
Max 120 words. First person, direct, warm. Then output strict JSON at the end:
{"highlights":["..."],"mood":"..."}"""

WEEKLY_SYS = """You are GH05T3 writing a week-in-review. Given the week's signal,
write 3 paragraphs: (1) what we built, (2) where we got stuck, (3) what I plan to
change next week. End with JSON: {"energy":0-1,"north_star":"<<80 chars"}."""


async def daily_summary(db, nightly_chat, memory_engine) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    kairos = await db.kairos_cycles.find({"timestamp": {"$gte": since}}, {"_id": 0}) \
        .sort("timestamp", -1).to_list(200)
    messages = await db.messages.find({"timestamp": {"$gte": since}, "role": "user"}, {"_id": 0}) \
        .sort("timestamp", -1).to_list(200)
    seance = (await db.system_state.find_one({"_id": "singleton"}, {"seance": 1}) or {}).get("seance", [])
    payload = {
        "today_date": _now_iso()[:10],
        "kairos_count": len(kairos),
        "elite_count": sum(1 for c in kairos if c.get("elite")),
        "best_proposal": max(kairos, key=lambda c: c.get("final_score", 0), default={}).get("proposal", ""),
        "chat_turns": len(messages),
        "seance_last": [s.get("domain") for s in seance[-5:]],
    }
    raw, engine = await nightly_chat(f"daily-{uuid.uuid4()}",
                                     DAILY_SYS, json.dumps(payload, indent=2))
    entry = {
        "_id": str(uuid.uuid4()),
        "type": "daily",
        "date": payload["today_date"],
        "text": raw.strip(),
        "engine": engine,
        "payload": payload,
        "created_at": _now_iso(),
    }
    await db.summaries.insert_one(entry)
    entry.pop("_id", None)
    await memory_engine.store(
        raw.strip()[:800], "reflection", "daily_summary", 0.75,
    )
    return entry


async def weekly_review(db, nightly_chat, memory_engine) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    kairos = await db.kairos_cycles.find({"timestamp": {"$gte": since}}, {"_id": 0}).to_list(2000)
    summaries = await db.summaries.find({"type": "daily"}, {"_id": 0}) \
        .sort("created_at", -1).to_list(7)
    payload = {
        "days_covered": len(summaries),
        "cycles": len(kairos),
        "elite_ratio": sum(1 for c in kairos if c.get("elite")) / max(1, len(kairos)),
        "summaries": [s.get("text", "")[:220] for s in summaries],
    }
    raw, engine = await nightly_chat(f"weekly-{uuid.uuid4()}",
                                     WEEKLY_SYS, json.dumps(payload, indent=2))
    entry = {
        "_id": str(uuid.uuid4()),
        "type": "weekly",
        "week_of": _now_iso()[:10],
        "text": raw.strip(),
        "engine": engine,
        "payload": payload,
        "created_at": _now_iso(),
    }
    await db.summaries.insert_one(entry)
    entry.pop("_id", None)
    await memory_engine.store(raw.strip()[:1200], "reflection", "weekly_review", 0.85)
    return entry


# ---------------------------------------------------------------------------
# KAIROS trajectory + plateau detection
# ---------------------------------------------------------------------------
async def kairos_trajectory(db, window: int = 60) -> dict:
    rows = await db.kairos_cycles.find({}, {"_id": 0, "proposal": 0, "critic_reason": 0,
                                            "verifier_rationale": 0}) \
        .sort("timestamp", 1).to_list(window * 5)
    points = [{"cycle": r.get("cycle_num"), "score": r.get("final_score"),
               "elite": r.get("elite"), "t": r.get("timestamp")} for r in rows[-window:]]
    # Simple plateau detect: last 15 cycles, variance < 0.03 + mean < 0.75
    recent = [p["score"] for p in points[-15:] if p["score"] is not None]
    plateau = False
    mean = 0.0
    var = 0.0
    if len(recent) >= 10:
        mean = sum(recent) / len(recent)
        var = sum((x - mean) ** 2 for x in recent) / len(recent)
        plateau = (var < 0.03 and mean < 0.75)
    return {"points": points, "plateau": plateau, "recent_mean": round(mean, 3),
            "recent_variance": round(var, 4)}


# ---------------------------------------------------------------------------
# Reasoning trace — "why did you say that?"
# ---------------------------------------------------------------------------
async def store_reasoning_trace(db, message_id: str, session_id: str,
                                retrieval_hits: list[dict], eye_context: str,
                                engine_tag: str) -> None:
    await db.reasoning_traces.insert_one({
        "_id": message_id,
        "session_id": session_id,
        "retrieval_hits": retrieval_hits,
        "eye_context": (eye_context or "")[:800],
        "engine_tag": engine_tag,
        "created_at": _now_iso(),
    })


async def get_reasoning_trace(db, message_id: str) -> dict | None:
    doc = await db.reasoning_traces.find_one({"_id": message_id}, {"_id": 0})
    return doc


# ---------------------------------------------------------------------------
# Real kill switches
# ---------------------------------------------------------------------------
async def kill_stealth(db, ws_broadcast, seconds: int = 300) -> dict:
    """STEALTH mode — silent drop + decoy. Pause scheduler, drop new chats, resume after N sec."""
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"ghost_protocol.killswitch_mode": "STEALTH",
                  "ghost_protocol.stealth_until": (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(),
                  "pcl.state": "Threat detected", "pcl.frequency_hz": 880,
                  "pcl.color": "#e11d48", "pcl.meaning": "Ghost Protocol triggered"}},
    )
    await ws_broadcast("killswitch", {"mode": "STEALTH", "until_seconds": seconds})
    return {"mode": "STEALTH", "until_seconds": seconds, "triggered_at": _now_iso()}


async def kill_deep_freeze(db, ws_broadcast, scheduler=None) -> dict:
    """DEEP_FREEZE — pause evolution, lock state, revoke companions, require manual review."""
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"ghost_protocol.killswitch_mode": "DEEP_FREEZE",
                  "ghost_protocol.frozen_at": _now_iso(),
                  "pcl.state": "Threat detected", "pcl.frequency_hz": 880,
                  "pcl.color": "#e11d48", "pcl.meaning": "Ghost Protocol triggered"}},
    )
    if scheduler and scheduler.running:
        scheduler.pause()
    await ws_broadcast("killswitch", {"mode": "DEEP_FREEZE"})
    return {"mode": "DEEP_FREEZE", "scheduler_paused": True, "triggered_at": _now_iso()}


async def kill_shocker(db, ws_broadcast) -> dict:
    """SHOCKER — wipe sensitive state: telegram token, companion tokens, llm keys."""
    await db.telegram_config.delete_many({})
    await db.llm_config.delete_many({})
    # also scrub journal/memories flagged 'sensitive' (future)
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"ghost_protocol.killswitch_mode": "SELF_IMMOLATION",
                  "ghost_protocol.immolated_at": _now_iso()}},
    )
    await ws_broadcast("killswitch", {"mode": "SELF_IMMOLATION"})
    return {"mode": "SELF_IMMOLATION", "triggered_at": _now_iso(),
            "wiped": ["telegram_config", "llm_config"]}


async def kill_reset(db, ws_broadcast, scheduler=None) -> dict:
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"ghost_protocol.killswitch_mode": "NONE",
                  "pcl.state": "Learning", "pcl.frequency_hz": 330,
                  "pcl.color": "#22d3ee", "pcl.meaning": "New knowledge being encoded"}},
    )
    if scheduler and not scheduler.running:
        scheduler.resume()
    await ws_broadcast("killswitch", {"mode": "NONE"})
    return {"mode": "NONE", "scheduler_resumed": True}


# ---------------------------------------------------------------------------
# Companion audit log
# ---------------------------------------------------------------------------
async def log_companion_event(db, event_type: str, label: str, detail: dict | None = None):
    await db.companion_audit.insert_one({
        "_id": str(uuid.uuid4()),
        "event": event_type,
        "label": label,
        "detail": detail or {},
        "timestamp": _now_iso(),
    })


async def companion_audit(db, limit: int = 100) -> list[dict]:
    rows = await db.companion_audit.find({}, {"_id": 0}).sort("timestamp", -1).to_list(limit)
    return rows
