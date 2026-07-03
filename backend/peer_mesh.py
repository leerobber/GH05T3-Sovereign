"""
GH05T3 PeerMesh — multi-instance knowledge sync over HTTP.

Each GH05T3 node maintains a list of peer URLs.
Syncs: memories (confidence >= 0.75), autotelic goals (last-writer-wins),
KAIROS elite cycles (union), and Seance lessons (union by domain).

Env vars (set in backend/.env):
  INSTANCE_LABEL   human name for this node  (default: hostname)
  INSTANCE_ROLE    primary | peer            (default: peer)
  INSTANCE_URL     URL other peers reach us at  e.g. http://10.x.x.x:8001
  PEER_URLS        comma-separated peer base URLs
  SYNC_INTERVAL    seconds between auto-syncs (default: 300)
"""
from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import platform
import time
from datetime import datetime, timezone
from typing import Optional

LOG = logging.getLogger("ghost.peers")

SYNC_MIN_CONFIDENCE = 0.75
SYNC_MAX_MEMORIES   = 150

try:
    import httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False
    LOG.warning("httpx not installed — peer sync disabled")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class PeerInfo:
    def __init__(self, url: str, label: str, role: str = "peer"):
        self.url         = url.rstrip("/")
        self.label       = label
        self.role        = role
        self.online      = False
        self.last_seen:  Optional[str]   = None
        self.latency_ms: Optional[float] = None
        self.last_sync:  Optional[str]   = None
        self.sync_sent   = 0
        self.sync_recv   = 0

    def to_dict(self) -> dict:
        return {
            "url":        self.url,
            "label":      self.label,
            "role":       self.role,
            "online":     self.online,
            "last_seen":  self.last_seen,
            "latency_ms": self.latency_ms,
            "last_sync":  self.last_sync,
            "sync_sent":  self.sync_sent,
            "sync_recv":  self.sync_recv,
        }


class PeerMesh:
    def __init__(
        self,
        db,
        ws,
        self_url:   str,
        self_label: str,
        self_role:  str = "peer",
    ):
        self.db         = db
        self.ws         = ws
        self.self_url   = self_url.rstrip("/")
        self.self_label = self_label
        self.self_role  = self_role
        self._peers:     dict[str, PeerInfo] = {}
        self._sync_task: Optional[asyncio.Task] = None

    # ── peer registry ────────────────────────────────────────────────────────

    def add_peer(self, url: str, label: str, role: str = "peer") -> Optional[PeerInfo]:
        url = url.rstrip("/")
        if url == self.self_url:
            return None
        if url in self._peers:
            self._peers[url].label = label
            self._peers[url].role  = role
            return self._peers[url]
        p = PeerInfo(url, label, role)
        self._peers[url] = p
        LOG.info("PeerMesh: registered peer %s @ %s", label, url)
        return p

    def remove_peer(self, url: str):
        self._peers.pop(url.rstrip("/"), None)

    @property
    def peers(self) -> list[dict]:
        return [p.to_dict() for p in self._peers.values()]

    @property
    def online_peers(self) -> list[PeerInfo]:
        return [p for p in self._peers.values() if p.online]

    def self_info(self) -> dict:
        return {
            "url":          self.self_url,
            "label":        self.self_label,
            "role":         self.self_role,
            "peers_known":  len(self._peers),
            "peers_online": len(self.online_peers),
        }

    # ── health check ─────────────────────────────────────────────────────────

    async def ping(self, peer: PeerInfo) -> bool:
        if not _HTTPX:
            return False
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{peer.url}/api/health")
                ok = r.status_code == 200
                if ok:
                    data = r.json()
                    # update label/role from remote self-info if available
                    info = data.get("self", {})
                    if info.get("label"):
                        peer.label = info["label"]
                    if info.get("role"):
                        peer.role = info["role"]
        except Exception:
            ok = False
        peer.online     = ok
        peer.latency_ms = round((time.monotonic() - t0) * 1000, 1)
        if ok:
            peer.last_seen = _now_iso()
        return ok

    async def ping_all(self):
        if self._peers:
            await asyncio.gather(
                *(self.ping(p) for p in self._peers.values()),
                return_exceptions=True,
            )

    # ── build outbound payload ────────────────────────────────────────────────

    async def _build_payload(self, since: Optional[str] = None) -> dict:
        payload: dict = {
            "from_label": self.self_label,
            "from_url":   self.self_url,
            "from_role":  self.self_role,
            "timestamp":  _now_iso(),
            "memories":   [],
            "goals":      [],
            "kairos_elite": [],
            "seance":     [],
        }

        try:
            query: dict = {"confidence": {"$gte": SYNC_MIN_CONFIDENCE}}
            if since:
                query["timestamp"] = {"$gt": since}
            mems = await (
                self.db.memories
                .find(query, {"_id": 0})
                .sort("confidence", -1)
                .limit(SYNC_MAX_MEMORIES)
                .to_list(SYNC_MAX_MEMORIES)
            )
            payload["memories"] = mems
        except Exception:
            pass

        try:
            doc = await self.db.system_state.find_one(
                {"_id": "singleton"}, {"autotelic_goals": 1, "seance": 1})
            if doc:
                payload["goals"]  = doc.get("autotelic_goals", [])
                payload["seance"] = doc.get("seance", [])
        except Exception:
            pass

        try:
            doc = await self.db.kairos_cycles.find(
                {"is_elite": True}, {"_id": 0}
            ).sort("timestamp", -1).limit(50).to_list(50)
            payload["kairos_elite"] = doc
        except Exception:
            pass

        return payload

    # ── apply inbound payload ─────────────────────────────────────────────────

    async def apply_payload(self, payload: dict) -> dict:
        counts = {"memories": 0, "goals": 0, "seance": 0, "kairos": 0}
        src = payload.get("from_label", "unknown")

        # memories — union by content hash
        try:
            for m in payload.get("memories", []):
                content = m.get("content", "")
                if not content:
                    continue
                h = _content_hash(content)
                existing = await self.db.memories.find_one({"content_hash": h})
                if not existing:
                    m["content_hash"]  = h
                    m["synced_from"]   = src
                    await self.db.memories.insert_one(m)
                    counts["memories"] += 1
        except Exception as e:
            LOG.warning("memory sync error: %s", e)

        # goals — last updated_at wins per ID
        try:
            incoming_goals = payload.get("goals", [])
            if incoming_goals:
                doc = await self.db.system_state.find_one(
                    {"_id": "singleton"}, {"autotelic_goals": 1})
                existing = {
                    g["id"]: g
                    for g in (doc or {}).get("autotelic_goals", [])
                    if g.get("id")
                }
                changed = False
                for g in incoming_goals:
                    gid = g.get("id")
                    if not gid:
                        continue
                    if (gid not in existing or
                            g.get("updated_at", "") > existing[gid].get("updated_at", "")):
                        existing[gid] = g
                        counts["goals"] += 1
                        changed = True
                if changed:
                    await self.db.system_state.update_one(
                        {"_id": "singleton"},
                        {"$set": {"autotelic_goals": list(existing.values())}},
                    )
        except Exception as e:
            LOG.warning("goal sync error: %s", e)

        # seance — union by domain
        try:
            incoming_seance = payload.get("seance", [])
            if incoming_seance:
                doc = await self.db.system_state.find_one(
                    {"_id": "singleton"}, {"seance": 1})
                existing_domains = {
                    s["domain"]
                    for s in (doc or {}).get("seance", [])
                    if s.get("domain")
                }
                new_entries = [
                    s for s in incoming_seance
                    if s.get("domain") and s["domain"] not in existing_domains
                ]
                if new_entries:
                    await self.db.system_state.update_one(
                        {"_id": "singleton"},
                        {"$push": {"seance": {"$each": new_entries, "$slice": -40}}},
                    )
                    counts["seance"] = len(new_entries)
        except Exception as e:
            LOG.warning("seance sync error: %s", e)

        # kairos elite — union by proposal hash
        try:
            for cycle in payload.get("kairos_elite", []):
                h = _content_hash(cycle.get("proposal", ""))
                existing = await self.db.kairos_cycles.find_one({"proposal_hash": h})
                if not existing:
                    cycle["proposal_hash"] = h
                    cycle["synced_from"]   = src
                    await self.db.kairos_cycles.insert_one(cycle)
                    counts["kairos"] += 1
        except Exception as e:
            LOG.warning("kairos sync error: %s", e)

        LOG.info("Applied sync from %s: %s", src, counts)
        return counts

    # ── sync with one peer ────────────────────────────────────────────────────

    async def push_to(self, peer: PeerInfo) -> Optional[dict]:
        if not _HTTPX:
            return None
        payload = await self._build_payload(since=peer.last_sync)
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(
                    f"{peer.url}/api/peers/sync",
                    json=payload,
                    headers={"X-GH05T3-From": self.self_label},
                )
                r.raise_for_status()
                result = r.json()
                peer.last_sync  = payload["timestamp"]
                peer.sync_sent += sum((result.get("applied") or {}).values())
                return result
        except Exception as e:
            LOG.warning("Push to %s failed: %s", peer.label, e)
            return None

    async def sync_peer(self, peer: PeerInfo):
        online = await self.ping(peer)
        if not online:
            return
        result = await self.push_to(peer)
        await self.ws.broadcast("peer_synced", {
            "label":  peer.label,
            "url":    peer.url,
            "result": result,
        })

    async def sync_all(self):
        for peer in list(self._peers.values()):
            try:
                await self.sync_peer(peer)
            except Exception as e:
                LOG.warning("sync_peer %s: %s", peer.label, e)

    # ── auto-sync loop ────────────────────────────────────────────────────────

    def start_auto_sync(self, interval: int = 300):
        async def _loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.sync_all()
                except Exception as e:
                    LOG.warning("auto-sync error: %s", e)
        self._sync_task = asyncio.create_task(_loop())
        LOG.info("PeerMesh auto-sync every %ds", interval)

    def stop(self):
        if self._sync_task:
            self._sync_task.cancel()
            self._sync_task = None
