"""
Shared Working Memory — ephemeral blackboard for multi-agent tasks.

SQLite-backed with TTL expiry so agents in a PDAC chain can share
intermediate results without re-deriving them. Falls back to an
in-memory dict if SQLite is unavailable.

Usage:
    swm = get_shared_memory()
    eid = swm.write(task_id, agent_id, content={"plan": ...}, entry_type="plan")
    entries = swm.read(task_id, entry_type="plan")
    swm.clear(task_id)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("ghost.shared_memory")
_UTC = lambda: datetime.now(timezone.utc).isoformat()
_DB_PATH = Path("data/shared_memory.db")


@dataclass
class MemoryEntry:
    entry_id:   str
    task_id:    str
    agent_id:   str
    content:    Dict[str, Any]
    entry_type: str
    ttl_seconds: int = 300
    timestamp:  str = field(default_factory=_UTC)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id":    self.entry_id,
            "task_id":     self.task_id,
            "agent_id":    self.agent_id,
            "content":     self.content,
            "entry_type":  self.entry_type,
            "ttl_seconds": self.ttl_seconds,
            "timestamp":   self.timestamp,
        }


class SharedWorkingMemory:
    """
    Ephemeral blackboard for multi-agent task chains.
    Prevents redundant context derivation by giving every agent in a
    PDAC sequence a live view of prior-phase results.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db  = Path(db_path) if db_path else _DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._mem: Dict[str, Dict[str, MemoryEntry]] = {}
        self._sql = self._init_db()

    def _init_db(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS swm_entries (
                        entry_id   TEXT PRIMARY KEY,
                        task_id    TEXT NOT NULL,
                        agent_id   TEXT NOT NULL,
                        content    TEXT NOT NULL,
                        entry_type TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        timestamp  TEXT NOT NULL
                    )
                """)
            return True
        except Exception as exc:
            LOG.warning("SharedWorkingMemory: sqlite init failed, using in-memory: %s", exc)
            return False

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db))

    def _now(self) -> float:
        return datetime.now(timezone.utc).timestamp()

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        task_id:     str,
        agent_id:    str,
        content:     Dict[str, Any],
        entry_type:  str = "result",
        ttl_seconds: int = 300,
    ) -> str:
        eid = f"swm_{uuid.uuid4().hex[:10]}"
        entry = MemoryEntry(
            entry_id=eid, task_id=task_id, agent_id=agent_id,
            content=content, entry_type=entry_type, ttl_seconds=ttl_seconds,
        )
        if self._sql:
            try:
                exp = self._now() + ttl_seconds
                with self._conn() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO swm_entries
                           (entry_id, task_id, agent_id, content, entry_type, expires_at, timestamp)
                           VALUES (?,?,?,?,?,?,?)""",
                        (eid, task_id, agent_id, json.dumps(content),
                         entry_type, exp, entry.timestamp),
                    )
                return eid
            except Exception as exc:
                LOG.debug("swm write sqlite failed, falling to mem: %s", exc)
        self._mem.setdefault(task_id, {})[eid] = entry
        return eid

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(
        self,
        task_id:    str,
        entry_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if self._sql:
            try:
                self._cleanup()
                params: List[Any] = [task_id, self._now()]
                clause = ""
                if entry_type:
                    clause = " AND entry_type=?"
                    params.append(entry_type)
                with self._conn() as conn:
                    rows = conn.execute(
                        f"SELECT entry_id, task_id, agent_id, content, entry_type, timestamp"
                        f" FROM swm_entries WHERE task_id=? AND expires_at>?{clause}"
                        f" ORDER BY timestamp ASC",
                        params,
                    ).fetchall()
                return [
                    MemoryEntry(
                        entry_id=r[0], task_id=r[1], agent_id=r[2],
                        content=json.loads(r[3]), entry_type=r[4], timestamp=r[5],
                    )
                    for r in rows
                ]
            except Exception as exc:
                LOG.debug("swm read sqlite failed: %s", exc)
        entries = list(self._mem.get(task_id, {}).values())
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        return entries

    def latest(self, task_id: str, entry_type: str) -> Optional[MemoryEntry]:
        entries = self.read(task_id, entry_type=entry_type)
        return entries[-1] if entries else None

    # ── Clear ─────────────────────────────────────────────────────────────────

    def clear(self, task_id: str) -> None:
        if self._sql:
            try:
                with self._conn() as conn:
                    conn.execute("DELETE FROM swm_entries WHERE task_id=?", (task_id,))
            except Exception:
                pass
        self._mem.pop(task_id, None)

    def summary(self, task_id: str) -> Dict[str, Any]:
        entries = self.read(task_id)
        return {
            "task_id":     task_id,
            "entry_count": len(entries),
            "types":       list({e.entry_type for e in entries}),
            "agents":      list({e.agent_id for e in entries}),
        }

    def _cleanup(self) -> None:
        try:
            with self._conn() as conn:
                conn.execute("DELETE FROM swm_entries WHERE expires_at<?", (self._now(),))
        except Exception:
            pass


# ── Singleton ─────────────────────────────────────────────────────────────────

_swm: Optional[SharedWorkingMemory] = None

def get_shared_memory() -> SharedWorkingMemory:
    global _swm
    if _swm is None:
        _swm = SharedWorkingMemory()
    return _swm
