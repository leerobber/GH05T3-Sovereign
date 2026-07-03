"""Persistent memory layer for site agent knowledge — SQLite, sensitive data safe."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

LOG = logging.getLogger("site_agents.memory")

DB_PATH = Path(__file__).parent.parent / "memory" / "palace.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _ensure_tables() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS site_knowledge (
                id          TEXT PRIMARY KEY,
                agent       TEXT NOT NULL,
                category    TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                tags        TEXT DEFAULT '[]',
                confidence  REAL DEFAULT 1.0,
                timestamp   REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS sk_agent ON site_knowledge(agent)")
        c.execute("CREATE INDEX IF NOT EXISTS sk_category ON site_knowledge(category)")
        c.execute("CREATE INDEX IF NOT EXISTS sk_key ON site_knowledge(key)")

        c.execute("""
            CREATE TABLE IF NOT EXISTS site_tasks (
                id          TEXT PRIMARY KEY,
                agent       TEXT NOT NULL,
                task_type   TEXT NOT NULL,
                input       TEXT,
                output      TEXT,
                status      TEXT DEFAULT 'pending',
                score       REAL DEFAULT 0.0,
                timestamp   REAL NOT NULL
            )
        """)


_ensure_tables()


def _ensure_email_schedule() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS email_schedule (
                id          TEXT PRIMARY KEY,
                user_email  TEXT NOT NULL,
                first_name  TEXT DEFAULT '',
                email_num   INTEGER NOT NULL,
                send_after  REAL NOT NULL,
                sent_at     REAL DEFAULT NULL,
                error       TEXT DEFAULT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS es_pending ON email_schedule(send_after) WHERE sent_at IS NULL")


_ensure_email_schedule()


def schedule_email(user_email: str, first_name: str, email_num: int, send_after: float) -> str:
    import hashlib
    eid = hashlib.md5(f"{user_email}:{email_num}".encode()).hexdigest()[:16]
    with _conn() as c:
        c.execute("""
            INSERT OR IGNORE INTO email_schedule (id, user_email, first_name, email_num, send_after)
            VALUES (?, ?, ?, ?, ?)
        """, (eid, user_email, first_name, email_num, send_after))
    return eid


def get_due_emails(now: float | None = None) -> list[dict]:
    t = now or time.time()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM email_schedule WHERE sent_at IS NULL AND send_after <= ? ORDER BY send_after",
            (t,),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_email_sent(eid: str, error: str | None = None) -> None:
    with _conn() as c:
        if error:
            c.execute("UPDATE email_schedule SET sent_at = ?, error = ? WHERE id = ?",
                      (time.time(), error[:500], eid))
        else:
            c.execute("UPDATE email_schedule SET sent_at = ? WHERE id = ?", (time.time(), eid))


def store(
    agent: str,
    category: str,
    key: str,
    value: Any,
    tags: list[str] | None = None,
    confidence: float = 1.0,
) -> str:
    """Store a knowledge item. Returns the record ID."""
    import hashlib
    doc_id = hashlib.md5(f"{agent}:{category}:{key}".encode()).hexdigest()[:16]
    with _conn() as c:
        c.execute("""
            INSERT OR REPLACE INTO site_knowledge
              (id, agent, category, key, value, tags, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id, agent, category, key,
            json.dumps(value) if not isinstance(value, str) else value,
            json.dumps(tags or []),
            confidence,
            time.time(),
        ))
    return doc_id


def recall(
    agent: str | None = None,
    category: str | None = None,
    key: str | None = None,
    limit: int = 20,
) -> list[dict]:
    clauses, params = [], []
    if agent:
        clauses.append("agent = ?"); params.append(agent)
    if category:
        clauses.append("category = ?"); params.append(category)
    if key:
        clauses.append("key LIKE ?"); params.append(f"%{key}%")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM site_knowledge {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def log_task(
    agent: str,
    task_type: str,
    input_data: str,
    output_data: str,
    status: str = "done",
    score: float = 0.0,
) -> str:
    import hashlib
    task_id = hashlib.md5(f"{agent}:{task_type}:{time.time()}".encode()).hexdigest()[:16]
    with _conn() as c:
        c.execute("""
            INSERT INTO site_tasks (id, agent, task_type, input, output, status, score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (task_id, agent, task_type, input_data[:2000], output_data[:4000], status, score, time.time()))
    return task_id


def recent_tasks(agent: str | None = None, limit: int = 10) -> list[dict]:
    where = "WHERE agent = ?" if agent else ""
    params = [agent] if agent else []
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM site_tasks {where} ORDER BY timestamp DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def delete_by_category(category: str) -> int:
    with _conn() as c:
        cur = c.execute("DELETE FROM site_knowledge WHERE category = ?", (category,))
        return cur.rowcount
