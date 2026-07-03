"""GH05T3 — Entropy Drift Tracker: per-agent intent deviation detection.

Computes cosine distance between each agent's output embedding and its stored
baseline embedding. Rising drift = the agent is mutating away from its defined
role. This is the D_ε term in the Sentinel Equation.

Design:
  - Baseline = element-wise mean of the first ENTROPY_BASELINE_WARMUP outputs
  - Embeddings via Ollama nomic-embed-text (768-dim) — same as MemoryPalace
  - Drift scores persisted to SQLite (appended to the existing palace.db)
  - Alert logged when drift > ENTROPY_DRIFT_ALERT (default 0.35)

Env vars:
    ENTROPY_BASELINE_WARMUP   samples before baseline is fixed  (default 10)
    ENTROPY_DRIFT_ALERT       drift threshold for warning log   (default 0.35)
    OLLAMA_EMBED_URL          Ollama embedding endpoint
    OLLAMA_EMBED_MODEL        embedding model                   (default nomic-embed-text)
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("ghost.entropy")

DB_PATH               = Path(os.environ.get("MEMORY_DB_PATH", "memory/palace.db"))
DRIFT_ALERT_THRESHOLD = float(os.environ.get("ENTROPY_DRIFT_ALERT", "0.35"))
BASELINE_WARMUP       = int(os.environ.get("ENTROPY_BASELINE_WARMUP", "10"))


# ---------------------------------------------------------------------------
# Pure-math helpers — no dependencies
# ---------------------------------------------------------------------------

def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance: 0.0 = identical direction, 1.0 = orthogonal, 2.0 = opposite."""
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - (dot / (na * nb))


def _vec_mean(vecs: list[list[float]]) -> list[float]:
    """Element-wise mean over a list of equal-length vectors."""
    if not vecs:
        return []
    n = len(vecs[0])
    return [sum(v[i] for v in vecs) / len(vecs) for i in range(n)]


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class EntropyDriftTracker:
    """Per-agent cosine drift tracking with SQLite persistence.

    Usage:
        tracker = EntropyDriftTracker()
        drift = await tracker.compute_drift(output_text, agent_id="ORACLE", cycle_id=42)
        # returns 0.0 during warmup, cosine distance after baseline is set
    """

    def __init__(self, db_path: Path = None):
        self._db          = db_path or DB_PATH
        self._db.parent.mkdir(parents=True, exist_ok=True)
        self._baselines:  dict[str, list[float]]       = {}   # agent_id → baseline vec
        self._warmup_buf: dict[str, list[list[float]]] = {}   # agent_id → warmup vecs
        self._init_db()
        self._load_baselines()

    # ── DB setup ─────────────────────────────────────────────────────────────

    def _conn(self):
        """Context manager that commits/rolls back AND closes the connection."""
        db_path = self._db

        class _Ctx:
            def __enter__(self):
                self._c = sqlite3.connect(db_path)
                self._c.__enter__()
                return self._c

            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    self._c.__exit__(exc_type, exc_val, exc_tb)
                finally:
                    self._c.close()

        return _Ctx()

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_baselines (
                    agent_id  TEXT    PRIMARY KEY,
                    vector    TEXT    NOT NULL,
                    updated   REAL    NOT NULL,
                    samples   INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_drift_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id  TEXT    NOT NULL,
                    drift     REAL    NOT NULL,
                    cycle_id  INTEGER DEFAULT 0,
                    timestamp REAL    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_drift_agent "
                "ON agent_drift_log(agent_id)"
            )

    def _load_baselines(self):
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT agent_id, vector FROM agent_baselines"
            ).fetchall()
        for agent_id, vec_json in rows:
            self._baselines[agent_id] = json.loads(vec_json)
        if self._baselines:
            LOG.info("[entropy] loaded baselines for: %s",
                     list(self._baselines.keys()))

    def _save_baseline(self, agent_id: str, vec: list[float], samples: int):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_baselines
                    (agent_id, vector, updated, samples)
                VALUES (?, ?, ?, ?)
            """, (agent_id, json.dumps(vec), time.time(), samples))
        self._baselines[agent_id] = vec

    def _log_drift(self, agent_id: str, drift: float, cycle_id: int):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO agent_drift_log "
                "(agent_id, drift, cycle_id, timestamp) VALUES (?, ?, ?, ?)",
                (agent_id, drift, cycle_id, time.time()),
            )
        if drift > DRIFT_ALERT_THRESHOLD:
            LOG.warning(
                "[entropy] %s drift=%.4f > alert=%.2f — possible role deviation",
                agent_id, drift, DRIFT_ALERT_THRESHOLD,
            )

    # ── Embedding ─────────────────────────────────────────────────────────────

    async def _embed(self, text: str) -> Optional[list[float]]:
        url   = os.environ.get("OLLAMA_EMBED_URL",
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
            LOG.debug("entropy embed failed: %s", e)
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def compute_drift(self, text: str, agent_id: str,
                            cycle_id: int = 0) -> float:
        """Embed text, update warmup or measure drift against baseline.

        Returns 0.0 during warmup (not enough samples to establish a baseline).
        Returns cosine distance [0.0, 1.0] once baseline is established.
        """
        vec = await self._embed(text)
        if not vec:
            return 0.0

        has_baseline = agent_id in self._baselines

        if not has_baseline:
            buf = self._warmup_buf.setdefault(agent_id, [])
            buf.append(vec)
            if len(buf) >= BASELINE_WARMUP:
                baseline = _vec_mean(buf)
                self._save_baseline(agent_id, baseline, len(buf))
                self._warmup_buf.pop(agent_id, None)
                LOG.info("[entropy] baseline established for %s (%d samples)",
                         agent_id, BASELINE_WARMUP)
            return 0.0

        drift = _cosine_distance(vec, self._baselines[agent_id])
        self._log_drift(agent_id, drift, cycle_id)
        return drift

    def reset_baseline(self, agent_id: str) -> bool:
        """Force re-calibration for an agent (e.g. after intentional role update)."""
        if agent_id not in self._baselines:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM agent_baselines WHERE agent_id=?", (agent_id,))
        del self._baselines[agent_id]
        self._warmup_buf.pop(agent_id, None)
        LOG.info("[entropy] baseline reset for %s — warmup will restart", agent_id)
        return True

    def drift_stats(self, agent_id: str = None, limit: int = 100) -> dict:
        with self._conn() as conn:
            if agent_id:
                rows = conn.execute(
                    "SELECT drift FROM agent_drift_log WHERE agent_id=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (agent_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT drift FROM agent_drift_log "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        drifts = [r[0] for r in rows]
        return {
            "agent_id":              agent_id or "all",
            "samples":               len(drifts),
            "avg_drift":             round(sum(drifts) / len(drifts), 4) if drifts else 0.0,
            "max_drift":             round(max(drifts), 4) if drifts else 0.0,
            "alert_threshold":       DRIFT_ALERT_THRESHOLD,
            "baselines_established": list(self._baselines.keys()),
            "agents_in_warmup":      list(self._warmup_buf.keys()),
        }
