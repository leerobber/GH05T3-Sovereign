"""
BreakthroughDetector — real-time breakthrough detection with SQLite persistence.

A breakthrough is flagged when an agent's task output clears simultaneous
thresholds on novelty, impact, and rarity. Detections are persisted to SQLite
so they can be queried by live monitoring endpoints (/breakthroughs).

Integration points:
  - Called per-task in OmniSentientEcosystem.run_cycle()
  - Queried by the monitoring API (GET /oss/breakthroughs)
  - Optionally wired to FrontierResearchLab domain scoring
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("ghost.breakthrough_detector")
_UTC = lambda: datetime.now(timezone.utc).isoformat()

_DB_PATH = Path("data/breakthroughs.db")


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Breakthrough:
    breakthrough_id: str
    agent_id: str
    description: str
    novelty_score: float
    impact_score: float
    rarity_score: float
    fitness: float
    domain: str = "general"
    verified: bool = False
    timestamp: str = field(default_factory=_UTC)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "breakthrough_id": self.breakthrough_id,
            "agent_id":        self.agent_id,
            "description":     self.description,
            "novelty_score":   round(self.novelty_score, 4),
            "impact_score":    round(self.impact_score, 4),
            "rarity_score":    round(self.rarity_score, 4),
            "fitness":         round(self.fitness, 4),
            "domain":          self.domain,
            "verified":        self.verified,
            "timestamp":       self.timestamp,
        }


# ── Detector ───────────────────────────────────────────────────────────────────

class BreakthroughDetector:
    """
    Detects and persists breakthroughs.

    Usage:
        bd = BreakthroughDetector()
        bt = bd.detect(agent_id, novelty=0.92, impact=0.80, rarity=0.85, fitness=0.78, description="...")
        if bt:
            print(bt.to_dict())
    """

    NOVELTY_THRESHOLD = 0.85
    IMPACT_THRESHOLD  = 0.75
    RARITY_THRESHOLD  = 0.75

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._recent: List[Breakthrough] = []   # in-memory ring buffer (last 200)

    # ── DB setup ───────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS breakthroughs (
                        breakthrough_id TEXT PRIMARY KEY,
                        agent_id        TEXT NOT NULL,
                        description     TEXT,
                        novelty_score   REAL,
                        impact_score    REAL,
                        rarity_score    REAL,
                        fitness         REAL,
                        domain          TEXT DEFAULT 'general',
                        verified        INTEGER DEFAULT 0,
                        timestamp       TEXT NOT NULL
                    )
                """)
        except Exception as exc:
            LOG.warning("breakthrough_detector: db init failed (non-fatal): %s", exc)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect(
        self,
        agent_id: str,
        novelty_score: float,
        impact_score: float,
        rarity_score: float,
        fitness: float,
        description: str = "",
        domain: str = "general",
    ) -> Optional[Breakthrough]:
        """
        Check if scores clear all three thresholds.
        If yes, create + persist a Breakthrough and return it; else return None.
        """
        if not (
            novelty_score >= self.NOVELTY_THRESHOLD
            and impact_score  >= self.IMPACT_THRESHOLD
            and rarity_score  >= self.RARITY_THRESHOLD
        ):
            return None

        bt = Breakthrough(
            breakthrough_id=f"bt_{uuid.uuid4().hex[:12]}",
            agent_id=agent_id,
            description=description[:200],
            novelty_score=novelty_score,
            impact_score=impact_score,
            rarity_score=rarity_score,
            fitness=fitness,
            domain=domain,
        )
        self._persist(bt)
        self._recent.append(bt)
        if len(self._recent) > 200:
            self._recent = self._recent[-200:]
        LOG.info(
            "BREAKTHROUGH: agent=%s novelty=%.3f impact=%.3f rarity=%.3f domain=%s",
            agent_id, novelty_score, impact_score, rarity_score, domain,
        )
        return bt

    def verify(self, breakthrough_id: str) -> bool:
        """Mark a breakthrough as verified (human or secondary-agent confirmed)."""
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE breakthroughs SET verified=1 WHERE breakthrough_id=?",
                    (breakthrough_id,),
                )
            for bt in self._recent:
                if bt.breakthrough_id == breakthrough_id:
                    bt.verified = True
            return True
        except Exception as exc:
            LOG.warning("breakthrough verify failed: %s", exc)
            return False

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fast path: return the in-memory ring buffer (most recent first)."""
        return [bt.to_dict() for bt in reversed(self._recent[-limit:])]

    def query(
        self,
        limit: int = 50,
        domain: Optional[str] = None,
        verified_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Query SQLite for breakthroughs (slower but persistent across restarts)."""
        try:
            conditions, params = [], []
            if domain:
                conditions.append("domain = ?"); params.append(domain)
            if verified_only:
                conditions.append("verified = 1")
            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            params.append(limit)
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT * FROM breakthroughs {where} ORDER BY timestamp DESC LIMIT ?",
                    params,
                ).fetchall()
            cols = ["breakthrough_id","agent_id","description","novelty_score",
                    "impact_score","rarity_score","fitness","domain","verified","timestamp"]
            return [dict(zip(cols, r)) for r in rows]
        except Exception as exc:
            LOG.warning("breakthrough query failed: %s", exc)
            return self.get_recent(limit)

    def stats(self) -> Dict[str, Any]:
        """Summary statistics for monitoring endpoints."""
        try:
            with self._conn() as conn:
                total     = conn.execute("SELECT COUNT(*) FROM breakthroughs").fetchone()[0]
                verified  = conn.execute("SELECT COUNT(*) FROM breakthroughs WHERE verified=1").fetchone()[0]
                by_domain = {
                    row[0]: row[1]
                    for row in conn.execute(
                        "SELECT domain, COUNT(*) FROM breakthroughs GROUP BY domain"
                    ).fetchall()
                }
        except Exception:
            total, verified, by_domain = len(self._recent), 0, {}
        return {
            "total":      total,
            "verified":   verified,
            "unverified": total - verified,
            "by_domain":  by_domain,
            "in_memory":  len(self._recent),
        }

    def _persist(self, bt: Breakthrough) -> None:
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO breakthroughs
                       (breakthrough_id, agent_id, description, novelty_score,
                        impact_score, rarity_score, fitness, domain, verified, timestamp)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (bt.breakthrough_id, bt.agent_id, bt.description,
                     bt.novelty_score, bt.impact_score, bt.rarity_score,
                     bt.fitness, bt.domain, int(bt.verified), bt.timestamp),
                )
        except Exception as exc:
            LOG.debug("breakthrough persist failed (non-fatal): %s", exc)


# ── Singleton ──────────────────────────────────────────────────────────────────

_detector: Optional[BreakthroughDetector] = None

def get_breakthrough_detector() -> BreakthroughDetector:
    global _detector
    if _detector is None:
        _detector = BreakthroughDetector()
    return _detector
