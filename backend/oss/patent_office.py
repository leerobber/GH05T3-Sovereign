"""
PatentOffice (LEX-GEN) — autonomous IP-capture observer for the Aethyro swarm.

Architecture position:
  Auto-Distiller → [Canonical Memory + BreakthroughDetector]
                              ↓
                      PatentOffice (passive observer)
                              ↓
                   data/patent_portfolio.db  (SQLite, Control Plane)

The PatentOffice agent has its own LEX-GEN genome (7 patent molecules)
and runs as a passive observer loop — it never blocks the primary
execution plane.

Control Plane: patent_portfolio.db is the persistent IP store. It feeds
the "Patent Portfolio" UI on aethyro.com and triggers Stripe notifications
when viable disclosures are generated.

Execution Plane: LEX-GEN reads the ChronosLedger for swarm-wide novelty;
its own binary slot is also tracked in the ledger.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LOG = logging.getLogger("ghost.patent_office")

_DB_PATH   = Path("data/patent_portfolio.db")
_UTC       = lambda: datetime.now(timezone.utc).isoformat()
_NOVELTY_GATE = 0.85


# ── Patent Disclosure record ──────────────────────────────────────────────────

@dataclass
class PatentDisclosure:
    disclosure_id:    str = field(default_factory=lambda: f"PD_{uuid.uuid4().hex[:10].upper()}")
    title:            str = ""
    domain:           str = ""
    novelty_score:    float = 0.0
    impact_score:     float = 0.0
    rarity_score:     float = 0.0
    enabling_summary: str = ""
    claim_draft:      str = ""
    ip_score:         float = 0.0
    status:           str = "DRAFT"       # DRAFT | REVIEWED | FILED | REJECTED
    source_agent:     str = ""
    timestamp:        str = field(default_factory=_UTC)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disclosure_id":    self.disclosure_id,
            "title":            self.title,
            "domain":           self.domain,
            "novelty_score":    self.novelty_score,
            "impact_score":     self.impact_score,
            "rarity_score":     self.rarity_score,
            "enabling_summary": self.enabling_summary,
            "claim_draft":      self.claim_draft,
            "ip_score":         self.ip_score,
            "status":           self.status,
            "source_agent":     self.source_agent,
            "timestamp":        self.timestamp,
        }


# ── PatentOffice ──────────────────────────────────────────────────────────────

class PatentOffice:
    """
    Passive-observer patent agent.

    - Monitors BreakthroughDetector for novelty >= 0.85
    - Drafts disclosures using LEX-GEN IP scoring
    - Persists to SQLite patent_portfolio.db (Control Plane)
    - Notifies GenesisThread when IP is captured
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._portfolio: List[PatentDisclosure] = []
        self._lex_genome: Optional[Any] = None
        self._init_db()
        self._load_lex_genome()

    # ── LEX-GEN genome ────────────────────────────────────────────────────────

    def _load_lex_genome(self) -> None:
        try:
            from backend.oss.genomic.lex_gen import create_lex_gen_genome
            self._lex_genome = create_lex_gen_genome()
            LOG.info("PatentOffice: LEX-GEN genome initialised")
        except Exception as exc:
            LOG.debug("PatentOffice: could not load LEX genome: %s", exc)

    def _ip_score(self) -> float:
        if self._lex_genome is None:
            return 0.75
        try:
            from backend.oss.genomic.lex_gen import score_ip_potential
            return score_ip_potential(self._lex_genome)
        except Exception:
            return 0.75

    # ── Main scan cycle ───────────────────────────────────────────────────────

    def scan_cycle(self) -> List[PatentDisclosure]:
        """
        One full scan: find novelty events → filter → draft → persist.
        Returns list of new disclosures generated this cycle.
        """
        events = self.find_novelty_events()
        new_disclosures: List[PatentDisclosure] = []
        for event in events:
            if float(event.get("novelty_score", 0)) < _NOVELTY_GATE:
                continue
            disclosure = self.package_disclosure(event)
            self._save_disclosure(disclosure)
            new_disclosures.append(disclosure)
        return new_disclosures

    def find_novelty_events(self) -> List[Dict[str, Any]]:
        """Retrieve high-novelty breakthroughs from the detector."""
        try:
            from backend.oss.breakthrough_detector import get_breakthrough_detector
            bd = get_breakthrough_detector()
            return bd.get_recent(limit=20)
        except Exception as exc:
            LOG.debug("patent_office: detector error: %s", exc)
            return []

    def handle_breakthrough_signal(self, bt_dict: Dict[str, Any]) -> Optional[PatentDisclosure]:
        """Called by GenesisThread when a threshold crossing is detected."""
        if float(bt_dict.get("novelty_score", 0)) < _NOVELTY_GATE:
            return None
        disc = self.package_disclosure(bt_dict)
        self._save_disclosure(disc)
        self._seal_disclosure(disc, bt_dict)
        LOG.info(
            "PatentOffice: new disclosure %s — domain=%s novelty=%.3f ip_score=%.3f",
            disc.disclosure_id, disc.domain, disc.novelty_score, disc.ip_score,
        )
        return disc

    # ── Domain-agnostic detection ─────────────────────────────────────────────

    _DOMAIN_KEYWORDS: Dict[str, List[str]] = {
        "llm_architecture":    ["transformer", "attention", "llm", "language model", "token", "embedding"],
        "genomic_evolution":   ["genome", "mutation", "fitness", "desire", "evolution", "swarm", "locus"],
        "binary_computing":    ["mmap", "binary", "ledger", "struct", "memory", "simd", "float16"],
        "market_intelligence": ["market", "trade", "price", "volatility", "revenue", "yield", "alpha"],
        "cognitive_systems":   ["reasoning", "cognitive", "context", "synthesis", "pdac", "planning"],
        "ip_strategy":         ["patent", "disclosure", "novelty", "claim", "prior art", "ip", "filing"],
    }

    def _infer_domain(self, event: Dict[str, Any]) -> str:
        """
        Keyword-based domain inference from breakthrough description and metadata.
        Ranks by keyword hit count. Falls back to 'general' if no domain matches.
        """
        text = " ".join([
            str(event.get("description", "")),
            str(event.get("task_prompt",  "")),
            str(event.get("domain",       "")),
        ]).lower()

        best_domain  = "general"
        best_count   = 0
        for domain, keywords in self._DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count  = count
                best_domain = domain

        return best_domain

    def package_disclosure(self, event: Dict[str, Any]) -> PatentDisclosure:
        """Translate a breakthrough event into a structured patent disclosure."""
        novelty = float(event.get("novelty_score", 0))
        impact  = float(event.get("impact_score",  0))
        rarity  = float(event.get("rarity_score",  0))
        desc    = event.get("description", "")
        agent   = event.get("agent_id", "unknown")
        domain  = self._infer_domain(event)     # auto-inferred, not pre-assigned

        claim = (
            f"A method and system for autonomous genomic evolution comprising: "
            f"(a) a binary memory-mapped swarm ledger storing agent desire vectors; "
            f"(b) a dissent-adjusted fitness function using exponential reward for "
            f"population outliers; "
            f"(c) an asynchronous genomic distillation pipeline converting task metrics "
            f"into molecule corrections; wherein the method produces a breakthrough "
            f"event characterized by novelty≥{novelty:.2f}, impact≥{impact:.2f}, "
            f"rarity≥{rarity:.2f} in the domain of {domain!r}."
        )

        return PatentDisclosure(
            title=f"Autonomous Genomic Evolution System — {domain.replace('_', ' ').title()} Breakthrough",
            domain=domain,
            novelty_score=round(novelty, 4),
            impact_score=round(impact, 4),
            rarity_score=round(rarity, 4),
            enabling_summary=desc[:500],
            claim_draft=claim,
            ip_score=self._ip_score(),
            source_agent=agent,
        )

    # ── Seal integration ──────────────────────────────────────────────────────

    def _seal_disclosure(self, disc: PatentDisclosure, source_event: Dict[str, Any]) -> None:
        """Call LexGenSeal to create a SHA256-signed vault record for this disclosure."""
        try:
            from backend.oss.core.seal import get_lex_gen_seal
            seal   = get_lex_gen_seal()
            parent = source_event.get("parent_agent_id", "genesis")
            seal.seal_breakthrough(
                agent_id      = disc.source_agent,
                parent_id     = parent,
                data_snapshot = disc.to_dict(),
                timestamp     = disc.timestamp,
                disclosure_id = disc.disclosure_id,
            )
        except Exception as exc:
            LOG.debug("PatentOffice: seal skipped: %s", exc)

    # ── Portfolio API ─────────────────────────────────────────────────────────

    def get_portfolio(self, limit: int = 50, status: str = "") -> List[Dict[str, Any]]:
        try:
            with self._conn() as conn:
                if status:
                    rows = conn.execute(
                        "SELECT data FROM patent_disclosures WHERE status=? ORDER BY timestamp DESC LIMIT ?",
                        (status, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT data FROM patent_disclosures ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            return [json.loads(r[0]) for r in rows]
        except Exception:
            return [d.to_dict() for d in self._portfolio[-limit:]]

    def update_status(self, disclosure_id: str, status: str) -> bool:
        valid = {"DRAFT", "REVIEWED", "FILED", "REJECTED"}
        if status.upper() not in valid:
            return False
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE patent_disclosures SET status=? WHERE disclosure_id=?",
                    (status.upper(), disclosure_id),
                )
            return True
        except Exception:
            return False

    def stats(self) -> Dict[str, Any]:
        try:
            with self._conn() as conn:
                total = conn.execute("SELECT COUNT(*) FROM patent_disclosures").fetchone()[0]
                by_status = {
                    row[0]: row[1]
                    for row in conn.execute(
                        "SELECT status, COUNT(*) FROM patent_disclosures GROUP BY status"
                    ).fetchall()
                }
            return {"total": total, "by_status": by_status, "lex_ip_score": self._ip_score()}
        except Exception:
            return {"total": len(self._portfolio), "lex_ip_score": self._ip_score()}

    # ── SQLite Control Plane ──────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path))

    def _init_db(self) -> None:
        try:
            with self._conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS patent_disclosures (
                        disclosure_id TEXT PRIMARY KEY,
                        status        TEXT NOT NULL DEFAULT 'DRAFT',
                        novelty_score REAL,
                        ip_score      REAL,
                        domain        TEXT,
                        timestamp     TEXT NOT NULL,
                        data          TEXT NOT NULL
                    )
                """)
        except Exception as exc:
            LOG.debug("patent_office db init: %s", exc)

    def _save_disclosure(self, disc: PatentDisclosure) -> None:
        self._portfolio.append(disc)
        try:
            with self._conn() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO patent_disclosures
                       (disclosure_id, status, novelty_score, ip_score, domain, timestamp, data)
                       VALUES (?,?,?,?,?,?,?)""",
                    (disc.disclosure_id, disc.status, disc.novelty_score,
                     disc.ip_score, disc.domain, disc.timestamp,
                     json.dumps(disc.to_dict())),
                )
        except Exception as exc:
            LOG.debug("patent_office save: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

_office: Optional[PatentOffice] = None


def get_patent_office() -> PatentOffice:
    global _office
    if _office is None:
        _office = PatentOffice()
    return _office
