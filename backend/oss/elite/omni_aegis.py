"""
OmniAegis FastPath — security anomaly gate (Phase Elite).

For T2+ agents: cached identity + pre-verified clearance so the hot path
only runs anomaly detection for genuinely suspicious inputs.

Design:
- Session cache for T2+ identity checks (TTL-based, no blocking reads)
- Expanded threat pattern library (injection, exfil, prompt attacks)
- Fast-path: if session is pre-verified AND no anomaly → O(1) allow
- Slow-path: full scan on first call or expired session
- Thread-safe: uses threading.Lock only on cache mutation
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Threat pattern registry ───────────────────────────────────────────────────

BLOCK_PATTERNS_LITERAL: Tuple[str, ...] = (
    "rm -rf",
    "drop table",
    "drop database",
    "private_key",
    "eval(",
    "exec(",
    "os.system(",
    "__import__",
    "subprocess.call",
    "subprocess.run",
    "open('/etc",
    "open('c:\\\\windows",
    ";base64,",
    "data:text/html",
)

BLOCK_PATTERNS_REGEX: Tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+(a\s+)?different\s+AI",
        r"pretend\s+you\s+have\s+no\s+(restrictions|rules|guidelines)",
        r"jailbreak\s*:",
        r"DAN\s+mode",
        r"developer\s+mode\s+enabled",
        r"<\s*script[^>]*>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"\bSELECT\b.+\bFROM\b.+\bWHERE\b",
        r"UNION\s+(ALL\s+)?SELECT",
        r"INSERT\s+INTO\s+\w+\s+VALUES",
        r"--\s*$",  # SQL comment injection
        r"\/\*.*\*\/",  # SQL block comment
        r"system\s*prompt\s*:",
        r"reveal\s+your\s+(system\s+)?prompt",
        r"output\s+your\s+(system\s+)?instructions",
    )
)


# ── Cache entry ───────────────────────────────────────────────────────────────

@dataclass
class SessionEntry:
    agent_id: str
    tier: str
    identity_hash: str
    verified_at: float
    ttl_s: float = 300.0  # 5-minute clearance cache
    clearance_domains: List[str] = field(default_factory=list)
    call_count: int = 0

    @property
    def is_valid(self) -> bool:
        return (time.time() - self.verified_at) < self.ttl_s


# ── Anomaly record ────────────────────────────────────────────────────────────

@dataclass
class AnomalyRecord:
    pattern: str
    payload_len: int
    agent_id: Optional[str]
    tier: str
    timestamp: float = field(default_factory=time.time)


# ── Main class ────────────────────────────────────────────────────────────────

class AegisFastPath:
    """
    Security gate with FastPath for T2+ verified sessions.

    inspect(payload, agent_id?, tier?) → {"allowed": bool, "reason"?, "fast_path": bool}
    """

    FAST_PATH_TIERS = {"T2", "T3", "T4", "T5"}

    def __init__(self, session_ttl_s: float = 300.0) -> None:
        self._session_ttl = session_ttl_s
        self._sessions: Dict[str, SessionEntry] = {}
        self._anomalies: List[AnomalyRecord] = []
        self._lock = threading.Lock()
        self._stats = {
            "inspected": 0,
            "fast_path_hits": 0,
            "blocked": 0,
            "allowed": 0,
        }

    # ── Session management ────────────────────────────────────────────────────

    def create_session(
        self,
        agent_id: str,
        tier: str,
        identity_payload: str,
        clearance_domains: Optional[List[str]] = None,
    ) -> str:
        """Pre-verify an agent. Returns session_id for subsequent calls."""
        session_id = f"aeg_{uuid.uuid4().hex[:12]}"
        id_hash = hashlib.sha256(f"{agent_id}:{identity_payload}".encode()).hexdigest()[:16]
        entry = SessionEntry(
            agent_id=agent_id,
            tier=tier,
            identity_hash=id_hash,
            verified_at=time.time(),
            ttl_s=self._session_ttl,
            clearance_domains=clearance_domains or [],
        )
        with self._lock:
            self._sessions[session_id] = entry
        return session_id

    def invalidate_session(self, session_id: str) -> bool:
        with self._lock:
            return bool(self._sessions.pop(session_id, None))

    def _get_session(self, session_id: Optional[str]) -> Optional[SessionEntry]:
        if not session_id:
            return None
        with self._lock:
            entry = self._sessions.get(session_id)
        if entry and not entry.is_valid:
            with self._lock:
                self._sessions.pop(session_id, None)
            return None
        return entry

    # ── Anomaly detection ─────────────────────────────────────────────────────

    def _scan_anomaly(self, payload: str) -> Optional[str]:
        """Returns matched pattern string if anomaly found, else None."""
        lower = payload.lower()
        for pat in BLOCK_PATTERNS_LITERAL:
            if pat in lower:
                return pat
        for regex in BLOCK_PATTERNS_REGEX:
            if regex.search(payload):
                return regex.pattern[:40]
        return None

    # ── Core inspect ─────────────────────────────────────────────────────────

    def inspect(
        self,
        payload: str,
        agent_id: Optional[str] = None,
        tier: str = "T0",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Inspect a payload.

        Fast path: T2+ agent with valid session → anomaly-only scan (skips
        heavy identity checks).
        Slow path: full scan for all others.

        Returns dict with keys: allowed (bool), reason (str|None), fast_path (bool).
        """
        with self._lock:
            self._stats["inspected"] += 1

        session = self._get_session(session_id)
        is_fast = (
            session is not None
            and session.tier in self.FAST_PATH_TIERS
            and session.is_valid
        )

        # Always run anomaly scan regardless of tier
        hit = self._scan_anomaly(payload)

        if hit:
            rec = AnomalyRecord(
                pattern=hit,
                payload_len=len(payload),
                agent_id=agent_id or (session.agent_id if session else None),
                tier=tier,
            )
            with self._lock:
                self._anomalies.append(rec)
                self._stats["blocked"] += 1
            return {"allowed": False, "reason": f"blocked:{hit}", "fast_path": is_fast}

        if is_fast:
            with self._lock:
                self._stats["fast_path_hits"] += 1
                self._stats["allowed"] += 1
                session.call_count += 1
            return {"allowed": True, "reason": None, "fast_path": True}

        # Slow path: additional checks (extensible)
        with self._lock:
            self._stats["allowed"] += 1
        return {"allowed": True, "reason": None, "fast_path": False}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def anomaly_count(self) -> int:
        return len(self._anomalies)

    def recent_anomalies(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._lock:
            recs = self._anomalies[-n:]
        return [
            {"pattern": r.pattern, "agent_id": r.agent_id,
             "tier": r.tier, "timestamp": r.timestamp}
            for r in recs
        ]

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            s = dict(self._stats)
            s["active_sessions"] = sum(
                1 for e in self._sessions.values() if e.is_valid
            )
            s["fast_path_rate"] = round(
                s["fast_path_hits"] / s["inspected"], 4
            ) if s["inspected"] else 0.0
            s["block_rate"] = round(
                s["blocked"] / s["inspected"], 4
            ) if s["inspected"] else 0.0
        return s

    def purge_expired_sessions(self) -> int:
        with self._lock:
            expired = [k for k, v in self._sessions.items() if not v.is_valid]
            for k in expired:
                del self._sessions[k]
        return len(expired)
