"""
Auto-Distiller — converts rejected / low-quality agent outputs into genome
corrections stored in data/learnings.jsonl.

Every task cycle:
  1. Metrics are checked for known failure patterns.
  2. A matching "don't do X" rule is generated.
  3. The corresponding genome molecule is adjusted in place.
  4. The learning is appended to the JSONL log so it persists across restarts.

This is the missing feedback loop: memory that steers future behaviour.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import uuid

LOG = logging.getLogger("ghost.auto_distiller")
_UTC = lambda: datetime.now(timezone.utc).isoformat()
_LEARNINGS_PATH = Path("data/learnings.jsonl")

# ── Molecule → locus resolution ───────────────────────────────────────────────
# Prefix-based map so new molecules added to existing loci are auto-resolved.

_PREFIX_TO_LOCUS: List[Tuple[str, str]] = [
    ("M_CONTEXT_",    "context"),
    ("M_DESIRE_",     "desire"),
    ("M_REASONING_",  "cognitive"), ("M_PATTERN_",   "cognitive"),
    ("M_LEARNING_",   "cognitive"), ("M_ADAPTAB",    "cognitive"),
    ("M_RISK_",       "market"),    ("M_TREND_",     "market"),
    ("M_PRICING_",    "market"),
    ("M_CONSISTENCY", "loyalty"),   ("M_CONTRIBUTION", "loyalty"),
    ("M_ALIGNMENT",   "loyalty"),
    ("M_VISUAL_",     "psychology"), ("M_COLOR_",   "psychology"),
    ("M_LAYOUT_",     "psychology"), ("M_CURIOSITY", "psychology"),
    ("M_TRUST_",      "psychology"), ("M_EMOTIONAL", "psychology"),
    ("M_SCARCITY_",   "psychology"), ("M_VALUE_",   "psychology"),
    ("M_IDENTITY_",   "psychology"), ("M_PREMIUM_", "psychology"),
    ("M_COHERENCE",   "psychology"), ("M_MEMORAB",  "psychology"),
]


def _locus_for(molecule_id: str) -> Optional[str]:
    uid = molecule_id.upper()
    for prefix, locus in _PREFIX_TO_LOCUS:
        if uid.startswith(prefix.upper()):
            return locus
    return None


# ── Issue → correction mapping ────────────────────────────────────────────────
# Each entry: (issue_key, metric_key, bad_when, molecule_id, delta)

_RULES: List[Tuple[str, str, float, str, float]] = [
    ("low_novelty",          "novelty",          0.45, "M_CURIOSITY_TRIGGER",   +0.08),
    ("low_impact",           "impact",           0.45, "M_REASONING_DEPTH",     +0.08),
    ("low_coherence",        "coherence",        0.45, "M_COHERENCE",           +0.07),
    ("low_context",          "context_efficiency",0.35,"M_CONTEXT_EFFICIENCY",  +0.08),
    ("low_desire_alignment", "desire_alignment", 0.35, "M_DESIRE_CREATION",     +0.06),
    ("low_trust",            "trust",            0.45, "M_TRUST_TONE",          +0.07),
    ("low_engagement",       "engagement",       0.40, "M_EMOTIONAL_CHARGE",    +0.07),
    ("low_task_success",     "task_success",     0.40, "M_ADAPTABILITY",        +0.06),
]


# ── Learning record ───────────────────────────────────────────────────────────

@dataclass
class Learning:
    learning_id: str = field(default_factory=lambda: f"l_{uuid.uuid4().hex[:10]}")
    source:      str = "auto"
    issue:       str = ""
    molecule_id: str = ""
    locus:       str = ""
    delta:       float = 0.0
    task_type:   str = ""
    timestamp:   str = field(default_factory=_UTC)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_id": self.learning_id,
            "source":      self.source,
            "issue":       self.issue,
            "molecule_id": self.molecule_id,
            "locus":       self.locus,
            "delta":       self.delta,
            "task_type":   self.task_type,
            "timestamp":   self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Learning":
        return cls(
            learning_id=d.get("learning_id", f"l_{uuid.uuid4().hex[:10]}"),
            source=d.get("source", "auto"),
            issue=d.get("issue", ""),
            molecule_id=d.get("molecule_id", ""),
            locus=d.get("locus", ""),
            delta=float(d.get("delta", 0.0)),
            task_type=d.get("task_type", ""),
            timestamp=d.get("timestamp", _UTC()),
        )


# ── AutoDistiller ─────────────────────────────────────────────────────────────

class AutoDistiller:
    """
    Watches task metrics after each act(), detects failure patterns,
    and applies molecule corrections to the genome while persisting the
    learning rule to data/learnings.jsonl.

    Integrates with OmniSentientEcosystem.run_cycle() — call
    distill_from_metrics(task, result["metrics"], agent.genome)
    after each task.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else _LEARNINGS_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._log: List[Learning] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._log.append(Learning.from_dict(json.loads(line)))
        except Exception as exc:
            LOG.debug("auto_distiller: failed to load %s: %s", self._path, exc)

    def _persist(self, learning: Learning) -> None:
        try:
            with self._path.open("a") as f:
                f.write(json.dumps(learning.to_dict()) + "\n")
        except Exception as exc:
            LOG.debug("auto_distiller: failed to persist: %s", exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def distill_from_metrics(
        self,
        task: Dict[str, Any],
        metrics: Dict[str, float],
        genome: Optional[Any] = None,
    ) -> Optional[Learning]:
        """
        Check task metrics; if a failure pattern matches, apply the correction
        to `genome` (in-place) and return the Learning.  Returns None if all
        metrics are acceptable.
        """
        issue_data = self._detect_issue(metrics)
        if issue_data is None:
            return None
        issue, mol_id, delta = issue_data
        locus = _locus_for(mol_id) or ""
        learning = Learning(
            issue=issue, molecule_id=mol_id, locus=locus, delta=delta,
            task_type=task.get("type", "unknown"),
        )
        if genome is not None and locus:
            self._apply_to_genome(genome, locus, mol_id, delta)
        self._log.append(learning)
        if len(self._log) > 2000:
            self._log = self._log[-2000:]
        self._persist(learning)
        LOG.info(
            "distilled: issue=%s mol=%s locus=%s delta=%.3f",
            issue, mol_id, locus, delta,
        )
        return learning

    def distill_manual(
        self,
        issue: str,
        molecule_id: str,
        delta: float,
        genome: Optional[Any] = None,
        source: str = "human",
    ) -> Learning:
        """Apply an explicit human-provided correction."""
        locus = _locus_for(molecule_id) or ""
        learning = Learning(
            source=source, issue=issue,
            molecule_id=molecule_id, locus=locus, delta=delta,
        )
        if genome is not None and locus:
            self._apply_to_genome(genome, locus, molecule_id, delta)
        self._log.append(learning)
        self._persist(learning)
        return learning

    def replay_on_genome(self, genome: Any, limit: int = 20) -> int:
        """Re-apply the most recent `limit` learnings to a genome. Useful for newly spawned agents."""
        applied = 0
        for learning in self._log[-limit:]:
            if learning.locus:
                self._apply_to_genome(genome, learning.locus, learning.molecule_id, learning.delta)
                applied += 1
        return applied

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [l.to_dict() for l in reversed(self._log[-limit:])]

    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        issues = Counter(l.issue for l in self._log)
        return {
            "total_learnings": len(self._log),
            "top_issues":      dict(issues.most_common(5)),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _detect_issue(
        self, metrics: Dict[str, float]
    ) -> Optional[Tuple[str, str, float]]:
        for issue_key, metric_key, threshold, mol_id, delta in _RULES:
            val = metrics.get(metric_key)
            if val is not None and val < threshold:
                return issue_key, mol_id, delta
        return None

    def _apply_to_genome(
        self, genome: Any, locus: str, molecule_id: str, delta: float
    ) -> bool:
        try:
            mol = genome.get_molecule(locus, molecule_id)
            if mol is None:
                return False
            mol.set_value(mol.get_value() + delta)
            return True
        except Exception as exc:
            LOG.debug("auto_distiller apply failed: %s", exc)
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────

_distiller: Optional[AutoDistiller] = None

def get_auto_distiller() -> AutoDistiller:
    global _distiller
    if _distiller is None:
        _distiller = AutoDistiller()
    return _distiller
