"""
Proactive Context Synthesizer — background thread that watches agent_logs/*.jsonl
for failed/low-quality tasks, identifies context gaps, and stores enrichment
tasks in SharedWorkingMemory so future agents find pre-computed context.

This converts passive failure logs into active context pre-loading.

Usage:
    synth = get_proactive_synthesizer()
    synth.start()       # background thread starts
    synth.stop()        # clean shutdown
    synth.status()      # returns enrichment queue stats
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.oss.mind.shared_working_memory import get_shared_memory

LOG = logging.getLogger("ghost.proactive_synthesizer")

_LOGS_DIR   = Path("data/agent_logs")
_POLL_SECS  = 30          # re-scan every 30 seconds
_LOW_FITNESS = 0.45       # tasks below this are "failed"
_ENRICH_TTL  = 1800       # enrichment tasks live 30 min in SWM


# ── Gap → enrichment mapping ──────────────────────────────────────────────────

_GAP_RULES: List[Dict[str, Any]] = [
    {
        "gap":          "missing_domain_context",
        "trigger_keys": ["domain"],
        "prompt_tpl":   "Pre-fetch domain knowledge for {domain} tasks. Identify key terms, frameworks, and common pitfalls.",
        "entry_type":   "domain_enrichment",
    },
    {
        "gap":          "low_reasoning_depth",
        "trigger_keys": ["M_REASONING_DEPTH"],
        "prompt_tpl":   "Expand reasoning scaffold for task type '{type}'. List sub-questions that should be answered before acting.",
        "entry_type":   "reasoning_scaffold",
    },
    {
        "gap":          "failed_analysis_task",
        "trigger_keys": ["analysis"],
        "prompt_tpl":   "Pre-analyze common failure modes for '{type}' tasks to reduce future failure rate.",
        "entry_type":   "analysis_scaffold",
    },
    {
        "gap":          "missing_market_context",
        "trigger_keys": ["market", "trade"],
        "prompt_tpl":   "Cache current market context signals for '{domain}' domain tasks.",
        "entry_type":   "market_enrichment",
    },
    {
        "gap":          "low_context_efficiency",
        "trigger_keys": ["M_CONTEXT_EFFICIENCY", "context_efficiency"],
        "prompt_tpl":   "Pre-compress context for '{type}' task class. Identify the 3 highest-signal context keys.",
        "entry_type":   "context_compression",
    },
]


# ── Enrichment task ───────────────────────────────────────────────────────────

@dataclass
class EnrichmentTask:
    enrichment_id: str
    gap:           str
    prompt:        str
    source_task:   str
    entry_type:    str
    timestamp:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enrichment_id": self.enrichment_id,
            "gap":           self.gap,
            "prompt":        self.prompt,
            "source_task":   self.source_task,
            "entry_type":    self.entry_type,
            "timestamp":     self.timestamp,
        }


# ── ProactiveContextSynthesizer ───────────────────────────────────────────────

class ProactiveContextSynthesizer:
    """
    Background thread that turns failure logs into enrichment hints.

    Architecture:
      - Poll data/agent_logs/*.jsonl every 30s
      - Find entries where fitness < 0.45
      - Map each failure to a gap rule
      - Write an enrichment task to SharedWorkingMemory (keyed by task type)
      - Future agents reading SWM for a task type get pre-loaded context hints
    """

    def __init__(
        self,
        logs_dir:  Optional[Path] = None,
        poll_secs: int = _POLL_SECS,
    ):
        self._logs_dir  = Path(logs_dir) if logs_dir else _LOGS_DIR
        self._poll_secs = poll_secs
        self._memory    = get_shared_memory()
        self._thread:   Optional[threading.Thread] = None
        self._stop_evt  = threading.Event()
        self._stats     = {"scans": 0, "failures_found": 0, "enrichments_queued": 0}
        self._seen_ids: set = set()     # avoid re-processing same log entries

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="proactive-ctx-synth"
        )
        self._thread.start()
        LOG.info("ProactiveContextSynthesizer started (poll=%ds)", self._poll_secs)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        LOG.info("ProactiveContextSynthesizer stopped")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> Dict[str, Any]:
        return {
            "running":            self.is_running(),
            "poll_seconds":       self._poll_secs,
            "logs_dir":           str(self._logs_dir),
            "seen_entries":       len(self._seen_ids),
            **self._stats,
        }

    def trigger_scan(self) -> Dict[str, Any]:
        """Manual one-shot scan (for testing or on-demand enrichment)."""
        return self._scan_logs()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._scan_logs()
            except Exception as exc:
                LOG.debug("synthesizer scan error: %s", exc)
            self._stop_evt.wait(self._poll_secs)

    def _scan_logs(self) -> Dict[str, Any]:
        self._stats["scans"] += 1
        failures = self._collect_failures()
        enrichments_this_scan = 0
        for entry in failures:
            gap_rule = self._detect_gap(entry)
            if gap_rule is None:
                continue
            enrich = self._build_enrichment(entry, gap_rule)
            self._store_enrichment(entry, enrich)
            enrichments_this_scan += 1
            self._stats["enrichments_queued"] += 1
        return {
            "failures_found":      len(failures),
            "enrichments_queued":  enrichments_this_scan,
        }

    def _collect_failures(self) -> List[Dict[str, Any]]:
        failures: List[Dict[str, Any]] = []
        if not self._logs_dir.exists():
            return failures
        for log_file in self._logs_dir.glob("*.jsonl"):
            try:
                with log_file.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        eid = entry.get("task_id") or entry.get("id") or str(hash(line))
                        if eid in self._seen_ids:
                            continue
                        self._seen_ids.add(eid)
                        fitness = float(entry.get("fitness", entry.get("metrics", {}).get("task_success", 1.0)))
                        if fitness < _LOW_FITNESS:
                            self._stats["failures_found"] += 1
                            failures.append(entry)
            except Exception as exc:
                LOG.debug("error reading %s: %s", log_file, exc)
        return failures

    def _detect_gap(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        entry_text = json.dumps(entry).lower()
        for rule in _GAP_RULES:
            for key in rule["trigger_keys"]:
                if key.lower() in entry_text:
                    return rule
        return None

    def _build_enrichment(
        self, entry: Dict[str, Any], rule: Dict[str, Any]
    ) -> EnrichmentTask:
        import uuid
        prompt = rule["prompt_tpl"].format(
            domain=entry.get("domain", entry.get("type", "unknown")),
            type=entry.get("type", "unknown"),
        )
        return EnrichmentTask(
            enrichment_id=f"enrich_{uuid.uuid4().hex[:8]}",
            gap=rule["gap"],
            prompt=prompt,
            source_task=entry.get("task_id", "?"),
            entry_type=rule["entry_type"],
        )

    def _store_enrichment(
        self, entry: Dict[str, Any], enrich: EnrichmentTask
    ) -> None:
        # Key by task type so agents doing the same task type find pre-loaded hints
        task_type = entry.get("type", "unknown")
        self._memory.write(
            task_id=f"enrichment_{task_type}",
            agent_id="proactive_synthesizer",
            content=enrich.to_dict(),
            entry_type=enrich.entry_type,
            ttl_seconds=_ENRICH_TTL,
        )
        LOG.debug("enrichment stored: gap=%s type=%s", enrich.gap, task_type)

    def get_enrichments_for(self, task_type: str) -> List[Dict[str, Any]]:
        """Retrieve pre-computed enrichment hints for a given task type."""
        entries = self._memory.read(f"enrichment_{task_type}")
        return [e.content for e in entries]


# ── Singleton ─────────────────────────────────────────────────────────────────

_synthesizer: Optional[ProactiveContextSynthesizer] = None

def get_proactive_synthesizer() -> ProactiveContextSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = ProactiveContextSynthesizer()
    return _synthesizer
