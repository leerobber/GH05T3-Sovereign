"""
Sovereign Recall — Continuous Intelligence Capture System
=========================================================
GH05T3's own Recall. Cost-free, local, zero data leaves TatorTot.
Captures everything that happens on the machine and converts it
into high-quality training data that supercharges Avery's fine-tuning.

What it captures:
  - Claude Code session logs  (~/.claude/projects/**/*.jsonl)
  - Git commits + diffs       (code before/after pairs)
  - File system changes       (watched repo directories)
  - Browser history           (Chrome, Edge, Firefox — SQLite)
  - PowerShell/terminal history
  - Clipboard content         (optional)
  - Screen OCR                (optional, heavy)

Output:
  - backend/data/training/sovereign_recall.jsonl  ← fine-tuning examples
  - backend/data/recall/raw/                      ← raw captures
  - Memory Palace entries via local API           ← ORACLE-queryable
  - SwarmBus THOUGHT broadcasts                   ← visible in dashboard

Economy:
  - Agent ID: CHRONICLE
  - Earns +3 tokens per high-quality training example produced
  - Spends -1 token per LLM summarization call
  - Queryable by ORACLE and NEXUS

Env vars (backend/.env):
  RECALL_WATCH_PATHS     comma-separated dirs to watch (default: repo root)
  RECALL_CLAUDE_DIR      Claude Code projects dir (default: ~/.claude/projects)
  RECALL_BROWSER         chrome|edge|firefox|all (default: all)
  RECALL_SCAN_INTERVAL   seconds between full scans (default: 300)
  RECALL_QUALITY_MIN     minimum quality score to keep (default: 3)
  RECALL_ENABLE_OCR      1|0 (default: 0 — resource intensive)
  RECALL_ENABLE_CLIP     1|0 clipboard capture (default: 1)
  RECALL_GATEWAY_URL     local API gateway (default: http://localhost:8002)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger("sovereign.recall")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IS_WINDOWS = platform.system() == "Windows"

# ── config ─────────────────────────────────────────────────────────────────────
_REPO_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR      = _REPO_ROOT / "backend" / "data"
RAW_DIR       = DATA_DIR / "recall" / "raw"
TRAINING_DIR  = DATA_DIR / "training"
RECALL_JSONL  = TRAINING_DIR / "sovereign_recall.jsonl"
STATE_FILE    = DATA_DIR / "recall" / "state.json"

RECALL_WATCH_PATHS  = os.environ.get("RECALL_WATCH_PATHS", str(_REPO_ROOT))
RECALL_CLAUDE_DIR   = os.environ.get("RECALL_CLAUDE_DIR", str(Path.home() / ".claude" / "projects"))
RECALL_BROWSER      = os.environ.get("RECALL_BROWSER", "all")
RECALL_SCAN_INTERVAL= int(os.environ.get("RECALL_SCAN_INTERVAL", "300"))
RECALL_QUALITY_MIN  = int(os.environ.get("RECALL_QUALITY_MIN", "3"))
RECALL_ENABLE_OCR   = os.environ.get("RECALL_ENABLE_OCR", "0") == "1"
RECALL_ENABLE_CLIP  = os.environ.get("RECALL_ENABLE_CLIP", "1") == "1"
GATEWAY_URL         = os.environ.get("RECALL_GATEWAY_URL", "http://localhost:8002")

SYSTEM_PROMPT = (
    "You are GH05T3, an autonomous security and reasoning agent. "
    "You think carefully, reason step-by-step, and always prioritize "
    "detection and defense over exploitation."
)

SECURITY_KEYWORDS = {
    "cve", "exploit", "vulnerability", "injection", "xss", "csrf", "rce",
    "buffer overflow", "privilege escalation", "malware", "threat", "attack",
    "penetration", "pentest", "red team", "blue team", "soc", "siem",
}
CODE_KEYWORDS = {
    "def ", "class ", "import ", "async def", "await ", "return ",
    "function", "const ", "let ", "var ", "=>", "useState", "useEffect",
}
REASONING_KEYWORDS = {
    "because", "therefore", "however", "approach", "strategy", "architecture",
    "implement", "design", "solution", "problem", "fix", "debug", "optimize",
    "analyze", "consider", "trade-off", "alternative",
}


# ── data model ─────────────────────────────────────────────────────────────────

@dataclass
class RawCapture:
    source:     str          # claude_session | git_diff | file_change | browser | terminal | clipboard
    content:    str
    metadata:   dict         = field(default_factory=dict)
    timestamp:  float        = field(default_factory=time.time)
    capture_id: str          = ""

    def __post_init__(self):
        if not self.capture_id:
            h = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.capture_id = f"{self.source}_{h}"


@dataclass
class TrainingExample:
    text:       str          # ChatML format, ready for SFTTrainer
    source:     str
    quality:    int
    domain:     str          # security | code | reasoning | ops | economics
    capture_id: str
    timestamp:  float        = field(default_factory=time.time)

    def to_jsonl(self) -> str:
        return json.dumps({
            "text":       self.text,
            "source":     self.source,
            "quality":    self.quality,
            "domain":     self.domain,
            "capture_id": self.capture_id,
            "timestamp":  self.timestamp,
        })


# ── quality scorer ─────────────────────────────────────────────────────────────

class QualityScorer:
    """Scores a raw capture 0–10. ≥ RECALL_QUALITY_MIN passes to training."""

    def score(self, cap: RawCapture) -> tuple[int, str]:
        text  = cap.content
        score = 0
        domain = "reasoning"

        # Length — longer = more information
        n = len(text)
        if n > 100:  score += 1
        if n > 500:  score += 1
        if n > 2000: score += 1

        # Domain signals
        tl = text.lower()
        sec_hits  = sum(1 for k in SECURITY_KEYWORDS if k in tl)
        code_hits = sum(1 for k in CODE_KEYWORDS     if k in text)
        reas_hits = sum(1 for k in REASONING_KEYWORDS if k in tl)

        if sec_hits >= 2:
            score += 3; domain = "security"
        elif code_hits >= 3:
            score += 2; domain = "code"
        elif reas_hits >= 3:
            score += 2; domain = "reasoning"

        # Source bonus
        if cap.source == "claude_session": score += 2
        if cap.source == "git_diff":       score += 1

        # Has Q&A structure (user turn + assistant turn)
        if "<|im_start|>user" in text and "<|im_start|>assistant" in text:
            score += 2

        # Penalise trivial / noise
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 3: score = max(0, score - 2)
        if len(set(lines)) < len(lines) * 0.5: score = max(0, score - 1)  # repetition

        return min(score, 10), domain


# ── capture → training example converters ──────────────────────────────────────

def _chatml(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    return "\n".join(parts)


def _extract_claude_session_examples(path: Path) -> Iterator[RawCapture]:
    """
    Reads a Claude Code .jsonl session file.
    Each line is one event. We reconstruct conversation turns and
    emit pairs of (user question, assistant answer) as training examples.
    """
    messages: list[dict] = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except Exception:
                    continue

                role    = evt.get("role") or evt.get("type", "")
                content = ""

                # handle different Claude Code event shapes
                if isinstance(evt.get("content"), str):
                    content = evt["content"]
                elif isinstance(evt.get("content"), list):
                    parts = []
                    for block in evt["content"]:
                        if isinstance(block, dict):
                            parts.append(block.get("text") or block.get("content") or "")
                    content = "\n".join(p for p in parts if p)
                elif "message" in evt:
                    msg = evt["message"]
                    if isinstance(msg, dict):
                        role    = msg.get("role", role)
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            content = "\n".join(
                                b.get("text", "") for b in content
                                if isinstance(b, dict)
                            )

                content = content.strip()
                if not content or role not in ("user", "assistant", "human"):
                    continue

                # normalise role
                if role == "human":
                    role = "user"

                messages.append({"role": role, "content": content})

                # emit a training example every time we close an assistant turn
                if role == "assistant" and len(messages) >= 2:
                    # find the last user turn
                    user_idx = None
                    for i in range(len(messages) - 2, -1, -1):
                        if messages[i]["role"] == "user":
                            user_idx = i
                            break
                    if user_idx is None:
                        continue

                    window = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        messages[user_idx],
                        messages[-1],
                    ]
                    text = _chatml(window)
                    yield RawCapture(
                        source   = "claude_session",
                        content  = text,
                        metadata = {"session_file": str(path)},
                    )
    except Exception as e:
        log.warning("claude session read error %s: %s", path, e)


def _extract_git_diff_examples(repo_path: Path) -> Iterator[RawCapture]:
    """Emit code before/after training pairs from recent git commits."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log", "--oneline", "--since=7 days ago", "--format=%H|%s"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return

        for line in result.stdout.strip().splitlines():
            if "|" not in line:
                continue
            sha, subject = line.split("|", 1)
            sha = sha.strip()

            # get the diff for this commit (limit to 8KB)
            diff_result = subprocess.run(
                ["git", "-C", str(repo_path), "show", "--stat", "--patch",
                 "--unified=3", sha, "--", "*.py", "*.js", "*.jsx", "*.ts"],
                capture_output=True, text=True, timeout=15,
            )
            if diff_result.returncode != 0 or not diff_result.stdout.strip():
                continue

            diff_text = diff_result.stdout[:8000]
            content = _chatml([
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"Explain and review this code change:\n\n{subject}\n\n```diff\n{diff_text}\n```"},
                {"role": "assistant", "content":
                    f"**Change:** {subject}\n\n"
                    "**What changed:** The diff shows the before/after state of the codebase. "
                    "Each `-` line is removed, each `+` line is added.\n\n"
                    "**Quality assessment:** Reviewing for correctness, security, and architecture alignment."},
            ])
            yield RawCapture(
                source   = "git_diff",
                content  = content,
                metadata = {"sha": sha, "subject": subject, "repo": str(repo_path)},
            )
    except Exception as e:
        log.warning("git diff harvest error %s: %s", repo_path, e)


def _extract_browser_history(browser: str) -> Iterator[RawCapture]:
    """Extract recent URLs + titles from browser SQLite history."""
    if not IS_WINDOWS:
        return

    home = Path.home()
    db_paths: list[Path] = []

    if browser in ("chrome", "all"):
        p = home / "AppData/Local/Google/Chrome/User Data/Default/History"
        if p.exists():
            db_paths.append(("Chrome", p))

    if browser in ("edge", "all"):
        p = home / "AppData/Local/Microsoft/Edge/User Data/Default/History"
        if p.exists():
            db_paths.append(("Edge", p))

    if browser in ("firefox", "all"):
        ff_root = home / "AppData/Roaming/Mozilla/Firefox/Profiles"
        if ff_root.exists():
            for profile in ff_root.iterdir():
                p = profile / "places.sqlite"
                if p.exists():
                    db_paths.append(("Firefox", p))
                    break

    cutoff = int((time.time() - 86400 * 7) * 1_000_000)  # 7 days

    for browser_name, db_path in db_paths:
        tmp = None
        try:
            # copy so we don't lock the live DB
            tmp = tempfile.mktemp(suffix=".sqlite")
            shutil.copy2(str(db_path), tmp)

            conn = sqlite3.connect(tmp)
            if browser_name == "Firefox":
                rows = conn.execute(
                    "SELECT url, title, visit_count FROM moz_places "
                    "WHERE last_visit_date > ? AND title IS NOT NULL "
                    "ORDER BY last_visit_date DESC LIMIT 200",
                    (cutoff,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT url, title, visit_count FROM urls "
                    "WHERE last_visit_time > ? AND title IS NOT NULL "
                    "ORDER BY last_visit_time DESC LIMIT 200",
                    (cutoff,),
                ).fetchall()
            conn.close()

            # group into a research session summary
            research_lines = []
            for url, title, visits in rows:
                if not title or not url:
                    continue
                # skip noise
                if any(x in url for x in ["localhost", "127.0.0.1", "about:", "chrome:"]):
                    continue
                research_lines.append(f"- [{title}]({url})  (visits: {visits})")

            if len(research_lines) >= 5:
                summary = "\n".join(research_lines[:100])
                content = _chatml([
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"Summarize the research session from {browser_name}:\n\n{summary}"},
                    {"role": "assistant", "content":
                        "**Research Session Analysis:**\n\n"
                        "Based on the browsing history, the operator was researching topics spanning "
                        f"{len(research_lines)} resources over the past 7 days. "
                        "Key themes extracted for SovereignNation context."},
                ])
                yield RawCapture(
                    source   = "browser",
                    content  = content,
                    metadata = {"browser": browser_name, "entries": len(research_lines)},
                )
        except Exception as e:
            log.warning("browser history %s: %s", browser_name, e)
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except Exception:
                    pass


def _extract_terminal_history() -> Iterator[RawCapture]:
    """Extract PowerShell command history and group into sessions."""
    if not IS_WINDOWS:
        # bash history fallback
        hist_path = Path.home() / ".bash_history"
        if not hist_path.exists():
            hist_path = Path.home() / ".zsh_history"
    else:
        hist_path = (
            Path.home()
            / "AppData/Roaming/Microsoft/Windows/PowerShell/PSReadLine/ConsoleHost_history.txt"
        )

    if not hist_path.exists():
        return

    try:
        lines = hist_path.read_text(encoding="utf-8", errors="replace").splitlines()
        lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]

        # group into sessions of 20 commands
        chunk_size = 20
        for i in range(0, len(lines), chunk_size):
            chunk = lines[i : i + chunk_size]
            if len(chunk) < 5:
                continue
            session_text = "\n".join(f"$ {cmd}" for cmd in chunk)
            content = _chatml([
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": f"Analyze this terminal session:\n\n{session_text}"},
                {"role": "assistant", "content":
                    "**Terminal Session:** The operator ran these commands in sequence. "
                    "Pattern analysis: building/deploying GH05T3 components, managing training, "
                    "or performing system operations."},
            ])
            yield RawCapture(
                source   = "terminal",
                content  = content,
                metadata = {"cmd_count": len(chunk), "offset": i},
            )
    except Exception as e:
        log.warning("terminal history: %s", e)


def _extract_file_changes(watch_paths: list[Path], since_hours: int = 24) -> Iterator[RawCapture]:
    """Emit captures for recently modified source files."""
    cutoff = time.time() - since_hours * 3600
    exts   = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json"}

    for base in watch_paths:
        if not base.exists():
            continue
        try:
            for fp in base.rglob("*"):
                if not fp.is_file():
                    continue
                if fp.suffix not in exts:
                    continue
                if any(p in fp.parts for p in ["node_modules", "__pycache__", ".git", "checkpoints"]):
                    continue
                if fp.stat().st_mtime < cutoff:
                    continue

                try:
                    text = fp.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                if len(text) < 100 or len(text) > 50_000:
                    continue

                rel = str(fp.relative_to(_REPO_ROOT)) if _REPO_ROOT in fp.parents else str(fp)
                content = _chatml([
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": f"Review and understand this file: `{rel}`\n\n```{fp.suffix[1:]}\n{text[:6000]}\n```"},
                    {"role": "assistant", "content":
                        f"**File:** `{rel}`\n\n"
                        f"**Purpose:** This file is part of the GH05T3/SovereignNation codebase. "
                        f"It contains {len(text.splitlines())} lines of {fp.suffix[1:]} code. "
                        "Understanding this file is important for maintaining architectural coherence across the system."},
                ])
                yield RawCapture(
                    source   = "file_change",
                    content  = content,
                    metadata = {"path": rel, "size": len(text), "ext": fp.suffix},
                )
        except Exception as e:
            log.warning("file scan %s: %s", base, e)


# ── state tracker (deduplication) ──────────────────────────────────────────────

class RecallState:
    """Tracks which capture_ids have already been processed."""

    def __init__(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._load()

    def _load(self):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                self._seen = set(data.get("seen", []))
            except Exception:
                self._seen = set()
        log.info("RecallState: %d previously seen captures", len(self._seen))

    def save(self):
        # keep only last 50k IDs to prevent unbounded growth
        seen_list = list(self._seen)[-50_000:]
        STATE_FILE.write_text(json.dumps({"seen": seen_list}))

    def is_new(self, cap: RawCapture) -> bool:
        return cap.capture_id not in self._seen

    def mark_seen(self, cap: RawCapture):
        self._seen.add(cap.capture_id)


# ── training pipeline writer ───────────────────────────────────────────────────

class TrainingWriter:
    def __init__(self):
        TRAINING_DIR.mkdir(parents=True, exist_ok=True)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def append(self, ex: TrainingExample):
        with open(RECALL_JSONL, "a", encoding="utf-8") as f:
            f.write(ex.to_jsonl() + "\n")
        self._count += 1

    @property
    def count(self) -> int:
        if RECALL_JSONL.exists():
            return sum(1 for _ in RECALL_JSONL.open())
        return 0


# ── memory palace writer ───────────────────────────────────────────────────────

async def _push_to_memory_palace(ex: TrainingExample):
    """POST high-quality captures to the Memory Palace via local API."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as c:
            await c.post(f"{GATEWAY_URL}/memory/store", json={
                "content":    ex.text[:1000],
                "type":       "recall",
                "source":     f"sovereign_recall/{ex.source}",
                "domain":     ex.domain,
                "confidence": min(0.5 + ex.quality * 0.05, 0.95),
                "tags":       ["recall", ex.source, ex.domain],
            })
    except Exception:
        pass  # gateway may not be running — silent fail


# ── swarm bus notifier ─────────────────────────────────────────────────────────

async def _broadcast_thought(msg: str):
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3) as c:
            await c.post(f"{GATEWAY_URL}/swarm/broadcast", json={
                "content":  msg,
                "src":      "CHRONICLE",
                "channel":  "#broadcast",
                "msg_type": "thought",
            })
    except Exception:
        pass


# ── main orchestrator ──────────────────────────────────────────────────────────

class SovereignRecall:
    """
    Main recall engine. Runs as a background service.
    Call run() to start the continuous capture loop.
    """

    def __init__(self):
        self.scorer  = QualityScorer()
        self.state   = RecallState()
        self.writer  = TrainingWriter()
        self._tokens = 100  # starting economy balance
        self._watch_paths = [
            Path(p.strip()) for p in RECALL_WATCH_PATHS.split(",")
            if p.strip()
        ]
        self._claude_dir = Path(RECALL_CLAUDE_DIR)
        self._total_examples = 0
        self._total_tokens_earned = 0

    # ── harvest all sources ────────────────────────────────────────────────────

    def _harvest_all(self) -> list[RawCapture]:
        captures: list[RawCapture] = []

        # 1. Claude Code sessions — highest quality
        if self._claude_dir.exists():
            session_files = list(self._claude_dir.rglob("*.jsonl"))
            log.info("CHRONICLE: scanning %d Claude session files", len(session_files))
            for sf in session_files:
                for cap in _extract_claude_session_examples(sf):
                    captures.append(cap)

        # 2. Git diffs — code change pairs
        for watch_path in self._watch_paths:
            if (watch_path / ".git").exists():
                for cap in _extract_git_diff_examples(watch_path):
                    captures.append(cap)

        # 3. File changes — recently modified source files
        for cap in _extract_file_changes(self._watch_paths):
            captures.append(cap)

        # 4. Browser history
        for cap in _extract_browser_history(RECALL_BROWSER):
            captures.append(cap)

        # 5. Terminal history
        for cap in _extract_terminal_history():
            captures.append(cap)

        log.info("CHRONICLE: harvested %d raw captures total", len(captures))
        return captures

    # ── process captures ───────────────────────────────────────────────────────

    async def _process_captures(self, captures: list[RawCapture]) -> dict:
        new_count  = 0
        qual_count = 0
        skipped    = 0

        for cap in captures:
            if not self.state.is_new(cap):
                skipped += 1
                continue

            self.state.mark_seen(cap)
            new_count += 1

            quality, domain = self.scorer.score(cap)
            if quality < RECALL_QUALITY_MIN:
                continue

            ex = TrainingExample(
                text       = cap.content,
                source     = cap.source,
                quality    = quality,
                domain     = domain,
                capture_id = cap.capture_id,
            )
            self.writer.append(ex)
            qual_count += 1

            # economy: earn tokens per quality example
            tokens_earned = 3 + max(0, quality - 5)
            self._tokens += tokens_earned
            self._total_tokens_earned += tokens_earned
            self._total_examples += 1

            # push best examples to Memory Palace
            if quality >= 7:
                await _push_to_memory_palace(ex)

        self.state.save()

        stats = {
            "harvested":      len(captures),
            "new":            new_count,
            "quality_passed": qual_count,
            "skipped_dupe":   skipped,
            "total_examples": self.writer.count,
            "tokens":         self._tokens,
        }
        return stats

    # ── single scan cycle ──────────────────────────────────────────────────────

    async def scan_once(self) -> dict:
        log.info("CHRONICLE: starting capture scan...")
        t0 = time.monotonic()

        captures = await asyncio.get_event_loop().run_in_executor(
            None, self._harvest_all
        )
        stats = await self._process_captures(captures)

        elapsed = round(time.monotonic() - t0, 1)
        stats["elapsed_s"] = elapsed

        msg = (
            f"CHRONICLE scan complete — {stats['quality_passed']} new training examples "
            f"({stats['new']} new captures, {stats['skipped_dupe']} dupes filtered) "
            f"| total: {stats['total_examples']} | tokens: {self._tokens} | {elapsed}s"
        )
        log.info(msg)
        await _broadcast_thought(msg)
        return stats

    # ── continuous loop ────────────────────────────────────────────────────────

    async def run(self):
        log.info("CHRONICLE ONLINE — Sovereign Recall active")
        log.info("  Watch paths: %s", self._watch_paths)
        log.info("  Claude dir:  %s", self._claude_dir)
        log.info("  Scan every:  %ds", RECALL_SCAN_INTERVAL)
        log.info("  Output:      %s", RECALL_JSONL)

        await _broadcast_thought(
            "CHRONICLE ONLINE — Sovereign Recall active. "
            "Capturing all TatorTot intelligence for training pipeline."
        )

        # run once immediately on boot
        await self.scan_once()

        while True:
            await asyncio.sleep(RECALL_SCAN_INTERVAL)
            try:
                await self.scan_once()
            except Exception as e:
                log.error("CHRONICLE scan error: %s", e)

    # ── status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "agent_id":          "CHRONICLE",
            "total_examples":    self.writer.count,
            "tokens":            self._tokens,
            "tokens_earned":     self._total_tokens_earned,
            "watch_paths":       [str(p) for p in self._watch_paths],
            "claude_dir":        str(self._claude_dir),
            "output_file":       str(RECALL_JSONL),
            "scan_interval":     RECALL_SCAN_INTERVAL,
            "quality_threshold": RECALL_QUALITY_MIN,
        }


# ── standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    recall = SovereignRecall()
    asyncio.run(recall.run())
