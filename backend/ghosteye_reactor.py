"""GhostEye reactor — turns ambient screen observations into autonomic action.

Signals detected from each frame + window context:
  * STUCK        → same active_app + ~same text for >= STUCK_SECONDS
  * ERROR        → OCR text matches error regex (Traceback / Error: / FAIL / etc.)
  * GOAL         → OCR text matches "TODO" / "implement" / "fix" patterns
  * PROGRESS     → success signals (OK, PASS, green, "all tests passed")

Reactions (each with a per-type cooldown to bound LLM cost):
  * STUCK   → fire an unstuck KAIROS cycle (1 proposal tailored to current screen),
              push best proposal to Telegram + UI toast
  * ERROR   → append Séance entry, distill a rule if 3+ in a window
  * GOAL    → create a new Autotelic goal with progress 0.1 if not duplicate
  * PCL     → every frame lightly nudges PCL based on signal mix

State lives in-memory on the process (single-user personal build).
"""
from __future__ import annotations
import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

LOG = logging.getLogger("ghost.eye.reactor")

STUCK_SECONDS = 300          # 5 minutes on same screen == stuck
STUCK_COOLDOWN = 600         # at most 1 unstuck cycle per 10 minutes
ERROR_COOLDOWN = 120         # at most 1 Séance capture per 2 minutes
GOAL_COOLDOWN = 600          # at most 1 goal creation per 10 minutes
TEXT_SIMILAR_MIN = 0.85      # Jaccard >= this means "same context"

ERROR_RE = re.compile(
    r"(Traceback|Error:|Exception|FAIL(?:ED)?|undefined|segfault|panic:|NullPointer|"
    r"\bENOENT\b|cannot find|not found|ModuleNotFoundError|SyntaxError|TypeError|"
    r"TimeoutError|ConnectionError)",
    re.I,
)
GOAL_RE = re.compile(
    r"(TODO|FIXME|implement\b|build\b|wire\b|ship\b|refactor\b)",
    re.I,
)
SUCCESS_RE = re.compile(
    r"(all tests pass|passed|\bOK\b|success|green|built successfully|deployed)",
    re.I,
)


def _tok(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z_0-9]{2,}", (s or "").lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


@dataclass
class _Memo:
    last_text_tokens: set = field(default_factory=set)
    last_app: str = ""
    stuck_since: float = 0.0
    last_stuck_fire: float = 0.0
    last_error_fire: float = 0.0
    last_goal_fire: float = 0.0
    recent_errors: list[str] = field(default_factory=list)


class GhostEyeReactor:
    def __init__(
        self,
        db,
        memory,
        ws_mgr,
        kairos_cycle: Callable[[], Awaitable[dict]],
        telegram_send: Callable[[str], Awaitable[None]] | None = None,
    ):
        self.db = db
        self.memory = memory
        self.ws = ws_mgr
        self.kairos = kairos_cycle
        self.send_tg = telegram_send
        self.memo = _Memo()

    async def on_frame(self, frame: dict):
        """Called on every companion-pushed GhostEye frame. Non-blocking."""
        try:
            asyncio.create_task(self._react(frame))
        except Exception:
            LOG.exception("reactor dispatch failed")

    async def _react(self, frame: dict):
        text = (frame.get("text") or "").strip()
        app = (frame.get("active_app") or "").strip()
        ts = time.time()
        toks = _tok(text)

        # 1) PCL nudge based on signal mix (cheap, every frame)
        await self._update_pcl(text)

        # 2) Stuck detection (same app + high text overlap, N seconds)
        sim = _jaccard(toks, self.memo.last_text_tokens)
        same_context = (app and app == self.memo.last_app and sim >= TEXT_SIMILAR_MIN)
        if same_context:
            if not self.memo.stuck_since:
                self.memo.stuck_since = ts
            elif (ts - self.memo.stuck_since) >= STUCK_SECONDS and \
                 (ts - self.memo.last_stuck_fire) >= STUCK_COOLDOWN:
                self.memo.last_stuck_fire = ts
                asyncio.create_task(self._on_stuck(app, text))
        else:
            self.memo.stuck_since = ts if app else 0.0

        self.memo.last_text_tokens = toks
        self.memo.last_app = app

        # 3) Error capture → Séance
        if ERROR_RE.search(text) and (ts - self.memo.last_error_fire) >= ERROR_COOLDOWN:
            self.memo.last_error_fire = ts
            asyncio.create_task(self._on_error(app, text))

        # 4) Goal detection
        if GOAL_RE.search(text) and (ts - self.memo.last_goal_fire) >= GOAL_COOLDOWN:
            m = GOAL_RE.search(text)
            # extract surrounding sentence
            start = max(0, text.rfind(".", 0, m.start()) + 1)
            end = text.find(".", m.end())
            phrase = text[start:end if end > 0 else m.end() + 80].strip()[:140]
            if phrase and len(phrase) > 10:
                self.memo.last_goal_fire = ts
                asyncio.create_task(self._on_goal(phrase, app))

    # ------------------------------------------------------------------
    async def _update_pcl(self, text: str):
        """Lightly nudge PCL based on whatever's on screen."""
        state = None
        if SUCCESS_RE.search(text):
            state = ("High confidence", 440, "#8b5cf6", "I know this deeply")
        elif ERROR_RE.search(text):
            state = ("Uncertainty", 220, "#f59e0b", "Reasoning under uncertainty")
        elif "import" in text.lower() or "function" in text.lower():
            state = ("Learning", 330, "#22d3ee", "New knowledge being encoded")
        if not state:
            return
        name, hz, color, meaning = state
        await self.db.system_state.update_one(
            {"_id": "singleton"},
            {"$set": {"pcl.state": name, "pcl.frequency_hz": hz,
                      "pcl.color": color, "pcl.meaning": meaning}},
        )

    async def _on_stuck(self, app: str, text: str):
        LOG.info("GhostEye: STUCK detected on %s — firing unstuck KAIROS", app)
        try:
            cycle = await self.kairos()
        except Exception:
            LOG.exception("stuck-kairos failed")
            return
        blurb = (f"\U0001f441 GhostEye noticed you're stuck on \"{app or 'this'}\" "
                 f"for 5+ minutes.\n\nKAIROS #{cycle['cycle_num']} proposal "
                 f"({cycle['verdict']} · {cycle['final_score']}):\n{cycle['proposal']}")
        spoken = (f"Hey Robert. GhostEye thinks you've been stuck on "
                  f"{app or 'this screen'} for five minutes. Here's what I'd try: "
                  f"{cycle['proposal']}")
        await self.ws.broadcast("ghosteye_stuck", {
            "app": app, "cycle": cycle["cycle_num"], "proposal": cycle["proposal"],
            "score": cycle["final_score"], "verdict": cycle["verdict"],
        })
        # whisper — any listening client (browser, native voice loop) speaks this
        await self.ws.broadcast("ghosteye_whisper", {
            "text": spoken,
            "source": "stuck",
            "priority": "normal",
            "voice": "en-US-AvaMultilingualNeural",
        })
        if self.send_tg:
            try:
                await self.send_tg(blurb)
            except Exception:
                LOG.exception("telegram send failed")
        try:
            await self.memory.store(
                f"[unstuck] Robert was stuck on {app}. KAIROS proposed: {cycle['proposal']}",
                "observation", "ghosteye_reactor", 0.65,
            )
        except Exception:
            pass

    async def _on_error(self, app: str, text: str):
        # trim the error-looking sentence
        m = ERROR_RE.search(text)
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 160)
        snippet = text[start:end].strip()
        LOG.info("GhostEye: ERROR detected on %s — capturing to Séance", app)
        entry = {
            "domain": f"{app or 'screen'}",
            "mood": "burned",
            "lesson": snippet[:240],
            "timestamp": _now_iso(),
        }
        await self.db.system_state.update_one(
            {"_id": "singleton"},
            {"$push": {"seance": {"$each": [entry], "$slice": -40}}},
        )
        await self.ws.broadcast("seance", entry)
        self.memo.recent_errors.append(snippet[:120])
        self.memo.recent_errors = self.memo.recent_errors[-10:]
        # also record as a low-importance observation memory
        try:
            await self.memory.store(
                f"[error on {app}] {snippet[:240]}", "observation", "ghosteye_reactor", 0.50,
            )
        except Exception:
            pass

    async def _on_goal(self, phrase: str, app: str):
        # dedupe against existing goals
        doc = await self.db.system_state.find_one({"_id": "singleton"}, {"autotelic_goals": 1})
        existing = doc.get("autotelic_goals", []) if doc else []
        norm = phrase.lower()
        for g in existing:
            if norm[:40] in (g.get("title", "") + " " + g.get("detail", "")).lower():
                return
        new_goal = {
            "title": phrase[:80],
            "detail": f"auto-detected from GhostEye @ {app}",
            "progress": 0.05,
        }
        await self.db.system_state.update_one(
            {"_id": "singleton"},
            {"$push": {"autotelic_goals": {"$each": [new_goal], "$slice": -40}}},
        )
        LOG.info("GhostEye: GOAL added → %s", phrase[:60])
        await self.ws.broadcast("goal_added", new_goal)
        try:
            await self.memory.store(
                f"[auto-goal] {phrase}", "decision", "ghosteye_reactor", 0.70,
            )
        except Exception:
            pass


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
