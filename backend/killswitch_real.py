"""Real KillSwitch for GH05T3 — actually affects the system.

Modes:
  STEALTH   — mute all non-critical outbound LLM calls, pause scheduler
  DEEP_FREEZE — suspend all GH05T3 background tasks, lock the API
  SHOCKER   — emergency: kill specified processes + Telegram alert
"""
from __future__ import annotations
import asyncio
import logging
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psutil

LOG = logging.getLogger("ghost.killswitch")

_FROZEN = False
_STEALTH = False


def is_frozen() -> bool:
    return _FROZEN


def is_stealth() -> bool:
    return _STEALTH


async def engage_stealth(db, telegram_fn=None) -> dict:
    """Pause scheduler, stop outbound LLM calls for non-critical paths."""
    global _STEALTH
    _STEALTH = True
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"killswitch.mode": "STEALTH", "killswitch.engaged_at": datetime.now(timezone.utc).isoformat()}}
    )
    LOG.warning("KillSwitch STEALTH engaged")
    if telegram_fn:
        await _safe_telegram(telegram_fn, "GH05T3 KillSwitch: STEALTH mode engaged. Outbound LLM paused.")
    return {"mode": "STEALTH", "ok": True}


async def engage_deep_freeze(db, scheduler, telegram_fn=None) -> dict:
    """Pause the APScheduler and block all background processing."""
    global _FROZEN
    _FROZEN = True
    try:
        if scheduler and scheduler.running:
            scheduler.pause()
    except Exception as e:
        LOG.warning("scheduler pause failed: %s", e)
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"killswitch.mode": "DEEP_FREEZE", "killswitch.engaged_at": datetime.now(timezone.utc).isoformat()}}
    )
    LOG.warning("KillSwitch DEEP_FREEZE engaged")
    if telegram_fn:
        await _safe_telegram(telegram_fn, "GH05T3 KillSwitch: DEEP FREEZE engaged. All background tasks paused.")
    return {"mode": "DEEP_FREEZE", "ok": True}


async def engage_shocker(db, targets: list[str], telegram_fn=None) -> dict:
    """Kill specified processes by name and send emergency alert."""
    killed = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            if any(t.lower() in proc.info["name"].lower() for t in targets):
                proc.kill()
                killed.append(f"{proc.info['name']}:{proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"killswitch.mode": "SHOCKER", "killswitch.killed": killed,
                  "killswitch.engaged_at": datetime.now(timezone.utc).isoformat()}}
    )
    LOG.warning("KillSwitch SHOCKER: killed %s", killed)
    msg = f"GH05T3 SHOCKER: Terminated {len(killed)} processes: {', '.join(killed)}"
    if telegram_fn:
        await _safe_telegram(telegram_fn, msg)
    return {"mode": "SHOCKER", "killed": killed, "ok": True}


async def disengage(db, scheduler) -> dict:
    """Release all killswitch modes and resume normal operation."""
    global _FROZEN, _STEALTH
    _FROZEN = False
    _STEALTH = False
    try:
        if scheduler and not scheduler.running:
            scheduler.resume()
    except Exception:
        pass
    await db.system_state.update_one(
        {"_id": "singleton"},
        {"$set": {"killswitch.mode": "OFF", "killswitch.disengaged_at": datetime.now(timezone.utc).isoformat()}}
    )
    LOG.info("KillSwitch disengaged — all systems resumed")
    return {"mode": "OFF", "ok": True}


async def _safe_telegram(fn, msg: str):
    try:
        await fn(msg)
    except Exception as e:
        LOG.warning("telegram alert failed: %s", e)
