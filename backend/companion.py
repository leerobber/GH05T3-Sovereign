"""GH05T3 Companion — pairing + WebSocket relay.

Companion agent (running on Robert's laptop/Android) dials OUT to /api/companion/ws
with the pairing token. Dashboard sends commands via /api/companion/command; the
relay forwards them to the companion over its WebSocket and returns the response.
"""
from __future__ import annotations
import asyncio
import json
import logging
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

LOG = logging.getLogger("ghost.companion")
router = APIRouter(prefix="/api/companion")

# In-memory state (single-user personal build)
_pair_codes: dict[str, dict] = {}          # code -> {token, expires, consumed}
_companions: dict[str, "Companion"] = {}    # token -> Companion
_pending: dict[str, asyncio.Future] = {}    # req_id -> future


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Companion:
    def __init__(self, token: str, info: dict, ws: WebSocket):
        self.token = token
        self.info = info
        self.ws = ws
        self.connected_at = _now()
        self.last_seen = _now()
        self.capabilities: set[str] = set(info.get("capabilities", []))
        self.label = info.get("label", "unknown")


def _db_ref() -> dict:
    return {"db": None}


_DB = _db_ref()


def bind_db(db):
    _DB["db"] = db


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
@router.post("/pair")
async def create_pair_code():
    """Dashboard calls this to get a 6-digit code. Code expires in 10 min."""
    code = f"{secrets.randbelow(1_000_000):06d}"
    token = secrets.token_urlsafe(24)
    _pair_codes[code] = {
        "token": token,
        "expires": _now() + timedelta(minutes=10),
        "consumed": False,
    }
    return {"code": code, "expires_in": 600}


@router.post("/claim")
async def claim_pair_code(code: str, label: str = "laptop"):
    """Companion agent calls this with the code it got from the user."""
    entry = _pair_codes.get(code)
    if not entry:
        raise HTTPException(404, "invalid pairing code")
    if entry["consumed"]:
        raise HTTPException(409, "code already used")
    if _now() > entry["expires"]:
        raise HTTPException(410, "code expired")
    entry["consumed"] = True
    return {"token": entry["token"], "label": label}


@router.get("/status")
async def companion_status():
    out = []
    for tok, c in _companions.items():
        out.append({
            "token_hint": tok[:6] + "…",
            "label": c.label,
            "capabilities": sorted(c.capabilities),
            "connected_at": c.connected_at.isoformat(),
            "last_seen": c.last_seen.isoformat(),
            "info": {k: v for k, v in c.info.items() if k != "capabilities"},
        })
    return {"connected": out, "pending_pair_codes": len([
        c for c in _pair_codes.values() if not c["consumed"] and _now() <= c["expires"]
    ])}


class CompanionCmd(BaseModel):
    action: str  # screenshot | shell | fs_read | fs_write | clipboard_read | clipboard_write | notify
    args: dict = {}
    token: str | None = None  # if None, sends to first connected companion


@router.post("/command")
async def send_command(cmd: CompanionCmd):
    """Dashboard → companion. Returns the companion's response."""
    # pick target
    companion: Companion | None = None
    if cmd.token:
        companion = _companions.get(cmd.token)
    elif _companions:
        companion = next(iter(_companions.values()))
    if not companion:
        raise HTTPException(404, "no companion connected")

    # permission check
    cap_required = {
        "screenshot": "screen_read",
        "shell": "shell_exec",
        "fs_read": "fs_read",
        "fs_write": "fs_write",
        "clipboard_read": "clipboard",
        "clipboard_write": "clipboard",
        "notify": "notify",
    }.get(cmd.action)
    if cap_required and cap_required not in companion.capabilities:
        raise HTTPException(403, f"companion did not grant capability: {cap_required}")

    req_id = str(uuid.uuid4())
    fut: asyncio.Future = asyncio.get_event_loop().create_future()
    _pending[req_id] = fut
    try:
        await companion.ws.send_json({"req_id": req_id, "action": cmd.action, "args": cmd.args})
        result = await asyncio.wait_for(fut, timeout=30)
        return {"ok": True, "result": result}
    except asyncio.TimeoutError:
        raise HTTPException(504, "companion did not respond in time")
    finally:
        _pending.pop(req_id, None)


@router.post("/revoke")
async def revoke_companion(token: str):
    c = _companions.get(token)
    if not c:
        raise HTTPException(404, "no such companion")
    try:
        await c.ws.close()
    except Exception:
        pass
    _companions.pop(token, None)
    return {"ok": True}


@router.post("/ghosteye/toggle")
async def ghosteye_toggle(enabled: bool):
    """Tell connected companion(s) to pause or resume GhostEye streaming."""
    await broadcast_to_all({"control": "ghosteye", "enabled": enabled})
    return {"ok": True, "enabled": enabled, "broadcast_to": len(_companions)}


# ---------------------------------------------------------------------------
# WebSocket relay
# ---------------------------------------------------------------------------
async def _accept_companion_ws(ws: WebSocket, event_handler=None):
    """Accept a companion WS. First message must be auth {token, label, capabilities, info}.
    event_handler(companion, event_name, payload) is called for unsolicited messages."""
    await ws.accept()
    try:
        auth = await asyncio.wait_for(ws.receive_json(), timeout=10)
    except Exception:
        await ws.close(code=1008)
        return
    token = auth.get("token")
    valid = any(
        e["consumed"] and e["token"] == token for e in _pair_codes.values()
    )
    if not valid:
        valid = token in _companions
    if not token or not valid:
        await ws.send_json({"error": "invalid token"})
        await ws.close(code=1008)
        return

    companion = Companion(token, auth, ws)
    old = _companions.get(token)
    if old:
        try:
            await old.ws.close()
        except Exception:
            pass
    _companions[token] = companion
    await ws.send_json({"event": "hello", "capabilities": sorted(companion.capabilities),
                         "label": companion.label})
    LOG.info("companion connected label=%s caps=%s", companion.label, companion.capabilities)

    try:
        while True:
            msg = await ws.receive_json()
            companion.last_seen = _now()
            rid = msg.get("req_id")
            if rid and rid in _pending:
                fut = _pending.pop(rid)
                if not fut.done():
                    fut.set_result(msg.get("result") or msg.get("error") or {})
                continue
            # unsolicited push
            event = msg.get("event")
            if event and event_handler:
                try:
                    await event_handler(companion, event, msg.get("data") or msg)
                except Exception:
                    LOG.exception("event handler failed")
    except WebSocketDisconnect:
        LOG.info("companion disconnected label=%s", companion.label)
    except Exception:
        LOG.exception("companion ws error")
    finally:
        if _companions.get(token) is companion:
            _companions.pop(token, None)


bind_ws = _accept_companion_ws


async def broadcast_to_all(payload: dict):
    """Send a message to every connected companion. Used for ghosteye pause/resume, kill-switch."""
    dead = []
    for tok, c in list(_companions.items()):
        try:
            await c.ws.send_json(payload)
        except Exception:
            dead.append(tok)
    for tok in dead:
        _companions.pop(tok, None)
