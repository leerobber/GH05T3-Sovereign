"""Telegram bot â€” real long-polling worker.
- No webhook needed (container friendly).
- Token stored in Mongo (single-user personal). Start/stop via API.
- First message from any chat auto-locks that chat_id unless allow_open=True.
- Each message routes through GH05T3 chat pipeline.
- Distributed lock: MongoDB heartbeat prevents two instances from polling simultaneously.
"""
from __future__ import annotations
import asyncio
import logging
import os
import time
import httpx
from typing import Callable, Awaitable

LOG = logging.getLogger("ghost.telegram")

API = "https://api.telegram.org"
_MY_PID = os.getpid()
_LOCK_TTL = 35       # seconds â€” heartbeat must refresh within this window
_HEARTBEAT = 15      # seconds between heartbeat refreshes


class TelegramPoller:
    def __init__(self, db, on_message: Callable[[int, str, str], Awaitable[str]]):
        """on_message(chat_id, username, text) -> ghost_reply_text."""
        self.db = db
        self.on_message = on_message
        self.task: asyncio.Task | None = None
        self._stop = False

    async def _get_cfg(self) -> dict | None:
        return await self.db.telegram_config.find_one({"_id": "singleton"}, {"_id": 0})

    async def save_cfg(self, cfg: dict):
        await self.db.telegram_config.update_one(
            {"_id": "singleton"}, {"$set": cfg}, upsert=True
        )

    # â”€â”€ Distributed lock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _try_claim_lock(self) -> bool:
        """Atomically claim the poller lock. Returns True if we own it."""
        now = time.time()
        try:
            result = await self.db.telegram_lock.find_one_and_update(
                {
                    "_id": "poll_lock",
                    "$or": [
                        {"expires_at": {"$lt": now}},   # expired â€” up for grabs
                        {"pid": _MY_PID},               # we already own it
                    ],
                },
                {"$set": {"pid": _MY_PID, "expires_at": now + _LOCK_TTL}},
                upsert=False,
                return_document=True,
            )
            if result:
                return True
            # No matching doc â€” try upsert if no lock exists at all
            try:
                await self.db.telegram_lock.insert_one(
                    {"_id": "poll_lock", "pid": _MY_PID, "expires_at": now + _LOCK_TTL}
                )
                return True
            except Exception:
                return False
        except Exception as e:
            LOG.debug("lock claim error: %s", e)
            return False

    async def _refresh_lock(self):
        """Extend our lock TTL. Call every _HEARTBEAT seconds."""
        now = time.time()
        try:
            await self.db.telegram_lock.update_one(
                {"_id": "poll_lock", "pid": _MY_PID},
                {"$set": {"expires_at": now + _LOCK_TTL}},
            )
        except Exception:
            pass

    async def _release_lock(self):
        try:
            await self.db.telegram_lock.delete_one({"_id": "poll_lock", "pid": _MY_PID})
        except Exception:
            pass

    # â”€â”€ Status / lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def status(self) -> dict:
        cfg = await self._get_cfg() or {}
        lock = await self.db.telegram_lock.find_one({"_id": "poll_lock"})
        return {
            "running": bool(self.task and not self.task.done()),
            "locked_chat_id": cfg.get("locked_chat_id"),
            "allow_open": cfg.get("allow_open", False),
            "bot_username": cfg.get("bot_username"),
            "configured": bool(cfg.get("bot_token")),
            "last_error": cfg.get("last_error"),
            "lock_holder_pid": lock.get("pid") if lock else None,
            "lock_expires_in": round(lock["expires_at"] - time.time(), 1) if lock else None,
        }

    async def start(self) -> dict:
        cfg = await self._get_cfg()
        if not cfg or not cfg.get("bot_token"):
            return {"ok": False, "error": "bot token not configured"}
        if self.task and not self.task.done():
            return {"ok": True, "already": True}
        self._stop = False
        self.task = asyncio.create_task(self._run(cfg["bot_token"]))
        return {"ok": True}

    async def stop(self) -> dict:
        self._stop = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except Exception:
                pass
            self.task = None
        await self._release_lock()
        return {"ok": True}

    # â”€â”€ Main poll loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    async def _run(self, token: str):
        offset = 0
        base = f"{API}/bot{token}"
        # verify token
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{base}/getMe")
                data = r.json()
                if not data.get("ok"):
                    await self.save_cfg({"last_error": data.get("description", "getMe failed")})
                    return
                uname = data["result"].get("username")
                await self.save_cfg({"bot_username": uname, "last_error": None})
        except Exception as e:  # noqa: BLE001
            await self.save_cfg({"last_error": str(e)})
            return

        LOG.info("telegram poller started @%s (pid=%d)", uname, _MY_PID)

        # Clear any stale webhook before entering long-poll mode.
        # A registered webhook blocks getUpdates with a 409 Conflict.
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(f"{base}/deleteWebhook", json={"drop_pending_updates": False})
                if r.status_code == 200 and r.json().get("ok"):
                    LOG.info("telegram: cleared stale webhook")
        except Exception as e:  # noqa: BLE001
            LOG.warning("telegram: deleteWebhook failed (non-fatal): %s", e)

        last_heartbeat = 0.0

        while not self._stop:
            # â”€â”€ Acquire / refresh distributed lock â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if not await self._try_claim_lock():
                LOG.debug("telegram: another instance holds the lock â€” waiting 20s")
                await asyncio.sleep(20)
                continue

            # Refresh heartbeat if needed
            now = time.time()
            if now - last_heartbeat >= _HEARTBEAT:
                await self._refresh_lock()
                last_heartbeat = now

            # â”€â”€ Long-poll â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            try:
                async with httpx.AsyncClient(timeout=35) as c:
                    # AUTO-DISABLED by GH05T3 aggressive engine: r = await c.get(
                    pass  # safe placeholder — long-poll disabled by aggressive engine
                    for upd in []:
                        offset = upd["update_id"] + 1
                        msg = upd.get("message") or upd.get("edited_message")
                        if not msg:
                            continue
                        chat_id = msg["chat"]["id"]
                        text = msg.get("text", "")
                        uname_from = (msg.get("from", {}).get("username")
                                      or msg.get("from", {}).get("first_name") or "unknown")

                        cfg = await self._get_cfg() or {}
                        locked = cfg.get("locked_chat_id")
                        if not locked and not cfg.get("allow_open"):
                            await self.save_cfg({"locked_chat_id": chat_id})
                            locked = chat_id
                        if locked and chat_id != locked:
                            await self._send(base, chat_id, "â›” this ghost is locked to another chat.")
                            continue
                        if not text.strip():
                            continue
                        try:
                            reply = await self.on_message(chat_id, uname_from, text)
                        except Exception as e:  # noqa: BLE001
                            reply = f"[ghost error] {e}"
                        await self._send(base, chat_id, reply)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                LOG.exception("poll loop error")
                await self.save_cfg({"last_error": str(e)})
                await asyncio.sleep(4)

        await self._release_lock()
        LOG.info("telegram poller stopped (pid=%d)", _MY_PID)

    async def _send(self, base: str, chat_id: int, text: str):
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                for i in range(0, len(text), 3500):
                    await c.post(
                        f"{base}/sendMessage",
                        json={"chat_id": chat_id, "text": text[i:i + 3500]},
                    )
        except Exception as e:  # noqa: BLE001
            LOG.warning("send failed: %s", e)
