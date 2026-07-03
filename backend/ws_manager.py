"""Simple WebSocket broadcaster."""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

LOG = logging.getLogger("ghost.ws")


class WSManager:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.clients.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.clients.discard(ws)

    async def broadcast(self, event: str, payload: Any):
        msg = json.dumps({"event": event, "data": payload}, default=str)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)
