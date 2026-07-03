"""GH05T3 whisper listener â€” native edge-tts voice of "stuck alerts" etc.

Runs alongside the main voice loop (or standalone). Subscribes to
ws://.../api/ws and speaks every `ghosteye_whisper` event locally using
edge-tts. High-priority whispers interrupt the current utterance.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path

import sounddevice as sd
# AUTO-DISABLED by GH05T3 aggressive engine: import soundfile as sf
pass  # safe placeholder
import websockets
import edge_tts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ghost-whisper")

GATEWAY = os.environ.get("GHOST_GATEWAY_URL", "http://localhost:8001")
DEFAULT_VOICE = os.environ.get("GHOST_VOICE", "en-US-AvaMultilingualNeural")
TMP = Path(os.environ.get("TEMP", "/tmp")) / "ghost_whispers"
TMP.mkdir(parents=True, exist_ok=True)

_play_lock = asyncio.Lock()
_current_task: asyncio.Task | None = None


async def synth_and_play(text: str, voice: str, priority: str):
    global _current_task
    if priority == "high" and _current_task and not _current_task.done():
        sd.stop()
    async with _play_lock:
        path = TMP / f"w-{abs(hash(text)) % 10**9}.mp3"
        try:
            await edge_tts.Communicate(text, voice, rate="+0%").save(str(path))
            data, sr = sf.read(str(path), dtype="float32")
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            LOG.warning("tts failed: %s", e)


async def run():
    ws_url = GATEWAY.replace("http", "ws") + "/api/ws"
    backoff = 2
    while True:
        try:
            LOG.info("connecting to %s", ws_url)
            async with websockets.connect(ws_url) as ws:
                LOG.info("connected; listening for whispers")
                backoff = 2
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue
                    if msg.get("event") != "ghosteye_whisper":
                        continue
                    d = msg.get("data") or {}
                    text = (d.get("text") or "").strip()
                    if not text:
                        continue
                    voice = d.get("voice") or DEFAULT_VOICE
                    priority = d.get("priority") or "normal"
                    LOG.info("whisper [%s] %s", priority, text[:120])
                    global _current_task
                    _current_task = asyncio.create_task(synth_and_play(text, voice, priority))
        except Exception as e:
            LOG.warning("ws error: %s â€” retry in %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(60, backoff * 2)


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
