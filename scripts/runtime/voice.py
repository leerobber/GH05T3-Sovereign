"""GH05T3 wake-word + voice loop. All local, all free.

Pipeline:
    openwakeword listens for "hey ghost" / "hey jarvis" continuously
      → short silence-trimmed recording
      → faster-whisper local STT (runs on RTX 5050)
      → POST /api/chat
      → edge-tts neural voice synthesis
      → play reply through default speakers

If wake-word model isn't available, falls back to push-to-talk:
press F8 to start recording, release to send.
"""
from __future__ import annotations
import asyncio
import io
import logging
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import httpx
except ImportError:
    httpx = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    from openwakeword.model import Model as WakeModel
except ImportError:
    WakeModel = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ghost-voice")

API = os.environ.get("GHOST_GATEWAY_URL", "http://localhost:8001") + "/api"
SAMPLE_RATE = 16000
VAD_SILENCE_SEC = 0.9
MAX_REC_SEC = 12
VOICE = os.environ.get("GHOST_VOICE", "en-US-AvaMultilingualNeural")

_audio_q: "queue.Queue[np.ndarray]" = queue.Queue()


def _audio_cb(indata, frames, t, status):
    _audio_q.put(indata.copy().flatten())


def record_until_silence(pre_buffer: np.ndarray | None = None) -> np.ndarray:
    """Record 16 kHz mono audio until ~1s of silence."""
    chunks: list[np.ndarray] = []
    if pre_buffer is not None and len(pre_buffer):
        chunks.append(pre_buffer)
    silent = 0.0
    start = time.time()
    while True:
        try:
            data = _audio_q.get(timeout=0.5)
        except queue.Empty:
            continue
        chunks.append(data)
        rms = float(np.sqrt(np.mean(data * data)))
        if rms < 0.008:
            silent += len(data) / SAMPLE_RATE
        else:
            silent = 0.0
        if silent >= VAD_SILENCE_SEC:
            break
        if time.time() - start > MAX_REC_SEC:
            break
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def transcribe(audio: np.ndarray, model) -> str:
    if model is None:
        return ""
    segments, _ = model.transcribe(audio, language="en", vad_filter=True, beam_size=1)
    return " ".join(s.text.strip() for s in segments).strip()


def ask_ghost(text: str) -> str:
    if not httpx:
        return "[httpx missing]"
    try:
        with httpx.Client(timeout=120) as c:
            r = c.post(f"{API}/chat", json={"message": text, "session_id": "voice"})
            return r.json().get("ghost_message", {}).get("content", "")[:1500]
    except Exception as e:
        return f"[gateway error] {e}"


async def speak_async(text: str):
    if not edge_tts or not text.strip():
        LOG.info("[ghost says] %s", text)
        return
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "ghost_reply.mp3"
    comm = edge_tts.Communicate(text, VOICE, rate="+0%")
    await comm.save(str(tmp))
    # play via default player (Windows uses PlaySound via soundfile + sd)
    try:
        data, sr = sf.read(str(tmp), dtype="float32")
        sd.play(data, sr)
        sd.wait()
    except Exception:
        os.startfile(str(tmp))


def speak(text: str):
    try:
        asyncio.run(speak_async(text))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(speak_async(text))


# ---------------------------------------------------------------------------
def run_with_wakeword():
    assert WakeModel is not None, "openwakeword not installed"
    wake = WakeModel(inference_framework="onnx")   # downloads 'hey jarvis'/'alexa'/'hey mycroft'
    whisper = WhisperModel("base.en", device="auto", compute_type="int8") if WhisperModel else None
    LOG.info("voice loop running — say 'hey jarvis' / 'alexa' / 'hey mycroft' to wake GH05T3")
    LOG.info("(custom 'hey ghost' requires training your own model; Jarvis keyword works out of the box)")

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=1280, callback=_audio_cb)
    stream.start()
    while True:
        try:
            chunk = _audio_q.get(timeout=1)
        except queue.Empty:
            continue
        predictions = wake.predict((chunk * 32767).astype("int16"))
        triggered = any(v > 0.5 for v in predictions.values())
        if not triggered:
            continue
        LOG.info("wake word detected")
        speak("yes?")
        time.sleep(0.15)
        audio = record_until_silence()
        if len(audio) < SAMPLE_RATE * 0.5:
            continue
        text = transcribe(audio, whisper)
        if not text:
            continue
        LOG.info("you said: %s", text)
        reply = ask_ghost(text)
        LOG.info("ghost: %s", reply)
        speak(reply)


def run_push_to_talk():
    try:
        import keyboard
    except ImportError:
        LOG.error("install 'keyboard' package for push-to-talk. pip install keyboard")
        return
    whisper = WhisperModel("base.en", device="auto", compute_type="int8") if WhisperModel else None
    LOG.info("push-to-talk mode — hold F8 to speak")
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=1280, callback=_audio_cb)
    stream.start()
    while True:
        keyboard.wait("F8")
        # drain stale audio
        while not _audio_q.empty():
            _audio_q.get_nowait()
        LOG.info("recording (release F8 to send)...")
        chunks = []
        while keyboard.is_pressed("F8"):
            try:
                chunks.append(_audio_q.get(timeout=0.2))
            except queue.Empty:
                pass
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        if len(audio) < SAMPLE_RATE * 0.4:
            continue
        text = transcribe(audio, whisper)
        if not text:
            continue
        LOG.info("you said: %s", text)
        reply = ask_ghost(text)
        LOG.info("ghost: %s", reply)
        speak(reply)


def main():
    if WakeModel is not None:
        try:
            run_with_wakeword()
            return
        except Exception:
            LOG.exception("wake-word failed, falling back to push-to-talk")
    run_push_to_talk()


if __name__ == "__main__":
    main()
