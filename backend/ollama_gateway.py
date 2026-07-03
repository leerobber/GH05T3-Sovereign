"""Ollama gateway helpers for GH05T3 on LOQ (TatorTot).

Resolves `OLLAMA_GATEWAY_URL` from env/Mongo config, exposes health +
model list, and surfaces the models GH05T3 prefers:
    - Proposer / chat:  qwen2.5:7b-q4
    - Verifier / coder: deepseek-coder:6.7b
    - Critic:           llama3.1

GPU protection env vars (set in backend/.env):
    OLLAMA_MAX_CONCURRENT  max simultaneous GPU requests (default 1)
    OLLAMA_KEEP_ALIVE      seconds to keep model in VRAM after call;
                           0 = unload immediately, -1 = never unload (default 0)
    OLLAMA_NUM_CTX         context window tokens (default 2048, saves VRAM)
    OLLAMA_NUM_PREDICT     max output tokens per call (default 512)
"""
from __future__ import annotations

import asyncio
import os
import re
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LOG = logging.getLogger("ghost.ollama")

# VRAM-aware model defaults. Override via env vars or set OLLAMA_VRAM_GB
# to auto-select the right quant for your GPU.
#   4 GB  → tiny quants only (3b-q4)
#   6 GB  → 7b-q4_0
#   8 GB  → 7b-q4_K_M  ← RTX 5050 / 3060 Ti (recommended)
#   12 GB → 7b-q8 or 13b-q4
#   24 GB → 13b-q8 or 34b-q4

def _vram_model(default: str) -> str:
    """Pick a model quant that fits the configured VRAM budget."""
    gb = float(os.environ.get("OLLAMA_VRAM_GB", "0"))
    if gb <= 0:
        return default  # not configured — use the explicit env override or default
    if gb < 5:
        return "qwen2.5:3b-q4_K_M"
    if gb < 7:
        return "qwen2.5:7b-q4_0"
    if gb < 10:
        return "qwen2.5:7b-q4_K_M"
    if gb < 16:
        return "qwen2.5:7b-q8_0"
    return "qwen2.5:14b-q4_K_M"


PREFERRED = {
    "proposer": os.environ.get("OLLAMA_PROPOSER") or _vram_model("qwen2.5:7b-q4_K_M"),
    "verifier": os.environ.get("OLLAMA_VERIFIER") or _vram_model("deepseek-coder:6.7b"),
    "critic":   os.environ.get("OLLAMA_CRITIC")   or _vram_model("llama3.1"),
}

# ---------------------------------------------------------------------------
# Concurrency semaphore — shared across ALL callers in this process
# ---------------------------------------------------------------------------
_sem: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    """Lazy-init so env vars from load_dotenv are visible before creation."""
    global _sem
    if _sem is None:
        n = max(1, int(os.environ.get("OLLAMA_MAX_CONCURRENT", "1")))
        _sem = asyncio.Semaphore(n)
        LOG.info("OllamaGateway: semaphore created — max_concurrent=%d", n)
    return _sem


# ---------------------------------------------------------------------------
# Core guarded call — use this everywhere instead of raw httpx
# ---------------------------------------------------------------------------
async def call(
    model: str,
    system: str,
    user: str,
    timeout: int = 120,
) -> str:
    """Semaphore-guarded, VRAM-budgeted Ollama call.

    Acquires _get_sem() before touching the GPU so concurrent peer requests
    queue rather than OOM. Passes keep_alive and num_ctx to control VRAM
    lifetime and footprint per request.
    """
    url = resolved_url()
    if not url:
        raise RuntimeError("OLLAMA_GATEWAY_URL not configured")

    keep_alive_raw = os.environ.get("OLLAMA_KEEP_ALIVE", "0").strip()
    keep_alive = int(keep_alive_raw) if keep_alive_raw.lstrip("-").isdigit() else keep_alive_raw
    num_ctx     = int(os.environ.get("OLLAMA_NUM_CTX",     "2048"))
    num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "512"))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.6,
        "keep_alive": keep_alive,
        "options": {
            "num_ctx":     num_ctx,
            "num_predict": num_predict,
        },
    }

    async with _get_sem():
        LOG.debug("ollama call — model=%s ctx=%d keep_alive=%d", model, num_ctx, keep_alive)
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{url}/v1/chat/completions", json=payload)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

def resolved_url() -> str:
    return (os.environ.get("OLLAMA_GATEWAY_URL") or "").rstrip("/")


async def ping() -> dict:
    url = resolved_url()
    if not url:
        return {"reachable": False, "url": None, "models": [], "preferred": PREFERRED,
                "error": "OLLAMA_GATEWAY_URL not set"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{url}/v1/models")
            r.raise_for_status()
            j = r.json()
            models = [m.get("id") for m in j.get("data", []) if m.get("id")]
            return {
                "reachable": True, "url": url, "models": models,
                "preferred": PREFERRED,
                "has_proposer": any(PREFERRED["proposer"] in m for m in models),
                "has_verifier": any(PREFERRED["verifier"] in m for m in models),
            }
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "url": url, "models": [], "preferred": PREFERRED,
                "error": str(e)[:140]}


async def pull_model(model: str) -> dict:
    """Trigger a pull on the remote Ollama (non-blocking; returns immediately)."""
    url = resolved_url()
    if not url:
        return {"ok": False, "error": "OLLAMA_GATEWAY_URL not set"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            # Ollama's native pull endpoint is /api/pull (stream: false for sync)
            r = await c.post(f"{url}/api/pull", json={"name": model, "stream": False})
            r.raise_for_status()
            return {"ok": True, "model": model, "status": r.json().get("status", "ok")}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "model": model, "error": str(e)[:200]}


async def set_gateway_url(db, url: str) -> dict:
    """Persist gateway URL in Mongo + live env so reloads keep it.
    Validates URL shape before persistence."""
    url = (url or "").strip().rstrip("/")
    if url and not re.match(r"^https?://[\w\.\-]+(:\d+)?(/.*)?$", url):
        return {"reachable": False, "url": None, "error": "invalid url shape",
                "models": [], "preferred": PREFERRED}
    os.environ["OLLAMA_GATEWAY_URL"] = url
    await db.llm_config.update_one(
        {"_id": "ollama"}, {"$set": {"gateway_url": url}}, upsert=True,
    )
    return await ping()


async def load_gateway_url(db) -> str:
    """Call at startup to hydrate env from Mongo if set."""
    doc = await db.llm_config.find_one({"_id": "ollama"}, {"_id": 0})
    if doc and doc.get("gateway_url"):
        os.environ["OLLAMA_GATEWAY_URL"] = doc["gateway_url"]
        LOG.info("ollama: gateway url loaded from mongo: %s", doc["gateway_url"])
        return doc["gateway_url"]
    return ""
