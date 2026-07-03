"""LLM helpers — native multi-provider router, zero third-party SDK wrapper.

Premium / chat path  (Anthropic-first, falls back through free tiers):
    1. Anthropic Claude  (ANTHROPIC_API_KEY)
    2. Groq free tier    (GROQ_API_KEY)        — llama-3.3-70b-versatile
    3. Google Gemini     (GOOGLE_AI_KEY)        — gemini-2.0-flash
    4. Ollama            (OLLAMA_GATEWAY_URL)   — local, completely free

Cost-free / nightly path  (cheapest first):
    1. Ollama            — local, free
    2. Groq free tier    (GROQ_API_KEY)
    3. Google Gemini     (GOOGLE_AI_KEY)
    4. Anthropic         (ANTHROPIC_API_KEY)   — only if key present

Set LLM_PROVIDER=ollama to force Ollama for all calls.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from ollama_gateway import call as ollama_call, resolved_url as ollama_resolved_url
from ollama_gateway import PREFERRED as OLLAMA_PREFERRED

LOG = logging.getLogger("ghost.llm")

LLM_PROVIDER    = os.environ.get("LLM_PROVIDER",    "ollama")
LLM_MODEL       = os.environ.get("LLM_MODEL",       "claude-sonnet-4-6")
ANTHROPIC_MODEL = os.environ.get("LLM_MODEL",       "claude-sonnet-4-6")

# GH05T3 fine-tuned model — served by gh05t3_inference.py on port 8010
GH05T3_MODEL_URL = os.environ.get("GH05T3_MODEL_URL", "http://localhost:8010")

# Lemonade — AMD Radeon 780M iGPU (port 13305)

SOVEREIGN_MODELS_DIR = os.environ.get("SOVEREIGN_MODELS_DIR", "/home/leer4/sovereign-project/models")
AGENT_MODEL_MAP = {
    "FORGE":  os.environ.get("FORGE_MODEL",  "forge-sovereign"),
    "ORACLE": os.environ.get("ORACLE_MODEL", "oracle-sovereign"),
    "CODEX":  os.environ.get("CODEX_MODEL",  "codex-sovereign"),
    "NEXUS":  os.environ.get("NEXUS_MODEL",  "nexus-sovereign"),
    "AVERY":  os.environ.get("AVERY_MODEL",  "avery-sovereign"),
    "SENTINEL": os.environ.get("SENTINEL_MODEL", "sentinel-sovereign"),
}

def resolve_agent_model(agent: str | None) -> str | None:
    if not agent:
        return None
    return AGENT_MODEL_MAP.get(agent.upper())

LEMONADE_URL = os.environ.get("LEMONADE_URL", "http://localhost:13305")

_LOCAL_ONLY_PROVIDERS = {"ollama", "local", "free", "cost_free", "cost-free", "gh05t3"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------
class NoLLMError(RuntimeError):
    """No LLM provider is configured or all configured providers failed."""


# ---------------------------------------------------------------------------
# Rate-limit circuit breaker
# When a provider returns 429 / quota-exceeded it is skipped for _cooldown_
# seconds.  No state survives a process restart — that's intentional.
# ---------------------------------------------------------------------------
_rl_until: dict[str, float] = {}   # provider_name -> epoch-seconds


def _provider_ok(name: str) -> bool:
    """True if the provider is NOT in a rate-limit cooldown."""
    until = _rl_until.get(name, 0.0)
    if time.time() < until:
        LOG.debug("[cascade] %s cooling off (%.0fs left)", name, until - time.time())
        return False
    return True


def _mark_rl(name: str, seconds: float = 60.0) -> None:
    _rl_until[name] = time.time() + seconds
    LOG.warning("[cascade] %s rate-limited — skipping for %.0fs", name, seconds)


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(t in s for t in ("429", "rate_limit", "rate limit", "quota", "too many",
                                 "exceeded", "resource_exhausted"))


# ---------------------------------------------------------------------------
# MoE-style task classifier — sparse routing without an extra model call
# ---------------------------------------------------------------------------
_SECURITY_KW = frozenset({
    "exploit", "cve-", "cve_", "vulnerability", "payload", "shellcode", "reverse shell",
    "privilege escalation", "pentest", "penetration test", "malware", "rootkit",
    "lateral movement", "command injection", "sqli", "xss", "lfi", "rfi",
    "buffer overflow", "zero-day", "zero day", "rop chain", "heap spray",
})
_CODE_KW = frozenset({
    "def ", "async def ", "function ", "class ", "import ", "```python", "```js",
    "```ts", "```rust", "```go", "implement", "refactor", "debug this", "fix this",
    "type error", "traceback", "syntaxerror", "nameerror", "unit test", "unittest",
})
_RESEARCH_KW = frozenset({
    "research", "analyze", "analyse", "compare", "survey", "explain in detail",
    "comprehensive", "summarize", "summarise", "literature", "state of the art",
    "how does", "why does", "what is the difference", "pros and cons",
})


def _classify_task(user: str, system: str = "") -> str:
    """Classify a request into a routing tier — no extra model call required.

    Returns one of: 'security' | 'code' | 'research' | 'quick' | 'default'

    Routing intent:
      security  → GH05T3 fine-tuned model (trained on CVEs, threat analysis)
      code      → GH05T3 fine-tuned model (trained on reasoning + code datasets)
      research  → large cloud model (Groq 70B / Anthropic — need broad knowledge)
      quick     → smallest available local model (latency-first)
      default   → standard cascade unchanged
    """
    combined = (user + " " + system).lower()

    if any(kw in combined for kw in _SECURITY_KW):
        return "security"

    code_hits = sum(1 for kw in _CODE_KW if kw in combined)
    if code_hits >= 2:
        return "code"

    if any(kw in combined for kw in _RESEARCH_KW):
        return "research"

    if len(user.split(maxsplit=15)) < 15:
        return "quick"

    return "default"


# ---------------------------------------------------------------------------
# Nightly config (persisted in Mongo, overridable via API)
# ---------------------------------------------------------------------------
_DB_REF: dict = {"db": None}


def bind_db(db) -> None:
    _DB_REF["db"] = db


async def get_nightly_config() -> dict:
    db = _DB_REF["db"]
    if db is None:
        return {}
    doc = await db.llm_config.find_one({"_id": "nightly"}, {"_id": 0})
    return doc or {}


async def set_nightly_config(cfg: dict) -> dict:
    db = _DB_REF["db"]
    cfg = {k: v for k, v in cfg.items() if v is not None}
    await db.llm_config.update_one(
        {"_id": "nightly"}, {"$set": cfg}, upsert=True,
    )
    return await get_nightly_config()


# ---------------------------------------------------------------------------
# Native provider calls — no wrappers
# ---------------------------------------------------------------------------
async def _call_anthropic(system: str, user: str, model: str | None = None) -> str:
    import anthropic  # in requirements.txt
    key = _env_key("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=key)
    kwargs: dict = {
        "model": model or ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        kwargs["system"] = system
    resp = await client.messages.create(**kwargs)
    return resp.content[0].text


async def _call_groq(system: str, user: str,
                     model: str = "llama-3.3-70b-versatile",
                     api_key: str | None = None) -> str:
    key = api_key or _env_key("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    return await _openai_compat(
        "https://api.groq.com/openai/v1", key, model, system, user,
    )


async def _call_google(system: str, user: str,
                       model: str = "gemini-2.0-flash",
                       api_key: str | None = None) -> str:
    key = api_key or _env_key("GOOGLE_AI_KEY")
    if not key:
        raise RuntimeError("GOOGLE_AI_KEY not set")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 1024},
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(url, json=body)
        r.raise_for_status()
        j = r.json()
        try:
            return j["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            return json.dumps(j)[:500]


async def _call_openrouter(system: str, user: str,
                           model: str = "meta-llama/llama-3.2-3b-instruct:free",
                           api_key: str | None = None) -> str:
    """OpenRouter free-tier models — near-unlimited, no daily cap on :free models."""
    key = api_key or _env_key("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return await _openai_compat(
        "https://openrouter.ai/api/v1", key, model, system, user,
    )


def _all_groq_keys() -> list[str]:
    """Return all configured Groq keys (primary + rotation slots) in order."""
    keys = []
    for var in ("GROQ_API_KEY", "GROQ_API_KEY_2", "GROQ_API_KEY_3"):
        k = _env_key(var)
        if k:
            keys.append(k)
    return keys


async def _openai_compat(base: str, api_key: str | None,
                         model: str, system: str, user: str) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{base.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "temperature": 0.6,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# GH05T3 fine-tuned model
# ---------------------------------------------------------------------------
async def gh05t3_available() -> bool:
    """Return True if the local GH05T3 inference server is running."""
    if not GH05T3_MODEL_URL:
        return False
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{GH05T3_MODEL_URL}/health")
            return r.status_code == 200 and r.json().get("status") == "ready"
    except Exception:
        return False


async def _call_gh05t3(
    system: str,
    user: str,
    *,
    task_domain: str = "",
    session_id: str = "",
    temperature: float = 0.6,
) -> str:
    """Call local GH05T3 inference with Omni MoE routing (/v1/chat/completions)."""
    base = GH05T3_MODEL_URL.rstrip("/")
    url = f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "model": "gh05t3",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "task_domain": task_domain,
                "session_id": session_id,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------
async def lemonade_available() -> bool:
    """True if Lemonade server is running (AMD Radeon 780M iGPU, port 13305)."""
    if not LEMONADE_URL:
        return False
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{LEMONADE_URL}/api/v1/models")
            return r.status_code == 200
    except Exception:
        return False


async def _call_lemonade(system: str, user: str) -> str:
    model = os.environ.get("LEMONADE_MODEL", "Gemma-4-E2B-it-GGUF")
    return await _openai_compat(
        base    = f"{LEMONADE_URL}/api/v1",
        api_key = "lemonade",
        model   = model,
        system  = system,
        user    = user,
    )


async def ollama_available() -> bool:
    url = ollama_resolved_url()
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            r = await c.get(f"{url}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def ollama_ensure_model(model: str = "qwen2.5:0.5b") -> bool:
    """Pull a small Ollama model if Ollama is running but has no models loaded.
    qwen2.5:0.5b is ~400 MB — guaranteed local fallback."""
    url = ollama_resolved_url()
    if not url:
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{url}/api/tags")
            if r.status_code != 200:
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            if models:
                return True  # already has models
            # Pull the smallest useful model
            LOG.info("[ollama] no models found, pulling %s as fallback...", model)
            pull = await c.post(f"{url}/api/pull", json={"name": model}, timeout=600)
            return pull.status_code == 200
    except Exception as e:
        LOG.warning("[ollama] ensure_model failed: %s", e)
        return False


_ENV_PATH = Path(__file__).parent / ".env"


def _classify_anthropic_error(e: Exception) -> str:
    """Return a short human-readable reason for an Anthropic failure."""
    s = str(e).lower()
    if "rate_limit" in s or "429" in s:
        return "Anthropic rate limit hit"
    if any(w in s for w in ("quota", "usage", "exceeded", "credit", "budget")):
        return "Anthropic quota/budget exceeded"
    if "overloaded" in s or "529" in s or "503" in s:
        return "Anthropic API overloaded"
    if "401" in s or "authentication" in s or "invalid.*key" in s:
        return "Anthropic API key invalid"
    return f"Anthropic error: {type(e).__name__}"


def _env_key(name: str) -> str:
    """Read key from os.environ, then re-read .env file so hot-saved keys (written by
    the gateway process after server startup) are always picked up without a restart."""
    val = os.environ.get(name, "")
    if val:
        return val
    try:
        from dotenv import dotenv_values
        val = dotenv_values(_ENV_PATH).get(name, "") or ""
        if val:
            os.environ[name] = val  # cache into env so next call is fast
    except Exception:
        pass
    return val


_anthropic_key = lambda: _env_key("ANTHROPIC_API_KEY")
_groq_key = lambda: _env_key("GROQ_API_KEY")
_google_key = lambda: _env_key("GOOGLE_AI_KEY")


def _llm_provider() -> str:
    return (os.environ.get("LLM_PROVIDER") or LLM_PROVIDER or "ollama").strip().lower()


def _cost_free_only() -> bool:
    raw = os.environ.get("COST_FREE_ONLY")
    if raw is not None:
        return raw.strip().lower() not in _FALSE_VALUES
    return _llm_provider() in _LOCAL_ONLY_PROVIDERS


def _paid_llm_allowed() -> bool:
    return os.environ.get("ALLOW_PAID_LLM", "").strip().lower() in _TRUE_VALUES


async def _ollama_loaded_models() -> list[str]:
    """Return models currently loaded in Ollama VRAM (via /api/ps)."""
    url = ollama_resolved_url()
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{url}/api/ps")
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


async def _ollama_available_models() -> list[str]:
    """Return all models listed in Ollama (/api/tags)."""
    url = ollama_resolved_url()
    try:
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(f"{url}/api/tags")
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


async def _call_ollama_preferred(system: str, user: str, role: str = "proposer",
                                 model_override: str | None = None) -> tuple[str, str]:
    if not await ollama_available():
        raise RuntimeError("Ollama is not reachable at OLLAMA_GATEWAY_URL")

    primary = (model_override
               or os.environ.get("OLLAMA_SAGE_MODEL")
               or OLLAMA_PREFERRED.get(role)
               or OLLAMA_PREFERRED.get("proposer")
               or "qwen2.5:0.5b")

    try:
        text = await ollama_call(primary, system, user)
        return text, f"ollama:{primary}"
    except Exception as e:
        if "404" not in str(e):
            raise  # not a model-missing error — propagate

    LOG.debug("[ollama] %s → 404 (VRAM full?), trying loaded models first", primary)

    # Try whatever is already in VRAM (no reload cost, instant)
    for m in await _ollama_loaded_models():
        if m == primary:
            continue
        try:
            text = await ollama_call(m, system, user)
            LOG.debug("[ollama] fallback to loaded model %s succeeded", m)
            return text, f"ollama:{m}"
        except Exception:
            pass

    # Try other available models in preference order (smallest-first sovereign roster)
    _SOVEREIGN_FALLBACK = [
        "codex-sovereign:latest", "oracle-sovereign:latest", "nexus-sovereign:latest",
        "forge-sovereign:latest", "sentinel-sovereign:latest", "avery-sovereign:latest",
    ]
    available = set(await _ollama_available_models())
    for m in _SOVEREIGN_FALLBACK:
        if m in available and m != primary:
            try:
                text = await ollama_call(m, system, user)
                LOG.debug("[ollama] fallback to %s succeeded", m)
                return text, f"ollama:{m}"
            except Exception:
                pass

    raise RuntimeError(
        f"Ollama model {primary} returned 404 (VRAM full — close LM Studio to free space) "
        "and no fallback models responded."
    )


# ---------------------------------------------------------------------------
# Public chat functions
# ---------------------------------------------------------------------------
async def chat_once(session: str, system: str, user: str,
                    role: str = "proposer") -> tuple[str, str]:
    """Zero-downtime cascade — GH05T3 NEVER stops due to provider limits.

    Tier order (cheapest / most reliable first):
      0. Ollama local   — free, unlimited, runs on RTX 5050  (always tried first)
      1. Groq free tier — llama-3.3-70b, 14 400 req/day      (key rotation supported)
      2. Google Gemini  — gemini-2.0-flash, 1 M tokens/day
      3. OpenRouter     — :free models, near-unlimited
      4. Anthropic      — paid, ALLOW_PAID_LLM=1 required    (last resort)

    Rate-limit circuit breaker: any provider that returns 429 / quota-exceeded
    is automatically skipped for 60 s then retried — no manual intervention needed.
    """
    _fail_reason = ""
    cfg = await get_nightly_config()
    provider = _llm_provider()

    # ── MoE task routing — classify before cascading ───────────────────────────
    task = _classify_task(user, system)
    LOG.debug("[moe] session=%s task=%s provider=%s", session, task, provider)

    # security + code → prefer GH05T3 fine-tuned regardless of provider setting
    _prefer_local = task in {"security", "code"}
    # research → skip GH05T3 local, prefer large cloud models (need broad knowledge)
    _prefer_cloud = task == "research"
    # quick → use smallest available model (override model inside Ollama call)
    _prefer_small = task == "quick"

    # ── Tier -1: GH05T3 fine-tuned local model (highest priority) ─────────────
    # Research tasks (_prefer_cloud) skip local models in auto mode to reach
    # cloud LLMs with broader training data; explicit provider="gh05t3" always wins.
    if (provider == "gh05t3"
            or (provider == "auto" and not _prefer_cloud and await gh05t3_available())
            or (_prefer_local and not _prefer_cloud and await gh05t3_available())):
        try:
            text = await _call_gh05t3(
                system,
                user,
                task_domain=task if task != "default" else "",
                session_id=session,
            )
            return text, "gh05t3:local"
        except Exception as e:
            LOG.warning("gh05t3 local inference failed: %s", e)
            if provider == "gh05t3":
                raise NoLLMError(
                    f"GH05T3 inference server unavailable at {GH05T3_MODEL_URL}. "
                    "Run: python gh05t3_inference.py"
                ) from e

    # ── Fast-fail: forced to Ollama-only but Ollama is down ───────────────────
    if provider == "ollama" and _cost_free_only() and not await ollama_available():
        raise NoLLMError(
            "Ollama unavailable and COST_FREE_ONLY=1 with LLM_PROVIDER=ollama. "
            "Start Ollama (ollama serve) or set COST_FREE_ONLY=0 to enable cloud fallbacks."
        )

    # ── Tier 0a: SovereignCore gateway (OpenAI-compat, local GPU cluster) ─────
    # Routes inference across RTX 5050 → Radeon 780M → Ryzen 7 CPU via Ollama.
    # Preferred over direct Ollama because the gateway handles load balancing and
    # health-aware routing automatically. Skipped for research tasks (_prefer_cloud).
    if not _prefer_cloud and _provider_ok("sovereign"):
        try:
            from sovereign_economy import sovereign_available as _sov_ok, sovereign_chat
            if await _sov_ok():
                sc_model = os.environ.get("SOVEREIGN_MODEL", "qwen2.5:0.5b")
                text = await sovereign_chat(system, user, model=sc_model, timeout=60.0)
                return text, f"sovereign:{sc_model}"
        except Exception as _sov_exc:
            if _is_rate_limit(_sov_exc):
                _mark_rl("sovereign", 30)
            LOG.warning("[cascade] sovereign gateway skipped (model=%s, session=%s): %s",
                        os.environ.get("SOVEREIGN_MODEL", "qwen2.5:0.5b"), session, _sov_exc)

    # ── Tier 0b: Ollama direct (local, always free, no network needed) ────────
    # quick tasks bypass larger models and hit the smallest local model directly
    if _provider_ok("ollama") and await ollama_available():
        try:
            if _prefer_small:
                text, _ = await _call_ollama_preferred(system, user, role,
                                                       model_override="qwen2.5:0.5b")
                return text, "ollama:qwen2.5:0.5b"
            elif not _prefer_cloud:
                return await _call_ollama_preferred(system, user, role)
        except Exception as e:
            if _is_rate_limit(e):
                _mark_rl("ollama", 30)
            LOG.debug("[cascade] ollama failed: %s", e)
            if _cost_free_only():
                raise NoLLMError(
                    "Ollama unavailable and COST_FREE_ONLY=1. "
                    "Start Ollama (ollama serve) or set COST_FREE_ONLY=0 to enable "
                    "free cloud fallbacks (Groq, Gemini)."
                ) from e

    # ── Tier 0c: Lemonade (AMD Radeon 780M iGPU — free, local, always-on) ────
    # Kicks in when Ollama is down or the task was already handled above.
    # Uses Lemonade's optimised GGUF/Vulkan pipeline on the 780M.
    # Skipped for research tasks (_prefer_cloud) just like the other local tiers.
    if not _prefer_cloud and _provider_ok("lemonade") and await lemonade_available():
        try:
            text = await _call_lemonade(system, user)
            return text, "lemonade:780M"
        except Exception as e:
            if _is_rate_limit(e):
                _mark_rl("lemonade", 30)
            LOG.warning("[cascade] lemonade failed: %s", e)

    # ── Tier 1: Groq free tier (key rotation — tries all configured keys) ────
    # research tasks start here, skipping local models for broader knowledge
    if _provider_ok("groq"):
        groq_model = cfg.get("groq_model", "llama-3.3-70b-versatile")
        groq_keys  = _all_groq_keys() or ([cfg.get("groq_api_key")] if cfg.get("groq_api_key") else [])
        for idx, gk in enumerate(groq_keys):
            slot = "groq" if idx == 0 else f"groq_{idx+1}"
            if not _provider_ok(slot):
                continue
            try:
                text = await _call_groq(system, user, groq_model, api_key=gk)
                return text, f"groq:{groq_model}"
            except Exception as e:
                if _is_rate_limit(e):
                    _mark_rl(slot, 3600)   # Groq daily limit → cool off 1 h
                    LOG.warning("[cascade] groq key #%d daily limit hit", idx + 1)
                else:
                    LOG.warning("[cascade] groq key #%d failed: %s", idx + 1, e)
        _mark_rl("groq", 60)

    # ── Tier 2: Google Gemini free tier ──────────────────────────────────────
    if _provider_ok("google"):
        google_key   = _env_key("GOOGLE_AI_KEY") or cfg.get("google_api_key", "")
        google_model = cfg.get("google_model", "gemini-2.0-flash")
        if google_key:
            try:
                text = await _call_google(system, user, google_model, api_key=google_key)
                return text, f"google:{google_model}"
            except Exception as e:
                if _is_rate_limit(e):
                    _mark_rl("google", 3600)
                else:
                    LOG.warning("[cascade] google failed: %s", e)

    # ── Tier 3: OpenRouter free models ───────────────────────────────────────
    if _provider_ok("openrouter"):
        or_key = _env_key("OPENROUTER_API_KEY") or cfg.get("openrouter_api_key", "")
        if or_key:
            or_model = cfg.get("openrouter_model", "meta-llama/llama-3.3-70b-instruct:free")
            try:
                text = await _call_openrouter(system, user, or_model, api_key=or_key)
                return text, f"openrouter:{or_model}"
            except Exception as e:
                if _is_rate_limit(e):
                    _mark_rl("openrouter", 300)
                else:
                    LOG.warning("[cascade] openrouter failed: %s", e)

    # ── Tier 4: Anthropic — paid, explicit opt-in only ────────────────────────
    if _paid_llm_allowed() and _env_key("ANTHROPIC_API_KEY") and _provider_ok("anthropic"):
        try:
            text = await _call_anthropic(system, user)
            tag  = LLM_MODEL.split("-2025")[0].split("-2026")[0]
            return text, f"anthropic:{tag}"
        except Exception as e:
            _fail_reason = _classify_anthropic_error(e)
            if _is_rate_limit(e):
                _mark_rl("anthropic", 60)
            LOG.warning("[cascade] anthropic failed (%s): %s", _fail_reason, e)

    # ── All tiers exhausted ───────────────────────────────────────────────────
    reason = f" ({_fail_reason})" if _fail_reason else ""
    raise NoLLMError(
        f"All LLM providers exhausted{reason}. "
        "Ensure Ollama is running (ollama serve) — it is the always-on local backbone. "
        "Free cloud fallbacks: Groq (console.groq.com) · Gemini (aistudio.google.com) · "
        "OpenRouter (openrouter.ai). Set keys in LLM Config panel or backend/.env."
    )


async def _openai_tools_loop(
    base_url: str, api_key: str | None, model: str,
    system: str, user: str, tag: str, max_rounds: int = 8,
) -> tuple[str, str]:
    """Shared OpenAI-compat tool-use loop. Works for Ollama, Groq, OpenRouter."""
    import json as _json
    from ghost_tools import OPENAI_TOOLS, execute_tool

    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    for _ in range(max_rounds):
        body = {
            "model": model, "messages": messages,
            "tools": OPENAI_TOOLS, "tool_choice": "auto",
            "temperature": 0.6,
        }
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{base_url.rstrip('/')}/chat/completions",
                             headers=headers, json=body)
            r.raise_for_status()
        choice = r.json()["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            return choice.get("content") or "(no response)", tag
        messages.append({
            "role": "assistant",
            "content": choice.get("content") or "",
            "tool_calls": tool_calls,
        })
        for tc in tool_calls:
            fn = tc["function"]
            try:
                args = _json.loads(fn["arguments"])
            except Exception:
                args = {}
            result = await execute_tool(fn["name"], args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
    return "(tool loop limit reached)", tag


async def chat_with_tools(session: str, system: str, user: str,
                          role: str = "proposer") -> tuple[str, str]:
    """Agentic chat with real tools. 100% cost-free cascade:
      0. Ollama local  (qwen2.5:7b-instruct / mistral — tool-capable, runs on TatorTot)
      1. Groq free     (llama-3.3-70b-versatile, 14 400 req/day — tool-capable)
      2. OpenRouter    (:free models — tool-capable)
      3. Anthropic     (paid — only if ALLOW_PAID_LLM=1, skipped by default)
      4. Plain chat    (no tools but always responds)
    """
    from ghost_tools import execute_tool  # noqa: ensure importable

    # ── Tier 0: Ollama local (free, offline, TatorTot RTX 5050) ─────────────
    if await ollama_available():
        ollama_url = ollama_resolved_url()
        # Prefer tool-capable instruct models
        tool_models = [
            OLLAMA_PREFERRED.get("proposer", ""),
            "qwen2.5:7b-instruct", "qwen2.5:7b", "mistral:latest",
            "llama3.2:3b", "llama3:latest",
        ]
        for model in tool_models:
            if not model:
                continue
            try:
                text, tag = await _openai_tools_loop(
                    f"{ollama_url}/v1", None, model, system, user,
                    f"ollama:{model}",
                )
                return text, tag
            except Exception as e:
                if "does not support" in str(e).lower() or "404" in str(e):
                    continue  # model doesn't support tools, try next
                LOG.warning("[tools] ollama %s failed: %s", model, e)
                break  # ollama is up but errored — don't retry all models

    # ── Tier 1: Groq free tier (14 400 req/day, tool-capable) ───────────────
    groq_keys = _all_groq_keys()
    if groq_keys and _provider_ok("groq"):
        for idx, gk in enumerate(groq_keys):
            slot = "groq" if idx == 0 else f"groq_{idx+1}"
            if not _provider_ok(slot):
                continue
            try:
                text, tag = await _openai_tools_loop(
                    "https://api.groq.com/openai/v1", gk,
                    "llama-3.3-70b-versatile", system, user,
                    "groq:llama-3.3-70b-versatile",
                )
                return text, tag
            except Exception as e:
                if _is_rate_limit(e):
                    _mark_rl(slot, 3600)
                    LOG.warning("[tools] groq key #%d quota hit", idx + 1)
                else:
                    LOG.warning("[tools] groq key #%d failed: %s", idx + 1, e)

    # ── Tier 2: OpenRouter free models (near-unlimited :free tier) ───────────
    or_key = _env_key("OPENROUTER_API_KEY")
    if or_key and _provider_ok("openrouter"):
        or_models = [
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ]
        for model in or_models:
            try:
                text, tag = await _openai_tools_loop(
                    "https://openrouter.ai/api/v1", or_key,
                    model, system, user, f"openrouter:{model}",
                )
                return text, tag
            except Exception as e:
                if _is_rate_limit(e):
                    _mark_rl("openrouter", 300)
                    break
                LOG.warning("[tools] openrouter %s failed: %s", model, e)

    # ── Tier 3: Anthropic — paid, explicit opt-in only ───────────────────────
    if _paid_llm_allowed() and _env_key("ANTHROPIC_API_KEY") and _provider_ok("anthropic"):
        try:
            from ghost_tools import ANTHROPIC_TOOLS, execute_tool as _et
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=_env_key("ANTHROPIC_API_KEY"))
            messages: list[dict] = [{"role": "user", "content": user}]
            tag = f"anthropic:{ANTHROPIC_MODEL}"
            for _ in range(8):
                resp = await client.messages.create(
                    model=ANTHROPIC_MODEL, max_tokens=4096, system=system,
                    tools=ANTHROPIC_TOOLS, messages=messages,
                )
                tool_uses = [b for b in resp.content if b.type == "tool_use"]
                if not tool_uses:
                    text_blocks = [b for b in resp.content if b.type == "text"]
                    return (text_blocks[0].text if text_blocks else "(no response)"), tag
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for tu in tool_uses:
                    res = await _et(tu.name, tu.input)
                    results.append({"type": "tool_result", "tool_use_id": tu.id, "content": res})
                messages.append({"role": "user", "content": results})
        except Exception as e:
            if _is_rate_limit(e):
                _mark_rl("anthropic", 60)
            LOG.warning("[tools] anthropic tool loop failed: %s", e)

    # ── Tier 4: Plain chat fallback — no tools but always responds ───────────
    LOG.info("[tools] falling back to plain chat_once (no tools)")
    return await chat_once(session, system, user, role)


async def nightly_chat(session: str, system: str, user: str) -> tuple[str, str]:
    """Cost-free path for nightly/background work — same zero-downtime cascade as chat_once
    but never touches paid Anthropic unless ALLOW_PAID_LLM=1."""
    if _cost_free_only():
        try:
            return await _call_ollama_preferred(system, user, "proposer")
        except Exception as e:
            LOG.warning("ollama local-only nightly failed: %s", e)
            raise NoLLMError(
                "Ollama unavailable for nightly work and COST_FREE_ONLY=1. "
                "Start Ollama or set COST_FREE_ONLY=0 to enable free cloud fallbacks."
            ) from e

    cfg = await get_nightly_config()
    provider = cfg.get("nightly_provider") or _auto_pick_provider(cfg)

    # --- Explicit Mongo config ---
    if provider == "google" and cfg.get("google_api_key"):
        try:
            model = cfg.get("google_model", "gemini-2.0-flash")
            text = await _call_google(system, user, model, cfg["google_api_key"])
            return text, f"google:{model}"
        except Exception as e:
            LOG.warning("google (mongo key) failed: %s", e)

    if provider == "groq" and cfg.get("groq_api_key"):
        try:
            model = cfg.get("groq_model", "llama-3.3-70b-versatile")
            text = await _call_groq(system, user, model, cfg["groq_api_key"])
            return text, f"groq:{model}"
        except Exception as e:
            LOG.warning("groq (mongo key) failed: %s", e)

    if provider == "ollama" and await ollama_available():
        try:
            model = cfg.get("ollama_model", "qwen2.5")
            text = await ollama_call(model, system, user)
            return text, f"ollama:{model}"
        except Exception as e:
            LOG.warning("ollama (config) failed: %s", e)

    # --- Auto-detect: cheapest first ---

    # Ollama — completely free, local; auto-pull tiny model if needed
    if await ollama_available():
        await ollama_ensure_model("qwen2.5:0.5b")
        try:
            text = await ollama_call("qwen2.5", system, user)
            return text, "ollama:qwen2.5"
        except Exception as e:
            LOG.warning("auto ollama failed: %s", e)

    # Groq env key — free tier (pass key explicitly so hot-reload works)
    groq_key = _env_key("GROQ_API_KEY")
    if groq_key:
        try:
            text = await _call_groq(system, user, api_key=groq_key)
            return text, "groq:llama-3.3-70b-versatile"
        except Exception as e:
            LOG.warning("env groq failed: %s", e)

    # Google env key — free tier
    google_key = _env_key("GOOGLE_AI_KEY")
    if google_key:
        try:
            text = await _call_google(system, user, api_key=google_key)
            return text, "google:gemini-2.0-flash"
        except Exception as e:
            LOG.warning("env google failed: %s", e)

    # Anthropic — paid, last resort for nightly and explicit opt-in only
    _fail_reason = ""
    if _paid_llm_allowed() and _env_key("ANTHROPIC_API_KEY"):
        try:
            text = await _call_anthropic(system, user)
            tag = LLM_MODEL.split("-2025")[0].split("-2026")[0]
            return text, f"anthropic:{tag}"
        except Exception as e:
            _fail_reason = _classify_anthropic_error(e)
            LOG.warning("anthropic nightly failed (%s): %s", _fail_reason, e)

    reason = f" ({_fail_reason})" if _fail_reason else ""
    raise NoLLMError(
        f"No LLM provider available for nightly chat{reason}. "
        "Set GROQ_API_KEY or GOOGLE_AI_KEY in the LLM Config panel for free fallback."
    )


def _auto_pick_provider(cfg: dict) -> str:
    if cfg.get("google_api_key"):
        return "google"
    if cfg.get("groq_api_key"):
        return "groq"
    return "auto"


async def nightly_status() -> dict:
    cfg = await get_nightly_config()
    ollama_ok, lemonade_ok, sovereign_ok = await asyncio.gather(
        ollama_available(), lemonade_available(), sovereign_available(),
    )
    return {
        "provider":            cfg.get("nightly_provider") or _auto_pick_provider(cfg),
        "has_anthropic_key":   bool(_env_key("ANTHROPIC_API_KEY")),
        "has_google_key":      bool(cfg.get("google_api_key") or _env_key("GOOGLE_AI_KEY")),
        "has_groq_key":        bool(cfg.get("groq_api_key")   or _env_key("GROQ_API_KEY")),
        "google_model":        cfg.get("google_model",  "gemini-2.0-flash"),
        "groq_model":          cfg.get("groq_model",    "llama-3.3-70b-versatile"),
        "ollama_reachable":    ollama_ok,
        "lemonade_reachable":  lemonade_ok,
        "sovereign_available": sovereign_ok,
        "fallback_chain":      ["sovereign-core (local GPU)", "ollama (local)",
                                "lemonade (780M iGPU)", "groq (free)", "google (free)", "anthropic"],
    }


def _json_block(s: str) -> dict | None:
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SAGE cycle — MAP-Elites emitter integration
# ---------------------------------------------------------------------------
PROPOSER_SYS = """You are the GH05T3 SAGE Proposer agent.
Propose ONE concrete, self-improvement change to GH05T3 that would measurably
improve KAIROS, HCM, Memory Palace, Ghost Protocol, or a sub-agent.
Under 25 words. Technical, specific, shippable. No fluff."""

CRITIC_SYS = """You are the GH05T3 SAGE Critic. You are a different model than the Proposer
(critic-capture prevention is sacred). Given a proposal, respond with strict JSON:
{"decision":"APPROVE|REJECT|REVISE","reason":"<<=25 words>>"}"""

VERIFIER_SYS = """You are the GH05T3 SAGE Verifier.
Decide if the proposal is technically coherent and sound.
Respond strict JSON: {"verdict":"PASS|PARTIAL|FAIL","rationale":"<<=20 words>>"}"""

# MAP-Elites batch state — targets from ask(), results awaiting tell()
_me_targets: list[dict] = []
_me_pending: list[tuple[float, float, int]] = []   # (objective, latency_s, tokens)
_me_ask_count: int = 0


def _me_next_target() -> dict | None:
    """Pop next emitter target, refilling from archive.ask() when buffer is empty."""
    global _me_targets, _me_ask_count
    if not _me_targets:
        try:
            from evolution.map_elites import ask
            targets = ask()
            if targets:
                _me_targets = list(targets)
                _me_ask_count = len(_me_targets)
                LOG.debug("[sage/me] refilled %d targets from emitter", len(_me_targets))
        except Exception as e:
            LOG.debug("[sage/me] ask() skipped: %s", e)
    return _me_targets.pop(0) if _me_targets else None


def _me_record(objective: float, latency_s: float, tokens: int) -> None:
    """Accumulate one result; flush tell() when a full batch is ready.

    The batch size matches what the last ask() returned -- tell() requires
    exactly the same number of results as the last ask() gave targets.
    """
    global _me_pending, _me_ask_count
    _me_pending.append((objective, latency_s, tokens))
    if _me_ask_count > 0 and len(_me_pending) >= _me_ask_count:
        try:
            from evolution.map_elites import tell
            objectives = [r[0] for r in _me_pending]
            measures = [[r[0], r[1] * 1000, r[2]] for r in _me_pending]  # latency_s -> ms
            me_batch = _me_ask_count
            tell(objectives, measures)
            LOG.debug("[sage/me] tell() flushed %d results to emitter", len(_me_pending))
        except Exception as e:
            LOG.debug("[sage/me] tell() failed: %s", e)
        finally:
            _me_pending.clear()
            _me_ask_count = 0


def _proposer_sys_for_target(target: dict | None) -> str:
    """Inject MAP-Elites target constraints into the proposer system prompt.

    quality_target drives specificity:  high → exploit known good regions
                                        low  → explore novel approaches
    token_budget drives brevity: tighter budget → shorter, punchier proposals
    """
    if not target:
        return PROPOSER_SYS

    qt = target.get("quality_target", 0.75)
    tb = int(target.get("token_budget", 500))

    if qt >= 0.85:
        style = "Be maximally specific and immediately implementable — exploit known good patterns."
    elif qt >= 0.60:
        style = "Be concrete but try a novel approach area not explored recently."
    else:
        style = "Be bold and exploratory — propose an unconventional angle even if uncertain."

    word_limit = max(10, min(25, tb // 20))

    return (
        f"You are the GH05T3 SAGE Proposer agent.\n"
        f"Propose ONE concrete, self-improvement change to GH05T3 that would measurably\n"
        f"improve KAIROS, HCM, Memory Palace, Ghost Protocol, or a sub-agent.\n"
        f"Under {word_limit} words. {style}"
    )


async def run_sage_cycle(cycle_num: int, use_nightly: bool = True) -> dict:
    async def _call(session, system, user, role="proposer"):
        if use_nightly:
            return await nightly_chat(session, system, user)
        return await chat_once(session, system, user, role)

    # ── MAP-Elites: get target for this cycle ──────────────────────────────────
    me_target   = _me_next_target()
    proposer_sys = _proposer_sys_for_target(me_target)

    session    = f"sage-{cycle_num}"
    t0         = time.monotonic()
    proposal, proposer_tag = await _call(session, proposer_sys,
                                         f"Propose improvement #{cycle_num}. Be distinctive.")
    latency_s  = round(time.monotonic() - t0, 3)
    proposal   = proposal.strip().split("\n")[0][:220]
    token_est  = int(len(proposal.split()) * 1.3)   # fast token estimate

    # critic + verifier are independent — run them in parallel (saves ~1 LLM round-trip per cycle)
    (critic_raw, critic_tag), (verifier_raw, verifier_tag) = await asyncio.gather(
        _call(f"{session}-critic",   CRITIC_SYS,   f"Proposal: {proposal}\nRespond with JSON only.", "critic"),
        _call(f"{session}-verifier", VERIFIER_SYS, f"Proposal: {proposal}\nRespond with JSON only.", "verifier"),
    )

    cj = _json_block(critic_raw) or {"decision": "REVISE", "reason": "critic parse failed"}
    decision = (cj.get("decision") or "REVISE").upper()
    if decision not in {"APPROVE", "REJECT", "REVISE"}:
        decision = "REVISE"

    vj = _json_block(verifier_raw) or {"verdict": "PARTIAL", "rationale": "verifier parse failed"}
    verdict = (vj.get("verdict") or "PARTIAL").upper()
    if verdict not in {"PASS", "PARTIAL", "FAIL"}:
        verdict = "PARTIAL"

    base  = {"PASS": 1.0, "PARTIAL": 0.6, "FAIL": 0.2}[verdict]
    mult  = {"APPROVE": 1.0, "REJECT": 0.5, "REVISE": 0.75}[decision]
    final = round(base * mult, 3)
    elite    = final >= 0.85
    archived = final >= 0.70 or verdict == "PASS"

    # ── MAP-Elites: feed result back to emitter ────────────────────────────────
    _me_record(final, latency_s, token_est)

    return {
        "cycle_num":          cycle_num,
        "proposer":           proposer_tag,
        "critic":             critic_tag,
        "verifier":           verifier_tag,
        "proposal":           proposal,
        "critic_decision":    decision,
        "critic_reason":      cj.get("reason", "")[:200],
        "verdict":            verdict,
        "verifier_rationale": vj.get("rationale", "")[:200],
        "base_score":         base,
        "multiplier":         mult,
        "final_score":        final,
        "archived":           archived,
        "elite":              elite,
        # MAP-Elites telemetry
        "me_target":          me_target,
        "latency_s":          latency_s,
        "token_est":          token_est,
    }


# ---------------------------------------------------------------------------
# Cassandra pre-mortem
# ---------------------------------------------------------------------------
CASSANDRA_SYS = """You are Cassandra — GH05T3's pre-mortem oracle. Given a proposed
change or launch, write a vivid short autopsy from 6 months in the future where
it failed. 1) What shipped. 2) What went wrong. 3) Root cause. 4) Mitigation to
apply before launch. Max 140 words. No fluff."""


async def cassandra_premortem(scenario: str) -> str:
    text, _ = await nightly_chat("cassandra", CASSANDRA_SYS, scenario)
    return text.strip()


async def load_economy_context() -> str:
    """Load live SovereignCore economy metrics for system prompt injection.

    Fetches health, KAIROS cycles, and ledger stats from the SovereignCore
    gateway (http://localhost:8000 by default via SOVEREIGN_CORE_URL).
    Falls back to local data files if the gateway is unreachable.
    Returns an empty string on total failure — never pollutes the prompt.
    """
    # Primary: live SovereignCore gateway
    try:
        from sovereign_economy import load_sovereign_economy_context
        ctx = await load_sovereign_economy_context()
        if ctx:
            return ctx
    except Exception:
        pass

    # Fallback: local data files (sovereign-core cloned next to GH05T3)
    import json as _json
    from pathlib import Path
    try:
        root = Path(__file__).parent.parent
        parts: list[str] = []
        spin_file = root / "data" / "spin_dataset.jsonl"
        if spin_file.exists():
            count = sum(1 for ln in spin_file.open(encoding="utf-8") if ln.strip())
            parts.append(f"SPIN training pairs: {count}")
        state_file = root / "data" / "continuous_state.json"
        if state_file.exists():
            s = _json.loads(state_file.read_text(encoding="utf-8"))
            parts.append(f"Flywheel cycles: {s.get('total_cycles', 0)}")
        if parts:
            return "[SovereignNation Economy]\n" + "\n".join(parts)
    except Exception:
        pass
    return ""


async def sovereign_available() -> bool:
    """True if the SovereignCore gateway (port 8000) is reachable."""
    try:
        from sovereign_economy import sovereign_available as _sa
        return await _sa()
    except Exception:
        return False
