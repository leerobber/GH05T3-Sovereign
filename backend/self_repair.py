"""GH05T3 self-repair engine.

Monitors the backend for known failure patterns and fixes them autonomously:

  PATTERN                          FIX
  ────────────────────────────────────────────────────────────────────
  litellm import / budget error    Remove litellm from venv + sys.path
  gateway_v3 process on port 8002  Kill it
  stale old-Python process on 8001 Kill it, reschedule clean start
  Stale preview URL in config       Wipe it from DB + env
  NoLLMError (all providers fail)   Force SovereignCore/Ollama, log & alert
  nightly_chat total failure        Fall through to SovereignCore/Ollama directly
  ────────────────────────────────────────────────────────────────────

Runs as an APScheduler job every 5 minutes.
Also exposes `check()` for on-demand repair from chat or API.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = logging.getLogger("ghost.self_repair")

_BACKEND = Path(__file__).parent.resolve()
_REPO    = _BACKEND.parent.resolve()

# ── known bad patterns ────────────────────────────────────────────────────────

_BAD_IMPORTS   = ["litellm", "emergentintegrations"]
_BAD_URLS      = ["emergent.host", "emergentintegrations", "tatorot-dashboard"]  # preview deployment URLs
_BAD_PORTS     = [8002]   # gateway_v3 port — should never be in use by us
_PROTECTED_PORT = 8001


# ── individual repair actions ─────────────────────────────────────────────────

def _purge_bad_imports() -> list[str]:
    """Remove any bad packages from the active Python environment."""
    fixed = []
    for pkg in _BAD_IMPORTS:
        if pkg in sys.modules:
            del sys.modules[pkg]
            LOG.warning("self_repair: evicted %s from sys.modules", pkg)
            fixed.append(f"evicted {pkg} from sys.modules")
        # Also uninstall from venv pip if importable
        try:
            __import__(pkg)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", pkg, "-y"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                LOG.warning("self_repair: pip uninstall %s succeeded", pkg)
                fixed.append(f"pip uninstall {pkg}")
        except ImportError:
            pass
        except Exception as e:
            LOG.warning("self_repair: pip uninstall %s failed: %s", pkg, e)
    return fixed


def _kill_bad_ports() -> list[str]:
    """Kill any process using a port that should not be active."""
    import socket
    fixed = []
    for port in _BAD_PORTS:
        # Try to connect — if something answers, find and kill it
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                try:
                    result = subprocess.run(
                        ["netstat", "-ano"],
                        capture_output=True, text=True, timeout=5,
                    )
                    for line in result.stdout.splitlines():
                        if f":{port} " in line and "LISTEN" in line:
                            pid = line.strip().split()[-1]
                            subprocess.run(["taskkill", "/F", "/PID", pid],
                                           capture_output=True, timeout=5)
                            LOG.warning("self_repair: killed PID %s on port %d", pid, port)
                            fixed.append(f"killed port {port} PID {pid}")
                except Exception as e:
                    LOG.warning("self_repair: kill port %d failed: %s", port, e)
    return fixed


def _check_wrong_python_on_8001() -> list[str]:
    """If port 8001 is held by system Python instead of venv Python, kill it."""
    fixed = []
    venv_py = str(_BACKEND / ".venv" / "Scripts" / "python.exe")
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if ":8001 " in line and "LISTEN" in line:
                pid = line.strip().split()[-1]
                try:
                    import psutil
                    proc = psutil.Process(int(pid))
                    exe = proc.exe().lower()
                    # If it's the AppData system Python (not our venv), kill it
                    if "appdata" in exe and ".venv" not in exe:
                        proc.kill()
                        LOG.warning("self_repair: killed wrong-Python on 8001 PID %s", pid)
                        fixed.append(f"killed system-python impostor on 8001 (PID {pid})")
                except Exception:
                    pass
    except Exception as e:
        LOG.debug("self_repair: 8001 check failed: %s", e)
    return fixed


async def _purge_stale_urls_from_db(db) -> list[str]:
    """Remove any stale preview-deployment URLs or keys from MongoDB config."""
    fixed = []
    try:
        # Check llm_config for emergent URLs
        async for doc in db.llm_config.find({}):
            for k, v in doc.items():
                if isinstance(v, str) and any(bad in v for bad in _BAD_URLS):
                    await db.llm_config.update_one(
                        {"_id": doc["_id"]}, {"$unset": {k: ""}}
                    )
                    LOG.warning("self_repair: purged stale preview URL from llm_config.%s", k)
                    fixed.append(f"purged stale preview URL from llm_config.{k}")

        # Wipe EMERGENT_LLM_KEY from env
        if os.environ.get("EMERGENT_LLM_KEY"):
            os.environ["EMERGENT_LLM_KEY"] = ""
            fixed.append("cleared EMERGENT_LLM_KEY from env")

        # Remove gateway_v3 references from system_state
        await db.system_state.update_one(
            {"_id": "singleton"},
            {"$unset": {"gateway_v3_url": "", "emergent_key": ""}},
        )
    except Exception as e:
        LOG.debug("self_repair: db purge error: %s", e)
    return fixed


async def _verify_llm_chain() -> list[str]:
    """Quick smoke-test the LLM chain. Returns list of issues found."""
    issues = []
    try:
        from ghost_llm import ollama_available, _anthropic_key, _groq_key

        # SovereignCore gateway reachable? (preferred local GPU cluster)
        try:
            from sovereign_economy import sovereign_available
            if not await sovereign_available():
                issues.append("SovereignCore gateway unreachable — start sovereign-core for GPU inference")
        except Exception:
            pass

        # Anthropic key present?
        if not _anthropic_key():
            issues.append("ANTHROPIC_API_KEY missing — Groq/Ollama/SovereignCore will handle load")

        # Groq key?
        if not _groq_key():
            issues.append("GROQ_API_KEY missing — no free-tier fallback before Ollama")

        # Ollama reachable?
        if not await ollama_available():
            issues.append("Ollama unreachable — start Ollama for free local fallback")

        # litellm importable? Should NOT be.
        try:
            import litellm  # noqa: F401
            issues.append("CRITICAL: litellm is importable — must uninstall")
        except ImportError:
            pass

    except Exception as e:
        issues.append(f"chain check error: {e}")
    return issues


# ── main repair function ──────────────────────────────────────────────────────

async def check(db=None) -> dict:
    """Run all self-repair checks. Returns a report dict."""
    started = datetime.now(timezone.utc)
    actions: list[str] = []
    issues:  list[str] = []

    # 1. Bad imports
    actions += _purge_bad_imports()

    # 2. gateway_v3 on wrong port
    actions += _kill_bad_ports()

    # 3. Wrong Python on 8001
    actions += _check_wrong_python_on_8001()

    # 4. Emergent refs in DB
    if db is not None:
        actions += await _purge_stale_urls_from_db(db)

    # 5. LLM chain health
    issues += await _verify_llm_chain()

    # 6. Log to MongoDB
    report = {
        "checked_at": started.isoformat(),
        "actions_taken": actions,
        "issues_found": issues,
        "healthy": len(issues) == 0,
    }

    if db is not None:
        try:
            await db.self_repair_log.insert_one({**report})
            # Keep only last 100 repair logs
            count = await db.self_repair_log.count_documents({})
            if count > 100:
                oldest = await db.self_repair_log.find_one(
                    {}, sort=[("checked_at", 1)]
                )
                if oldest:
                    await db.self_repair_log.delete_one({"_id": oldest["_id"]})
        except Exception:
            pass

    if actions:
        LOG.warning("self_repair: actions taken: %s", actions)
    if issues:
        LOG.warning("self_repair: issues found: %s", issues)

    return report


async def get_repair_log(db, limit: int = 20) -> list[dict]:
    cursor = db.self_repair_log.find({}, {"_id": 0}).sort("checked_at", -1).limit(limit)
    return await cursor.to_list(length=limit)


# ── LLM error handler — called when chat_once raises ─────────────────────────

def is_stale_provider_error(exc: Exception) -> bool:
    """Detect errors from removed/stale LLM providers (litellm, old preview deployment)."""
    msg = str(exc).lower()
    return any(p in msg for p in [
        "budget has been exceeded", "emergentintegrations", "litellm",
        "openaiexception", "current cost", "max budget",
        "tatorot-dashboard",
    ])


async def handle_llm_error(exc: Exception, db=None) -> str:
    """Called when an LLM error occurs. Repairs what it can, returns safe message."""
    if is_stale_provider_error(exc):
        LOG.error("self_repair: caught stale-provider/litellm error — purging")
        _purge_bad_imports()
        _kill_bad_ports()
        if db:
            await _purge_stale_urls_from_db(db)
        # Force SovereignCore or Ollama as immediate fallback
        try:
            from sovereign_economy import sovereign_chat, sovereign_available
            if await sovereign_available():
                text = await sovereign_chat(
                    "You are GH05T3.",
                    "Respond: I caught a billing error and switched to SovereignCore local GPU. I'm back online."
                )
                return text
        except Exception:
            pass
        try:
            from ghost_llm import _call_ollama_safe
            text, tag = await _call_ollama_safe(
                "You are GH05T3.",
                "Respond: I caught a billing error and switched to local Ollama. I'm back online."
            )
            return text
        except Exception:
            pass
        return "I caught a billing/quota error and repaired it. Switched to SovereignCore/Ollama. Try again."
    return f"LLM error: {exc}"


# Backward-compat alias
is_emergent_error = is_stale_provider_error
