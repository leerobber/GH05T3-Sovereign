# _common.py — shared helpers for the Claude <-> Gemini bridge
# Loads keys from GH05T3/.env (same convention as the rest of the stack).
import os
import sys
from pathlib import Path

# Windows consoles default to cp1252; LLM output + status glyphs are UTF-8.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BRIDGE_DIR = Path(__file__).resolve().parent
GH05T3_ROOT = BRIDGE_DIR.parent
TRANSCRIPT_DIR = BRIDGE_DIR / "transcripts"

_ENV_CACHE = None


def _parse_env(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_env() -> dict:
    """Merge GH05T3/.env then backend/.env, with real os.environ taking priority."""
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    merged = {}
    merged.update(_parse_env(GH05T3_ROOT / ".env"))
    merged.update(_parse_env(GH05T3_ROOT / "backend" / ".env"))
    # os.environ wins so callers can override per-invocation
    for k in list(merged.keys()):
        if os.environ.get(k):
            merged[k] = os.environ[k]
    _ENV_CACHE = merged
    return merged


def get_key(*names: str) -> str | None:
    """Return the first non-empty value among the given env var names."""
    env = load_env()
    for n in names:
        v = os.environ.get(n) or env.get(n)
        if v:
            return v
    return None


def gemini_key() -> str:
    k = get_key("GEMINI_API_KEY", "GOOGLE_AI_KEY", "GOOGLE_API_KEY")
    if not k:
        sys.exit("[bridge] No Gemini key found (GEMINI_API_KEY / GOOGLE_AI_KEY in GH05T3/.env).")
    return k


def anthropic_key() -> str:
    k = get_key("ANTHROPIC_API_KEY")
    if not k:
        sys.exit("[bridge] No ANTHROPIC_API_KEY found in GH05T3/.env.")
    return k
