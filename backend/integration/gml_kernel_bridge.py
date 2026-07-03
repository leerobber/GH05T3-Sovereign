"""GH05T3 -> gml_kernel (Rust) FFI bridge.

Loads the compiled Rust glyph kernel and calls into it via ctypes.
Kept separate from kernel_adapter.py, which is explicitly Rust-free
(wraps the pure-Python sovereign-core Runtime).
"""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time

# backend/ghost_llm.py uses bare sibling imports (e.g. "from ollama_gateway
# import call"), which only resolve when backend/ itself is on sys.path.
# That holds when these scripts are launched directly from backend/ (as
# run.bat does), but not when this bridge imports backend.ghost_llm as a
# submodule from the repo root. Replicate the same sys.path entry here so
# ghost_llm's internal imports work regardless of caller cwd.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_REQUIRED_MODEL_CALL_FIELDS = ("backend", "prompt", "version")
_SUPPORTED_MODEL_CALL_VERSIONS = {"v1", "v2"}

_LIB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "gml_kernel",
    "target",
    "release",
    "libgml_kernel.so",
)

_gml = ctypes.CDLL(_LIB_PATH)

_gml.gh05t3_run_core_loop.restype = ctypes.c_void_p
_gml.gh05t3_run_core_loop_json.restype = ctypes.c_void_p

_gml.gh05t3_model_call.restype = ctypes.c_void_p
_gml.gh05t3_model_call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]

_gml.gh05t3_free_string.argtypes = [ctypes.c_void_p]
_gml.gh05t3_free_string.restype = None


def _take_string(raw: int) -> str:
    try:
        return ctypes.cast(raw, ctypes.c_char_p).value.decode("utf-8")
    finally:
        _gml.gh05t3_free_string(raw)


def run_gh05t3_kernel_core() -> str:
    """Runs the whole Rust core loop. MODEL_CALL glyphs resolve to the Rust
    echo-stub (kernel/model.rs / ffi::model_call_summary) — this does NOT
    reach ghost_llm. Rust cannot call back into Python through this ctypes
    binding; that would need an explicit callback registration (a Python
    CFUNCTYPE passed into a Rust static), which is not built yet.
    """
    raw = _gml.gh05t3_run_core_loop()
    return _take_string(raw)


def run_gh05t3_kernel_core_json() -> dict:
    """Same run as run_gh05t3_kernel_core(), but via the JSON-returning FFI
    export (kernel::payload::KernelRunSummary) instead of Rust's Debug-format
    string. Returns {"tick": int, "short_term": [str, ...]}."""
    raw = _gml.gh05t3_run_core_loop_json()
    return json.loads(_take_string(raw))


def call_rust_model_stub(backend: str, prompt: str, version: str = "v2") -> str:
    """Direct call into Rust's gh05t3_model_call. Returns the v2 JSON envelope
    (kernel::payload::ModelCallPayload) as a string, e.g.:
      {"backend":"claude","prompt":"...","version":"v2","meta":{}}
    Isolated from the full core loop — useful for testing the FFI contract.
    Feed the result into handle_model_call_json() to actually run it.
    """
    raw = _gml.gh05t3_model_call(
        backend.encode("utf-8"), prompt.encode("utf-8"), version.encode("utf-8")
    )
    return _take_string(raw)


def check_fs(base_path: str | None = None) -> dict:
    """Filesystem sentinel: verifies we can read/write/delete in a target
    directory. Defaults to the GH05T3 repo root (two levels up from here)."""
    if base_path is None:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    result: dict = {"ok": False, "path": base_path, "details": ""}

    try:
        if not os.path.isdir(base_path):
            result["details"] = "base_path is not a directory"
            return result

        fd, tmp_path = tempfile.mkstemp(prefix="gh05t3_fs_sentinel_", dir=base_path)
        os.close(fd)

        payload = b"gh05t3-fs-sentinel"
        with open(tmp_path, "wb") as f:
            f.write(payload)

        with open(tmp_path, "rb") as f:
            data = f.read()

        os.remove(tmp_path)

        if data == payload:
            result["ok"] = True
            result["details"] = "rw ok"
        else:
            result["details"] = "payload mismatch"

    except Exception as e:
        result["details"] = f"exception: {e!r}"

    return result


def check_net(test_url: str = "https://example.com", timeout: float = 2.0) -> dict:
    """Network sentinel: verifies outbound HTTP and basic reachability.
    Uses httpx if available; reports explicitly if httpx is missing."""
    result: dict = {"ok": False, "url": test_url, "details": "", "latency_ms": None}

    try:
        import httpx
    except Exception as e:
        result["details"] = f"httpx missing or unusable: {e!r}"
        return result

    try:
        start = time.perf_counter()
        resp = httpx.get(test_url, timeout=timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        result["latency_ms"] = int(elapsed_ms)

        if 200 <= resp.status_code < 400:
            result["ok"] = True
            result["details"] = f"reachable, status={resp.status_code}"
        else:
            result["details"] = f"unhealthy status={resp.status_code}"

    except Exception as e:
        result["details"] = f"exception during request: {e!r}"

    return result


def check_gpu() -> dict:
    """GPU sentinel: prefers torch.cuda if torch is importable, otherwise
    falls back to `nvidia-smi`. Never hangs — subprocess calls are timed out."""
    result: dict = {"ok": False, "details": "", "device_count": None, "device_name": None}

    try:
        import torch

        available = torch.cuda.is_available()
        result["device_count"] = torch.cuda.device_count() if available else 0
        if available:
            result["ok"] = True
            result["device_name"] = torch.cuda.get_device_name(0)
            result["details"] = "torch.cuda available"
        else:
            result["details"] = "torch installed, no CUDA device visible"
        return result
    except Exception as e:
        result["details"] = f"torch unavailable ({e!r}), falling back to nvidia-smi"

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            names = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            result["ok"] = True
            result["device_count"] = len(names)
            result["device_name"] = names[0] if names else None
            result["details"] = "nvidia-smi reachable"
        else:
            result["details"] = f"nvidia-smi failed: {proc.stderr.strip() or proc.returncode}"
    except FileNotFoundError:
        result["details"] = "nvidia-smi not found on PATH"
    except Exception as e:
        result["details"] = f"nvidia-smi exception: {e!r}"

    return result


def check_wsl() -> dict:
    """WSL sentinel: detects whether we're running inside WSL and confirms
    basic subprocess execution works."""
    result: dict = {"ok": False, "is_wsl": False, "details": ""}

    try:
        with open("/proc/version", "r") as f:
            version_str = f.read()
        result["is_wsl"] = "microsoft" in version_str.lower()
    except Exception as e:
        result["details"] = f"could not read /proc/version: {e!r}"
        return result

    try:
        proc = subprocess.run(["uname", "-r"], capture_output=True, text=True, timeout=5)
        subprocess_ok = proc.returncode == 0
    except Exception as e:
        result["details"] = f"subprocess exec failed: {e!r}"
        return result

    result["ok"] = subprocess_ok
    result["details"] = (
        f"running in WSL ({proc.stdout.strip()})" if result["is_wsl"]
        else f"not WSL, subprocess exec ok ({proc.stdout.strip()})"
    )
    return result


def check_dependencies() -> dict:
    """Aggregated dependency health report for GH05T3.

    Returns fs/net/ghost_llm health, plus gml_kernel_so (whether the Rust
    shared lib this module loads is present) for continuity with the
    earlier importability-only version of this check.
    """
    status: dict = {}

    status["fs"] = check_fs()
    status["net"] = check_net()
    status["gpu"] = check_gpu()
    status["wsl"] = check_wsl()

    try:
        import backend.ghost_llm  # noqa: F401
        status["ghost_llm"] = {"ok": True, "details": "import ok"}
    except Exception as e:
        status["ghost_llm"] = {"ok": False, "details": f"import failed: {e!r}"}

    status["gml_kernel_so"] = os.path.isfile(_LIB_PATH)

    return status


def print_dependency_report() -> None:
    report = check_dependencies()
    print("=== GH05T3 Dependency Report ===")
    print(json.dumps(report, indent=2, sort_keys=True))


def run_model_via_ghost_llm(backend: str, prompt: str, version: str = "v1") -> str:
    """Routes a MODEL_CALL glyph's prompt through GH05T3's real LLM cascade
    (backend.ghost_llm.chat_once) instead of the Rust echo-stub.

    If `backend` names an entry in backend.ghost_llm.BACKENDS
    ("local_llama"/"local_mistral"/"local_phi"), that specific local Ollama
    model is hard-pinned instead of running the auto-cascade — real backend
    selection, unlike the "claude"/"gpt" cascade hints used elsewhere. Any
    other backend name falls through to chat_once's default cascade, which
    has no manual selector of its own.

    Checks check_dependencies() first: if ghost_llm isn't importable or
    outbound net is down, skips straight to LOCAL_FALLBACK rather than
    attempting (and waiting out) a call that can't succeed. If the
    pre-check passes but the cascade itself still throws, returns a
    MODEL_ERROR instead of silently falling back, since that's a real,
    unexpected failure rather than a known-bad environment.

    Uses asyncio.run, so call this from sync code only. If the caller is
    already inside an event loop (e.g. a FastAPI handler), await
    chat_once(...)/BACKENDS[backend](...) directly instead.
    """
    deps = check_dependencies()
    ghost_ok = deps.get("ghost_llm", {}).get("ok", False)
    net_ok = deps.get("net", {}).get("ok", False)

    if not ghost_ok or not net_ok:
        return f"[LOCAL_FALLBACK] backend={backend},version={version},prompt={prompt}"

    from backend.ghost_llm import BACKENDS, chat_once  # deferred: pulls in httpx et al.

    if backend in BACKENDS:
        try:
            text, _provider_used = asyncio.run(
                BACKENDS[backend](session="gml_kernel", system="", user=prompt)
            )
            return text
        except Exception as e:
            return f"[MODEL_ERROR] backend={backend} failure: {e!r}; prompt={prompt}"

    try:
        text, _provider_used = asyncio.run(
            chat_once(session="gml_kernel", system="", user=prompt)
        )
        return text
    except Exception as e:
        return f"[MODEL_ERROR] ghost_llm failure: {e!r}; prompt={prompt}"


def stream_model_via_ghost_llm(
    prompt: str, backend: str = "local_llama", version: str = "v3",
) -> tuple[str, list[str]]:
    """v3 streaming MODEL_CALL. Returns (full_text, chunks).

    If `backend` names an entry in backend.ghost_llm.STREAM_BACKENDS
    ("local_llama"/"local_mistral"/"local_phi"), that specific local Ollama
    model is hard-pinned. Any other backend name falls through to
    chat_stream's default (Ollama-only, no cloud-tier cascade — see that
    function's docstring).

    Same dependency pre-check as run_model_via_ghost_llm: skips straight to
    LOCAL_FALLBACK if ghost_llm/net aren't healthy. If the pre-check passes
    but streaming itself fails partway through, returns whatever chunks
    arrived plus a trailing MODEL_ERROR chunk rather than losing partial
    output.
    """
    deps = check_dependencies()
    ghost_ok = deps.get("ghost_llm", {}).get("ok", False)
    net_ok = deps.get("net", {}).get("ok", False)

    if not ghost_ok or not net_ok:
        fallback = f"[LOCAL_FALLBACK] backend={backend},version={version},prompt={prompt}"
        return fallback, [fallback]

    from backend.ghost_llm import STREAM_BACKENDS, chat_stream  # deferred: pulls in httpx et al.

    stream_fn = STREAM_BACKENDS.get(backend, chat_stream)
    chunks: list[str] = []

    async def _run() -> None:
        async for text, _provider_used in stream_fn(session="gml_kernel", system="", user=prompt):
            chunks.append(text)

    try:
        asyncio.run(_run())
    except Exception as e:
        chunks.append(f"[MODEL_ERROR] backend={backend} streaming failure: {e!r}; prompt={prompt}")

    return "".join(chunks), chunks


def handle_model_call_json(payload_json: str) -> str:
    """v2 MODEL_CALL contract: JSON in, JSON out.

    Input shape (matches Rust's kernel::payload::ModelCallPayload):
      {"backend": str, "prompt": str, "version": str, "meta": {...}}

    Output shape:
      {"backend": str|None, "version": str|None, "provider_used": str|None,
       "text": str|None, "error": str|None}

    Never raises. Missing/invalid fields and cascade failures both come back
    as a structured envelope rather than an exception.
    """
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return json.dumps({
            "error": f"invalid JSON: {e}",
            "backend": None, "version": None, "provider_used": None, "text": None,
        })

    missing = [f for f in _REQUIRED_MODEL_CALL_FIELDS if f not in payload]
    if missing:
        return json.dumps({
            "error": f"missing field(s): {', '.join(missing)}",
            "backend": payload.get("backend"),
            "version": payload.get("version"),
            "provider_used": None,
            "text": None,
        })

    backend = payload["backend"]
    prompt = payload["prompt"]
    version = payload["version"]

    if version not in _SUPPORTED_MODEL_CALL_VERSIONS:
        return json.dumps({
            "error": f"unsupported version: {version}",
            "backend": backend, "version": version,
            "provider_used": None, "text": None,
        })

    try:
        from backend.ghost_llm import chat_once  # deferred: pulls in httpx et al.

        text, provider_used = asyncio.run(
            chat_once(session="gml_kernel", system="", user=prompt)
        )
        return json.dumps({
            "backend": backend, "version": version,
            "provider_used": provider_used, "text": text, "error": None,
        })
    except Exception as e:
        return json.dumps({
            "backend": backend, "version": version,
            "provider_used": None,
            "text": f"[LOCAL_FALLBACK] {prompt}",
            "error": str(e),
        })


# ---------------------------------------------------------------------------
# v4: multi-model blending
#
# Real hard-pinning is only possible for entries in backend.ghost_llm.
# BACKENDS (local_llama/local_mistral/local_phi -> specific Ollama model
# tags) and the gh05t3 fine-tuned model (via LLM_PROVIDER=gh05t3). Any other
# backend name (e.g. "claude", "gpt") has no manual selector and falls
# through to chat_once's normal auto-cascade — every cloud tier (groq,
# google, openrouter, anthropic) runs in a fixed priority order regardless
# of what's requested, so requesting backends=["claude","gpt"] will run the
# same cascade for both and most likely get the SAME provider back for each
# (whichever wins first). This is a real ceiling of the underlying system,
# not a bug in this code. Each result's "provider_used" field reports what
# actually answered, not what was asked for — use that field, not the
# "backend" label, to see what really ran.
# ---------------------------------------------------------------------------

_REQUIRED_MULTI_MODEL_CALL_FIELDS = ("backends", "prompt", "version", "blend_strategy")
_GH05T3_FINE_TUNED_ALIASES = {"gh05t3", "local", "local_fallback"}


def _call_backend_once(backend: str, prompt: str) -> tuple[str, str]:
    """One best-effort call for a single requested backend name.

    Resolution order:
      1. backend.ghost_llm.BACKENDS[backend] — real hard-pinned local
         Ollama model (local_llama/local_mistral/local_phi). Same registry
         run_model_via_ghost_llm/stream_model_via_ghost_llm use, so
         "local_llama" means the same thing across v2/v3/v4.
      2. backend in {"gh05t3","local","local_fallback"} — forces
         LLM_PROVIDER=gh05t3, pinning the fine-tuned model server instead
         of a raw Ollama tag. Mutates the process-wide env var for the
         call's duration — not safe to call concurrently from multiple
         threads/tasks.
      3. anything else — chat_once's normal auto-cascade (no manual
         selector beyond cases 1-2 — see module docstring above).
    """
    from backend.ghost_llm import BACKENDS, chat_once

    if backend in BACKENDS:
        return asyncio.run(BACKENDS[backend](session="gml_kernel", system="", user=prompt))

    prior = os.environ.get("LLM_PROVIDER")
    try:
        if backend.lower() in _GH05T3_FINE_TUNED_ALIASES:
            os.environ["LLM_PROVIDER"] = "gh05t3"
        return asyncio.run(chat_once(session="gml_kernel", system="", user=prompt))
    finally:
        if prior is None:
            os.environ.pop("LLM_PROVIDER", None)
        else:
            os.environ["LLM_PROVIDER"] = prior


def _blend_concat(results: list[dict]) -> str:
    return "\n---\n".join(f"[{r['backend']}:{r['provider_used']}] {r['text']}" for r in results)


def _blend_vote(results: list[dict]) -> str:
    from collections import Counter

    texts = [r["text"] for r in results if r["text"]]
    if not texts:
        return ""
    return Counter(texts).most_common(1)[0][0]


def _blend_rank(results: list[dict]) -> str:
    candidates = [r for r in results if r["text"] and "[MODEL_ERROR]" not in r["text"]]
    pool = candidates or results
    return max(pool, key=lambda r: len(r["text"] or ""))["text"]


def _blend_chain(results: list[dict]) -> str:
    combined = ""
    for r in results:
        combined = f"{combined}\n{r['text']}".strip() if combined else (r["text"] or "")
    return combined


_BLEND_STRATEGIES = {
    "concat": _blend_concat,
    "vote": _blend_vote,
    "rank": _blend_rank,
    "chain": _blend_chain,
}


def run_multi_model_via_ghost_llm(
    backends: list[str], prompt: str, blend_strategy: str = "concat", version: str = "v4"
) -> dict:
    """v4 multi-model MODEL_CALL: calls each requested backend (best-effort,
    see module-level note above) and combines results per blend_strategy
    (concat/vote/rank/chain). Never raises — per-backend failures fall back
    to LOCAL_FALLBACK/MODEL_ERROR text rather than propagating."""
    deps = check_dependencies()
    ghost_ok = deps.get("ghost_llm", {}).get("ok", False)
    net_ok = deps.get("net", {}).get("ok", False)

    results = []
    for backend in backends:
        if not ghost_ok or not net_ok:
            results.append({
                "backend": backend, "provider_used": None,
                "text": f"[LOCAL_FALLBACK] backend={backend},prompt={prompt}",
            })
            continue
        try:
            text, provider_used = _call_backend_once(backend, prompt)
            results.append({"backend": backend, "provider_used": provider_used, "text": text})
        except Exception as e:
            results.append({
                "backend": backend, "provider_used": None,
                "text": f"[MODEL_ERROR] {backend} failure: {e!r}",
            })

    strategy_fn = _BLEND_STRATEGIES.get(blend_strategy, _blend_concat)
    blended = strategy_fn(results)

    return {
        "backends": backends,
        "version": version,
        "blend_strategy": blend_strategy,
        "results": results,
        "blended": blended,
    }


def handle_multi_model_call_json(payload_json: str) -> str:
    """v4 multi-model MODEL_CALL contract: JSON in, JSON out.

    Input shape (matches Rust's kernel::payload::MultiModelCallPayload):
      {"backends": [str, ...], "prompt": str, "version": str,
       "blend_strategy": str, "meta": {...}}

    Output: run_multi_model_via_ghost_llm()'s dict, JSON-encoded, plus
    "error": None on success. Never raises.
    """
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid JSON: {e}", "blended": None, "results": None})

    missing = [f for f in _REQUIRED_MULTI_MODEL_CALL_FIELDS if f not in payload]
    if missing:
        return json.dumps({
            "error": f"missing field(s): {', '.join(missing)}",
            "blended": None, "results": None,
        })

    backends = payload["backends"]
    if not isinstance(backends, list) or not backends:
        return json.dumps({"error": "backends must be a non-empty list", "blended": None, "results": None})

    result = run_multi_model_via_ghost_llm(
        backends=backends,
        prompt=payload["prompt"],
        blend_strategy=payload["blend_strategy"],
        version=payload["version"],
    )
    result["error"] = None
    return json.dumps(result)
