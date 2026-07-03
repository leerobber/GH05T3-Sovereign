"""GH05T3 — modular tool registry with sandboxed shell execution.

Each tool is a class implementing execute() / validate() / describe().
The registry auto-generates OPENAI_TOOLS and ANTHROPIC_TOOLS from registered tools.

Sandboxing (RunShellTool):
  - Allowlisted executables only (python, pip, git, npm, node, yarn, etc.)
  - Blocked dangerous command patterns
  - Sensitive env vars stripped (API keys, tokens, secrets)
  - Hard 30s timeout with process-tree kill
  - Output capped at 8 KB
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

LOG = logging.getLogger("ghost.tools")

ALLOWED_ROOTS = [Path(r"C:\Users\leer4\GH05T3"), Path(r"C:\sovereign")]
DEFAULT_CWD   = str(Path(r"C:\Users\leer4\GH05T3\backend"))

# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

_DANGEROUS_PATTERNS = [
    "rm -rf", "format c", "mkfs", "drop table", "drop database",
    "shutdown /", ":(){ :|:& }", "del /s /q", "rd /s /q",
    "net user", "net localgroup", "reg delete", "reg add",
    "wmic process", "powershell -enc", "iex (", "invoke-expression",
    "curl | sh", "wget | sh", "bash -c", "cmd /c",
]

# Only these executable prefixes are allowed at the start of a command
_ALLOWED_EXECUTABLES = {
    "python", "python3", "pip", "pip3", "git", "npm", "node", "yarn",
    "pnpm", "uvicorn", "pytest", "mypy", "ruff", "black", "isort",
    "cargo", "rustc", "go", "java", "mvn", "gradle",
    "echo", "type", "dir", "ls", "cat", "head", "tail", "grep",
    "find", "where", "which",
}

# Env vars that must never be passed to subprocesses
_STRIP_ENV_KEYS = {
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "GOOGLE_AI_KEY",
    "STRIPE_SECRET_KEY", "STRIPE_WEBHOOK_SECRET", "GH05T3_API_TOKEN",
    "GH05T3_SECRET", "GITHUB_PAT", "TAILSCALE_API_KEY", "HF_TOKEN",
    "RUNPOD_API_KEY", "MARKETPLACE_API_KEY", "KILLSWITCH_KEY_HASH",
}


def _safe_path(path: str) -> Path:
    p = Path(path).resolve()
    for root in ALLOWED_ROOTS:
        try:
            p.relative_to(root.resolve())
            return p
        except ValueError:
            continue
    raise PermissionError(f"Path outside allowed roots: {path}")


def _check_shell(cmd: str) -> None:
    low = cmd.lower().strip()
    # Check dangerous patterns
    for bad in _DANGEROUS_PATTERNS:
        if bad in low:
            raise PermissionError(f"Blocked dangerous pattern: {bad!r}")
    # Check executable allowlist — first token must be in allowed set
    first_token = low.split()[0] if low.split() else ""
    # Strip path prefix (e.g. C:\Python\python.exe → python)
    exe = Path(first_token).stem.lower()
    if exe not in _ALLOWED_EXECUTABLES:
        raise PermissionError(
            f"Executable {exe!r} not in allowlist. Allowed: {sorted(_ALLOWED_EXECUTABLES)}"
        )


def _safe_env() -> dict[str, str]:
    """Return os.environ with sensitive keys stripped."""
    return {k: v for k, v in os.environ.items()
            if k.upper() not in _STRIP_ENV_KEYS}


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the process and its entire tree (Windows + POSIX compatible)."""
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tool base class
# ---------------------------------------------------------------------------

class Tool(ABC):
    """Abstract base for all GH05T3 tools."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict: ...

    def validate(self, inputs: dict) -> None:
        """Raise ValueError if required parameters are missing or invalid."""
        required = self.parameters.get("required", [])
        for key in required:
            if key not in inputs:
                raise ValueError(f"Missing required parameter: {key!r}")

    @abstractmethod
    async def execute(self, inputs: dict) -> str: ...

    def describe(self) -> dict:
        return {
            "name":         self.name,
            "description":  self.description,
            "input_schema": self.parameters,
        }


# ---------------------------------------------------------------------------
# Concrete tools
# ---------------------------------------------------------------------------

class ReadFileTool(Tool):
    name        = "read_file"
    description = "Read full contents of a file. Call before editing."
    parameters  = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Absolute or repo-relative path"}},
        "required": ["path"],
    }

    async def execute(self, inputs: dict) -> str:
        try:
            p = _safe_path(inputs["path"])
        except PermissionError as e:
            return f"ERROR: {e}"
        if not p.exists():
            return f"ERROR: not found: {inputs['path']}"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            if len(text) > 40_000:
                return text[:40_000] + "\n...[truncated]"
            return text
        except Exception as e:
            return f"ERROR: {e}"


class WriteFileTool(Tool):
    name        = "write_file"
    description = "Write or overwrite a file with new content."
    parameters  = {
        "type": "object",
        "properties": {
            "path":    {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def execute(self, inputs: dict) -> str:
        try:
            p = _safe_path(inputs["path"])
        except PermissionError as e:
            return f"ERROR: {e}"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(inputs["content"], encoding="utf-8")
            return f"OK: wrote {len(inputs['content'])} chars to {inputs['path']}"
        except Exception as e:
            return f"ERROR: {e}"


class ListDirTool(Tool):
    name        = "list_dir"
    description = "List files and subdirectories at a given path."
    parameters  = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    async def execute(self, inputs: dict) -> str:
        try:
            p = _safe_path(inputs["path"])
        except PermissionError as e:
            return f"ERROR: {e}"
        if not p.exists():
            return f"ERROR: not found: {inputs['path']}"
        try:
            entries = sorted(p.iterdir())
            return "\n".join(
                f"[{'DIR' if e.is_dir() else 'FILE'}] {e.name}" for e in entries
            ) or "(empty)"
        except Exception as e:
            return f"ERROR: {e}"


class RunShellTool(Tool):
    """Sandboxed shell command execution.

    Security layers:
      1. Allowlist: first executable token must be in _ALLOWED_EXECUTABLES
      2. Blocklist: dangerous patterns rejected before exec
      3. Environment: sensitive API keys stripped from subprocess env
      4. Timeout: hard 30s cap; process tree killed on breach
      5. Output: capped at 8 KB
      6. CWD: must be inside ALLOWED_ROOTS
    """
    name        = "run_shell"
    description = ("Run an allowlisted shell command (python, pip, git, npm, …). "
                   "30s timeout. Output capped at 8 KB.")
    parameters  = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Command to execute"},
            "cwd":     {"type": "string", "description": "Working directory (optional)"},
        },
        "required": ["command"],
    }

    _TIMEOUT_S  = 30
    _OUTPUT_CAP = 8_000

    def validate(self, inputs: dict) -> None:
        super().validate(inputs)
        try:
            _check_shell(inputs["command"])
        except PermissionError as e:
            raise ValueError(str(e))

    async def execute(self, inputs: dict) -> str:
        cmd = inputs["command"]
        try:
            _check_shell(cmd)
        except PermissionError as e:
            return f"ERROR: {e}"

        work_dir = DEFAULT_CWD
        if inputs.get("cwd"):
            try:
                work_dir = str(_safe_path(inputs["cwd"]))
            except PermissionError as e:
                return f"ERROR: {e}"

        kwargs: dict[str, Any] = {
            "cwd":    work_dir,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "env":    _safe_env(),
        }
        # On POSIX, put subprocess in its own process group for clean kill
        if sys.platform != "win32":
            kwargs["preexec_fn"] = os.setsid  # type: ignore[assignment]

        try:
            proc = await asyncio.create_subprocess_shell(cmd, **kwargs)
            try:
                out, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=self._TIMEOUT_S
                )
            except asyncio.TimeoutError:
                await _kill_tree(proc)
                return f"ERROR: command timed out after {self._TIMEOUT_S}s"

            text = out.decode("utf-8", "replace")
            if len(text) > self._OUTPUT_CAP:
                text = text[: self._OUTPUT_CAP] + "\n...[truncated]"
            return f"[exit {proc.returncode}]\n{text}"
        except Exception as e:
            return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    _REGISTRY[tool.name] = tool


def _build_registry() -> None:
    for cls in (ReadFileTool, WriteFileTool, ListDirTool, RunShellTool):
        register(cls())


_build_registry()


async def execute_tool(name: str, inputs: dict) -> str:
    LOG.info("[tool] %s", name)
    tool = _REGISTRY.get(name)
    if not tool:
        return f"ERROR: unknown tool {name!r}. Available: {list(_REGISTRY)}"
    try:
        tool.validate(inputs)
    except ValueError as e:
        return f"ERROR: validation failed — {e}"
    return await tool.execute(inputs)


# ---------------------------------------------------------------------------
# Auto-generated schema lists for Anthropic and OpenAI tool-calling APIs
# ---------------------------------------------------------------------------

ANTHROPIC_TOOLS = [t.describe() for t in _REGISTRY.values()]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name":        t.name,
            "description": t.description,
            "parameters":  t.parameters,
        },
    }
    for t in _REGISTRY.values()
]
