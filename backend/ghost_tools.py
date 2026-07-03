"""GH05T3 filesystem + shell tools."""
from __future__ import annotations
import asyncio, logging, os
from pathlib import Path

LOG = logging.getLogger("ghost.tools")

ALLOWED_ROOTS = [Path(r"C:\Users\leer4\GH05T3"), Path(r"C:\sovereign")]

_DANGEROUS = ["rm -rf", "format c", "mkfs", "drop table", "drop database", "shutdown /", ":(){ :|:& }"]

DEFAULT_CWD = str(Path(r"C:\Users\leer4\GH05T3\backend"))


def _safe_path(path: str) -> Path:
    p = Path(path).resolve()
    for root in ALLOWED_ROOTS:
        try:
            p.relative_to(root.resolve()); return p
        except ValueError: continue
    raise PermissionError(f"Path outside allowed roots: {path}")


def _check_shell(cmd: str) -> None:
    low = cmd.lower()
    for bad in _DANGEROUS:
        if bad in low:
            raise PermissionError(f"Blocked: {bad!r}")


async def _read_file(path: str) -> str:
    p = _safe_path(path)
    if not p.exists(): return f"ERROR: not found: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:40000] + "\n...[truncated]" if len(text) > 40000 else text
    except Exception as e: return f"ERROR: {e}"


async def _write_file(path: str, content: str) -> str:
    p = _safe_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except Exception as e: return f"ERROR: {e}"


async def _list_dir(path: str) -> str:
    p = _safe_path(path)
    if not p.exists(): return f"ERROR: not found: {path}"
    try:
        return "\n".join(f"[{'DIR' if i.is_dir() else 'FILE'}] {i.name}" for i in sorted(p.iterdir())) or "(empty)"
    except Exception as e: return f"ERROR: {e}"


async def _run_shell(command: str, cwd: str | None = None) -> str:
    _check_shell(command)
    work_dir = DEFAULT_CWD
    if cwd:
        try: work_dir = str(_safe_path(cwd))
        except PermissionError as e: return f"ERROR: {e}"
    try:
        proc = await asyncio.create_subprocess_shell(command, cwd=work_dir,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env={**os.environ})
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        except asyncio.TimeoutError:
            proc.kill(); return "ERROR: timed out"
        result = out.decode("utf-8", "replace")
        return f"[exit {proc.returncode or 0}]\n" + (result[:8000] + "\n...[truncated]" if len(result) > 8000 else result)
    except Exception as e: return f"ERROR: {e}"


async def execute_tool(name: str, inputs: dict) -> str:
    LOG.info("[tool] %s", name)
    if name == "read_file": return await _read_file(inputs["path"])
    if name == "write_file": return await _write_file(inputs["path"], inputs["content"])
    if name == "list_dir": return await _list_dir(inputs["path"])
    if name == "run_shell": return await _run_shell(inputs["command"], inputs.get("cwd"))
    return f"ERROR: unknown tool {name!r}"


ANTHROPIC_TOOLS = [
    {"name": "read_file", "description": "Read full contents of a file. Use before editing.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write or overwrite a file with new content.",
     "input_schema": {"type": "object", "properties": {
         "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "list_dir", "description": "List files and directories at a path.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "run_shell", "description": "Run a shell command (pip, npm, git, python, etc.).",
     "input_schema": {"type": "object", "properties": {
         "command": {"type": "string"}, "cwd": {"type": "string"}}, "required": ["command"]}},
]


def _to_openai_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
             "parameters": t["input_schema"]}} for t in tools]


OPENAI_TOOLS = _to_openai_tools(ANTHROPIC_TOOLS)
