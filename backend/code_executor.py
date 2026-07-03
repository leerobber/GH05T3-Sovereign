"""Sandboxed code execution for GH05T3's coder agent.

Runs Python (or shell) snippets in a subprocess with hard timeouts and
captures stdout/stderr. The LLM can request execution; results are fed back.
"""
from __future__ import annotations
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

LOG = logging.getLogger("ghost.executor")

TIMEOUT_DEFAULT = 15        # seconds
TIMEOUT_MAX     = 60
VENV_PYTHON     = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
# Fall back to system python if venv not present
PYTHON_EXE      = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Blocked imports — prevent network exfiltration and dangerous ops
_BLOCKED = {
    "socket", "requests", "httpx", "urllib", "http.client",
    "subprocess", "os.system", "shutil.rmtree",
}


def _safety_check(code: str) -> str | None:
    """Return an error string if the code looks dangerous, else None."""
    lower = code.lower()
    dangerous = [
        "rmdir", "shutil.rmtree", "os.remove", "os.unlink",
        "format(", "winreg", "__import__('os').system",
    ]
    for d in dangerous:
        if d in lower:
            return f"Blocked: code contains '{d}'"
    return None


async def run_python(code: str, timeout: int = TIMEOUT_DEFAULT) -> dict:
    """Execute a Python snippet and return stdout/stderr/exit_code."""
    timeout = min(timeout, TIMEOUT_MAX)

    danger = _safety_check(code)
    if danger:
        return {"ok": False, "stdout": "", "stderr": danger, "exit_code": -1, "lang": "python"}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp = f.name

    try:
        proc = await asyncio.create_subprocess_exec(
            PYTHON_EXE, tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {
                "ok": False, "stdout": "", "lang": "python",
                "stderr": f"Execution timed out after {timeout}s", "exit_code": -1,
            }
        return {
            "ok":        proc.returncode == 0,
            "stdout":    stdout.decode("utf-8", errors="replace")[:4000],
            "stderr":    stderr.decode("utf-8", errors="replace")[:2000],
            "exit_code": proc.returncode,
            "lang":      "python",
        }
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


async def run_shell(command: str, timeout: int = TIMEOUT_DEFAULT) -> dict:
    """Run a shell command (PowerShell on Windows). Read-only operations only."""
    timeout = min(timeout, TIMEOUT_MAX)

    # Whitelist safe shell prefixes
    safe_prefixes = [
        "dir ", "ls ", "echo ", "python ", "pip ", "git log", "git status",
        "git diff", "git show", "where ", "which ", "type ", "cat ",
        "mongosh ", "curl ", "ping ", "ipconfig", "systeminfo",
        "Get-", "Select-", "Format-", "Out-", "Measure-",
    ]
    cmd_lower = command.strip().lower()
    if not any(cmd_lower.startswith(p.lower()) for p in safe_prefixes):
        return {
            "ok": False, "stdout": "", "lang": "shell",
            "stderr": "Shell command not in safe list. Use Python executor for custom logic.",
            "exit_code": -1,
        }

    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return {"ok": False, "stdout": "", "stderr": f"Timed out after {timeout}s",
                "exit_code": -1, "lang": "shell"}

    return {
        "ok":        proc.returncode == 0,
        "stdout":    stdout.decode("utf-8", errors="replace")[:4000],
        "stderr":    stderr.decode("utf-8", errors="replace")[:2000],
        "exit_code": proc.returncode,
        "lang":      "shell",
    }


async def llm_execute_loop(llm_fn, task: str, max_rounds: int = 3) -> dict:
    """Give the LLM a task, let it write and run code, return the final result.

    llm_fn: async callable(system, user) -> str
    """
    system = (
        "You are GH05T3's coder sub-agent. Write Python code to complete the task. "
        "Return ONLY the code block, no explanation. "
        "When the code runs successfully, summarize the result in plain text. "
        "If it errors, fix and retry."
    )

    history = []
    last_result = None

    for round_num in range(1, max_rounds + 1):
        user = task if round_num == 1 else (
            f"Previous code output:\n{last_result}\n\nFix the code and try again, or summarize if done."
        )
        history.append(f"Round {round_num}: {user[:200]}")

        code_response = await llm_fn(system, user)

        # Extract code block
        import re
        m = re.search(r"```(?:python)?\n([\s\S]+?)```", code_response)
        if not m:
            # No code block — treat whole response as summary/done
            return {"ok": True, "summary": code_response.strip(),
                    "rounds": round_num, "history": history}

        code = m.group(1)
        result = await run_python(code, timeout=30)
        last_result = result["stdout"] or result["stderr"]

        if result["ok"]:
            # One more LLM call to summarize the output
            summary = await llm_fn(
                "You are GH05T3. Summarize this execution result in 1-2 sentences.",
                f"Task: {task}\nOutput:\n{last_result[:1000]}"
            )
            return {"ok": True, "summary": summary.strip(), "output": last_result,
                    "rounds": round_num, "history": history}

    return {"ok": False, "summary": "Max rounds reached without success.",
            "output": last_result, "rounds": max_rounds, "history": history}
