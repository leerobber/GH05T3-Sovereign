"""GH05T3 Coder sub-agent — GitHub + PyTest full loop.

Capabilities
------------
1. List whitelisted repos the user has pre-approved (env `CODER_REPO_WHITELIST`).
2. Clone a repo into a per-task sandbox (`/tmp/gh05t3-coder/<task_id>/`).
3. Run PyTest on the current tree and capture pass/fail.
4. Ask the nightly LLM to patch the code to fix failures (strict diff contract).
5. Apply the patch, re-run tests, iterate up to `max_iterations`.
6. Commit the successful diff on a new branch, push, open a Pull Request.

Safety
------
- Repos outside `CODER_REPO_WHITELIST` are rejected.
- Writes ONLY on a new branch (`gh05t3/<task_id>`), never to main/master.
- Auto-merge is **disabled** — the user reviews & merges the PR.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from github import Github, GithubException

LOG = logging.getLogger("ghost.coder")

SANDBOX_ROOT = Path("/tmp/gh05t3-coder")
SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def whitelist() -> list[str]:
    raw = os.environ.get("CODER_REPO_WHITELIST", "")
    return [r.strip() for r in raw.split(",") if r.strip()]


def _is_allowed(full_name: str) -> bool:
    wl = whitelist()
    return full_name in wl


def _pat() -> str | None:
    return os.environ.get("GITHUB_PAT")


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
async def _run(cmd: list[str], cwd: str | None = None,
               env: dict | None = None, timeout: int = 120) -> tuple[int, str, str]:
    """Run a shell command, return (code, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=cwd, env={**os.environ, **(env or {})},
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return 124, "", f"timeout after {timeout}s"
    return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------
def _gh() -> Github:
    pat = _pat()
    if not pat:
        raise RuntimeError("GITHUB_PAT not configured")
    return Github(pat, per_page=100)


async def list_repos() -> list[dict]:
    """Return whitelisted repos with basic metadata (no clones)."""
    wl = whitelist()
    if not wl or not _pat():
        return []
    gh = _gh()
    out = []
    for full in wl:
        try:
            r = gh.get_repo(full)
            out.append({
                "full_name": r.full_name,
                "description": r.description or "",
                "default_branch": r.default_branch,
                "language": r.language or "",
                "stars": r.stargazers_count,
                "pushed_at": r.pushed_at.isoformat() if r.pushed_at else None,
                "html_url": r.html_url,
            })
        except GithubException as e:
            out.append({"full_name": full, "error": f"github {e.status}",
                        "detail": str(e.data)[:200] if hasattr(e, "data") else ""})
        except Exception as e:  # noqa: BLE001
            out.append({"full_name": full, "error": str(e)[:200]})
    return out


# ---------------------------------------------------------------------------
# PyTest runner
# ---------------------------------------------------------------------------
async def run_pytest(path: str, subdir: str = "", test_target: str = "") -> dict:
    """Run pytest in `path/subdir`, optionally scoped to a single file/dir
    via `test_target` (passed as the pytest positional). Captures output +
    pass/fail summary."""
    target = str(Path(path) / subdir) if subdir else path
    if not Path(target).is_dir():
        return {"ok": False, "error": f"target dir not found: {target}"}
    # Use the same interpreter that runs the backend (guarantees pytest is
    # installed; the system `python` on PATH may not have it).
    # Also add the repo root to PYTHONPATH so tests can import top-level
    # packages without requiring `pip install -e .` first.
    existing_pp = os.environ.get("PYTHONPATH", "")
    repo_root = path
    new_pp = repo_root + (os.pathsep + existing_pp if existing_pp else "")
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short",
           "--maxfail=25", "-p", "no:cacheprovider",
           f"--rootdir={repo_root}"]
    if test_target:
        cmd.append(test_target)
    code, out, err = await _run(
        cmd, cwd=target, timeout=180,
        env={"PYTHONPATH": new_pp},
    )
    tail = (out + "\n" + err).strip().splitlines()[-60:]
    passed = failed = 0
    m = re.search(r"(\d+)\s+passed", out)
    if m:
        passed = int(m.group(1))
    m = re.search(r"(\d+)\s+failed", out)
    if m:
        failed = int(m.group(1))
    return {
        "ok": code == 0, "exit_code": code,
        "passed": passed, "failed": failed,
        "tail": "\n".join(tail),
    }


# ---------------------------------------------------------------------------
# LLM patch loop
# ---------------------------------------------------------------------------
PATCH_SYS = """You are the GH05T3 Coder sub-agent.
You will receive a failing PyTest report plus the current contents of the
relevant files. Your job: emit one or more COMPLETE file rewrites that fix
the failures.

Output format — strict:
=== FILE: <relative/path/from/repo/root> ===
<entire new contents of the file — complete, ready to save>
=== END FILE ===

Repeat that block for every file you change. Only the files you actually
need to modify. No prose, no markdown fences, no diff syntax.

HARD RULES (violation = rejected patch):
1. Preserve EVERY function, class, import, constant, and docstring that
   exists in the original file. Do not delete code that isn't directly
   causing the failure.
2. Your rewrite MUST retain at least 75% of the original file's lines.
   Any rewrite shorter than 75% of the original will be auto-rejected.
3. Make the SMALLEST change that fixes the reported error. If you can't
   fix it surgically, emit NOTHING (no FILE block at all).
4. Copy unrelated code verbatim — same line breaks, same comments, same
   whitespace. Only the broken lines should differ.
5. When in doubt, include MORE of the original file, not less."""


# Parse "=== FILE: ... === ... === END FILE ===" blocks.
_FILE_BLOCK_RE = re.compile(
    r"===\s*FILE:\s*(?P<path>[^\n=]+?)\s*===\s*\n(?P<body>.*?)\n===\s*END\s*FILE\s*===",
    re.DOTALL,
)


def _extract_file_rewrites(text: str) -> list[tuple[str, str]]:
    """Return list of (rel_path, new_content) tuples."""
    out = []
    for m in _FILE_BLOCK_RE.finditer(text):
        p = m.group("path").strip().strip("`'\" ")
        body = m.group("body")
        if p and not p.startswith("/") and ".." not in p.split("/"):
            out.append((p, body))
    return out


def _extract_diff(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        t = re.sub(r"\n```\s*$", "", t)
    if "diff --git" in t:
        return t[t.index("diff --git"):].strip()
    return t.strip()


async def _apply_changes(repo_dir: str, llm_raw: str) -> tuple[bool, str, int]:
    """Apply either FILE rewrites or a unified diff. Returns
    (applied, message, files_changed).

    Safety: rejects a FILE block that deletes more than 60% of the original
    file's lines — this usually means the LLM dropped real functionality
    instead of making a surgical fix.
    """
    if not llm_raw:
        return False, "empty LLM response", 0
    rewrites = _extract_file_rewrites(llm_raw)
    if rewrites:
        changed = 0
        rejected = []
        for rel, body in rewrites:
            target = Path(repo_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not body.endswith("\n"):
                body += "\n"
            # Aggressive-rewrite guard: for files >=40 lines, reject if the
            # rewrite keeps less than 75% of the original — usually means the
            # LLM dropped real functionality instead of making a surgical fix.
            if target.is_file():
                try:
                    original = target.read_text(encoding="utf-8", errors="replace")
                    orig_lines = max(1, original.count("\n"))
                    new_lines = body.count("\n")
                    if orig_lines >= 40 and new_lines < 0.75 * orig_lines:
                        rejected.append(
                            f"{rel} ({new_lines} lines vs original {orig_lines})"
                        )
                        continue
                except Exception:
                    pass
            target.write_text(body, encoding="utf-8")
            changed += 1
        if rejected and changed == 0:
            return (False,
                    f"rewrites rejected as too aggressive (>60% deletion): {'; '.join(rejected)}",
                    0)
        msg = f"wrote {changed} file(s): {', '.join(r for r,_ in rewrites if r not in ' '.join(rejected))[:300]}"
        if rejected:
            msg += f" · rejected: {'; '.join(rejected)[:200]}"
        return changed > 0, msg, changed
    # Fallback: unified diff path (for LLMs that stubbornly emit diffs).
    diff = _extract_diff(llm_raw)
    if "diff --git" in diff:
        patch_path = Path(repo_dir) / ".gh05t3_patch.diff"
        patch_path.write_text(diff, encoding="utf-8")
        code, out, err = await _run(
            ["git", "apply", "--recount", "--ignore-whitespace",
             "--ignore-space-change", "--whitespace=nowarn",
             ".gh05t3_patch.diff"],
            cwd=repo_dir,
        )
        try:
            patch_path.unlink()
        except Exception:
            pass
        return code == 0, (out + err).strip()[:500], 0
    return False, "no FILE blocks or diff found in LLM output", 0


async def _read_context_files(repo_dir: str, hint_text: str, max_chars: int = 9000) -> str:
    """Extract file paths mentioned in the pytest output and include a snapshot."""
    # Look for .py files in tracebacks / import errors.
    candidates = set()
    for m in re.finditer(r"([A-Za-z0-9_./\-]+\.py)[:\"' ]", hint_text):
        p = m.group(1)
        if p.startswith("/"):
            # Convert absolute sandbox paths to repo-relative if possible.
            try:
                p = str(Path(p).relative_to(repo_dir))
            except ValueError:
                continue
        if ".." in p.split("/") or p.startswith("/"):
            continue
        candidates.add(p)
    # Also grab any `from X import` hints (just the path of what's mentioned).
    for m in re.finditer(r"from\s+([A-Za-z0-9_.]+)\s+import", hint_text):
        mod = m.group(1).replace(".", "/") + ".py"
        candidates.add(mod)
        candidates.add(m.group(1).replace(".", "/") + "/__init__.py")
    out = []
    budget = max_chars
    seen = 0
    for rel in sorted(candidates):
        if seen >= 6 or budget <= 0:
            break
        p = Path(repo_dir) / rel
        if not p.is_file():
            continue
        try:
            body = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        chunk = body[:budget]
        out.append(f"=== FILE: {rel} ===\n{chunk}\n=== END FILE ===")
        budget -= len(chunk)
        seen += 1
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------
async def run_task(
    full_name: str,
    task_description: str,
    nightly_chat,                 # free/cheap router (fallback)
    chat_once=None,               # premium Claude router (preferred for Coder)
    max_iterations: int = 3,
    subdir: str = "",
    test_target: str = "",
    open_pr: bool = True,
) -> dict:
    """Execute a full coder task against a whitelisted repo.

    Returns a JSON-serialisable trace of every step for the UI.
    """
    if not _is_allowed(full_name):
        return {"ok": False, "error": f"repo not whitelisted: {full_name}",
                "whitelist": whitelist()}
    if not _pat():
        return {"ok": False, "error": "GITHUB_PAT not configured"}

    task_id = uuid.uuid4().hex[:10]
    work = SANDBOX_ROOT / task_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)

    pat = _pat()
    clone_url = f"https://x-access-token:{pat}@github.com/{full_name}.git"
    trace: list[dict] = []
    trace.append({"step": "clone", "repo": full_name, "at": _now()})

    code, out, err = await _run(
        ["git", "clone", "--depth", "1", clone_url, str(work)], timeout=120,
    )
    if code != 0:
        return {"ok": False, "task_id": task_id, "trace": trace,
                "error": f"clone failed: {(out + err)[:500]}"}

    # configure git identity for commits
    await _run(["git", "config", "user.email", "gh05t3@robertlee.local"], cwd=str(work))
    await _run(["git", "config", "user.name", "GH05T3 Coder"], cwd=str(work))

    # discover default branch
    code, out, _ = await _run(["git", "symbolic-ref", "--short", "HEAD"], cwd=str(work))
    default_branch = out.strip() or "main"
    branch = f"gh05t3/{task_id}"
    await _run(["git", "checkout", "-b", branch], cwd=str(work))

    # initial test run
    first = await run_pytest(str(work), subdir=subdir, test_target=test_target)
    trace.append({"step": "pytest:initial", **first})

    attempts: list[dict] = []
    final_ok = first.get("ok", False)

    # Track the best-performing state across iterations so a regression
    # on a later iteration doesn't throw away real gains. We snapshot
    # the working tree into a git stash-like tag so we can restore it.
    def _score(r: dict) -> tuple:
        # higher is better: (ok flag, passed count, -failed count)
        return (
            1 if r.get("ok") else 0,
            int(r.get("passed") or 0),
            -int(r.get("failed") or 0),
        )

    best = {"result": first, "commit": None}
    # Initial commit of clone so we always have a restore point.
    await _run(["git", "add", "-A"], cwd=str(work))
    await _run(["git", "commit", "-m", "gh05t3: snapshot before coder loop",
                "--allow-empty"], cwd=str(work))
    c0, out0, _ = await _run(["git", "rev-parse", "HEAD"], cwd=str(work))
    best["commit"] = out0.strip() if c0 == 0 else None

    for i in range(max_iterations):
        if final_ok:
            break
        # Ask LLM to emit file rewrites.
        snap = await _read_context_files(str(work), first.get("tail", ""))
        user_msg = (
            f"Task: {task_description}\n\n"
            f"PyTest output (tail):\n{first.get('tail','')[-3500:]}\n\n"
            f"Current file contents:\n{snap[:9000] if snap else '(none found)'}\n\n"
            "Emit one or more `=== FILE: <path> ===` blocks as specified."
        )
        try:
            # Prefer the premium model (Claude Sonnet) for the Coder — it
            # reproduces long files verbatim far better than the nightly
            # free router. Fall back to nightly_chat if Claude is
            # budget-exhausted or unavailable.
            if chat_once is not None:
                try:
                    raw, tag = await chat_once(
                        f"coder-{task_id}-{i}", PATCH_SYS, user_msg, role="proposer",
                    )
                except Exception as ce:  # noqa: BLE001
                    LOG.warning("coder: premium chat_once failed, falling back: %s", ce)
                    raw, tag = await nightly_chat(f"coder-{task_id}-{i}", PATCH_SYS, user_msg)
            else:
                raw, tag = await nightly_chat(f"coder-{task_id}-{i}", PATCH_SYS, user_msg)
        except Exception as e:  # noqa: BLE001
            attempts.append({"iter": i, "llm_error": str(e)[:200]})
            break
        applied, apply_msg, n_files = await _apply_changes(str(work), raw)
        if not applied:
            attempts.append({"iter": i, "engine": tag, "applied": False,
                             "reason": apply_msg[:300]})
            break
        # Re-run tests
        result = await run_pytest(str(work), subdir=subdir, test_target=test_target)
        # Checkpoint: if this iteration is the best we've seen, commit it.
        if _score(result) > _score(best["result"]):
            await _run(["git", "add", "-A"], cwd=str(work))
            await _run(
                ["git", "commit", "-m",
                 f"gh05t3: iter {i} · passed={result.get('passed')} failed={result.get('failed')}"],
                cwd=str(work),
            )
            c, out_c, _ = await _run(["git", "rev-parse", "HEAD"], cwd=str(work))
            best = {"result": result, "commit": out_c.strip() if c == 0 else None}
        attempts.append({
            "iter": i, "engine": tag, "applied": True, "files_changed": n_files,
            "passed": result.get("passed"), "failed": result.get("failed"),
            "ok": result.get("ok"),
            "change_summary": apply_msg[:200],
            "best_so_far": _score(result) >= _score(best["result"]),
        })
        if result.get("ok"):
            final_ok = True
            first = result
            break
        first = result

    # If the last iteration regressed vs. our best checkpoint, restore the
    # best-known state before considering a PR.
    if best["commit"] and _score(first) < _score(best["result"]):
        await _run(["git", "reset", "--hard", best["commit"]], cwd=str(work))
        first = best["result"]
        trace.append({"step": "rollback_to_best",
                      "commit": best["commit"][:10],
                      "passed": best["result"].get("passed"),
                      "failed": best["result"].get("failed")})

    trace.append({"step": "iterations", "attempts": attempts})

    pr_url = None
    pushed = False
    # Compare initial vs best/final. Push & open PR whenever:
    #   - all tests pass (fully green), OR
    #   - the best result strictly improved over the initial run (more
    #     passed, or same passed with fewer failed, or a flipped ok flag).
    initial = trace[1] if len(trace) > 1 else {}
    initial_score = _score(initial)
    final_score = _score(first)
    improved = final_score > initial_score

    if (final_ok or improved) and attempts:
        # Ensure everything is committed. The best-checkpoint flow above
        # may have already committed; this is a no-op if tree is clean.
        await _run(["git", "add", "-A"], cwd=str(work))
        commit_msg = f"gh05t3 coder: {task_description[:80]}"
        await _run(["git", "commit", "-m", commit_msg, "--allow-empty"],
                   cwd=str(work))
        code, out, err = await _run(
            ["git", "push", "-u", "origin", branch], cwd=str(work), timeout=180,
        )
        pushed = code == 0
        trace.append({"step": "push", "ok": pushed, "stderr": err[:300]})
        if pushed and open_pr:
            try:
                gh = _gh()
                r = gh.get_repo(full_name)
                status_line = (
                    "✅ all green" if final_ok
                    else f"🟡 partial: {initial.get('passed',0)}p/{initial.get('failed',0)}f "
                         f"→ {first.get('passed',0)}p/{first.get('failed',0)}f"
                )
                pr = r.create_pull(
                    title=f"[GH05T3] {task_description[:100]}",
                    body=(
                        "Automated PR opened by the GH05T3 Coder sub-agent.\n\n"
                        f"**Task:** {task_description}\n\n"
                        f"**Iterations:** {len(attempts)}\n"
                        f"**PyTest progress:** {status_line}\n\n"
                        "Review the diff and merge if it looks good. "
                        "Auto-merge is disabled by design."
                    ),
                    head=branch, base=default_branch,
                )
                pr_url = pr.html_url
                trace.append({"step": "pr", "url": pr_url})
            except Exception as e:  # noqa: BLE001
                trace.append({"step": "pr", "error": str(e)[:300]})

    return {
        "ok": final_ok,
        "task_id": task_id,
        "repo": full_name,
        "branch": branch,
        "iterations": len(attempts),
        "initial_fail": not trace[1].get("ok", False) if len(trace) > 1 else None,
        "final": first,
        "pushed": pushed,
        "pr_url": pr_url,
        "attempts": attempts,
        "trace": trace,
    }
