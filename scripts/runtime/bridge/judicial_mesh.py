#!/usr/bin/env python
# judicial_mesh.py â€” Claude <-> Gemini judicial code-improvement mesh (local, no Docker).
#
# Gemini scans a repo and proposes a concrete fix. Claude reviews it with deeper
# analysis. They negotiate a single candidate set of file changes. A patch is
# only "signed" when BOTH sign-off conditions hold:
#   (1) consensus  â€” both agents approve the *same* candidate, and
#   (2) evidence   â€” the patched code passes the repo's tests in a sandbox.
# Neither alone is enough. A failing test is fed back so the agents must revise.
#
#   python judicial_mesh.py --path C:\Users\leer4\GH05T3\sovereignnation
#   python judicial_mesh.py --path . --test-cmd "python -m pytest -q" --turns 8
#   python judicial_mesh.py --path ..\aethyro_launch --focus "fix the broken voiceover script"
#
# Output: <workspace>/agent_patch.md  (default workspace = bridge/workspace)
import argparse
import datetime as dt
from difflib import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from _common import BRIDGE_DIR  # also forces UTF-8 stdout on import
from providers import make_agent

try:
    import memory_store  # long-horizon memory (best-effort; mesh runs without it)
except Exception:
    memory_store = None

# Dirs/files never scanned or copied into the test sandbox.
IGNORE = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", "dist", "build", "models", "ollama_models", "mongo-data", "logs",
    ".idea", ".vscode", "site-packages", "kaggle_output", "kaggle_push", "kaggle_pull_tmp",
}
IGNORE_SUFFIX = {".gguf", ".bin", ".safetensors", ".pt", ".onnx", ".zip", ".tar",
                 ".png", ".jpg", ".jpeg", ".ico", ".mp4", ".wav", ".db", ".sqlite", ".log"}
SCAN_SUFFIX = {".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".html", ".css",
               ".sh", ".ps1", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".txt"}

PROTOCOL = (
    "Respond in EXACTLY this plain-text format. Do NOT use JSON. Do NOT wrap the whole reply in a code fence.\n\n"
    "ACTION: <propose|approve|no_issue>\n"
    "ISSUE: <short title of the problem, or 'none'>\n"
    "COMMENT: <your reasoning / critique / defense, 1-4 sentences>\n"
    "\n"
    "Then, ONLY if ACTION is 'propose', append one block per changed file, verbatim, NO escaping:\n"
    "--- FILE: <repo-relative path> ---\n"
    "<the COMPLETE new content of the file>\n"
    "--- END FILE ---\n"
    "\n"
    "Rules:\n"
    "- Fix exactly ONE focused, REAL issue. Keep the change minimal and buildable.\n"
    "- 'propose' = put forward or revise the candidate; include FULL file content for every file you touch.\n"
    "- 'approve' = you accept the CURRENT candidate exactly as-is; add no FILE blocks.\n"
    "- 'no_issue' = the code already satisfies the task and tests; nothing should change.\n"
    "- Do NOT gold-plate. If the code works and tests pass, use 'no_issue' or 'approve' â€” never invent "
    "stylistic or speculative changes."
)

def proposer_sys(name: str) -> str:
    return (
        f"You are {name}, the proposing judge in a two-AI judicial code mesh. You scan the repository, "
        "find the single highest-value concrete fix, and propose complete file changes. You then defend "
        "or revise based on the reviewer's critique. Be rigorous; converge toward a correct, minimal patch.\n\n"
        + PROTOCOL
    )


def reviewer_sys(name: str, peer: str) -> str:
    return (
        f"You are {name}, the reviewing judge in a two-AI judicial code mesh, paired with {peer}. You scrutinize "
        f"{peer}'s proposal for correctness, edge cases, regressions, and whether it actually compiles/passes "
        "tests. Challenge weak proposals and counter-propose a better complete change; approve only when truly "
        "sound. When test results are provided, treat them as ground truth and revise to make them pass.\n\n"
        + PROTOCOL
    )


# â”€â”€â”€ repo helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _iter_files(root: Path):
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in IGNORE]
        for f in files:
            p = Path(dp) / f
            if p.suffix.lower() in IGNORE_SUFFIX:
                continue
            yield p


def build_digest(root: Path, max_files: int, char_budget: int, focus: str | None) -> str:
    """A budgeted snapshot of the repo: tree + contents of the most relevant source files."""
    files = [p for p in _iter_files(root) if p.suffix.lower() in SCAN_SUFFIX]
    tree = "\n".join(sorted(str(p.relative_to(root)) for p in files)[:400])
    # Prefer code files, smaller-to-mid size first so we fit more signal in budget.
    code = sorted(
        [p for p in files if p.suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx"}],
        key=lambda p: p.stat().st_size,
    )
    if focus:
        kws = [w for w in re.findall(r"\w+", focus.lower()) if len(w) > 3]
        code.sort(key=lambda p: -sum(k in str(p).lower() for k in kws))
    out, used = [f"## FILE TREE ({len(files)} files)\n{tree}\n", "\n## FILE CONTENTS\n"], len(tree)
    shown = 0
    for p in code:
        if shown >= max_files or used >= char_budget:
            break
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if len(text) > 12000:
            text = text[:12000] + "\n# â€¦ (truncated) â€¦"
        block = f"\n### {p.relative_to(root)}\n```\n{text}\n```\n"
        out.append(block); used += len(block); shown += 1
    return "".join(out)


def detect_test_cmd(root: Path) -> str | None:
    if (root / "pytest.ini").exists() or (root / "tests").is_dir() or any(root.glob("test_*.py")):
        return "python -m pytest -q"
    if (root / "package.json").exists():
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
            if "test" in pkg.get("scripts", {}):
                return "npm test --silent"
        except Exception:
            pass
    return None


# â”€â”€â”€ protocol parsing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Plain-text fence format â€” robust for code (no JSON escaping of newlines/quotes).
_FILE_RE = re.compile(r"---\s*FILE:\s*(.+?)\s*---\r?\n(.*?)\r?\n---\s*END FILE\s*---", re.DOTALL)


def parse_turn(text: str) -> dict | None:
    am = re.search(r"ACTION:\s*(propose|approve|no_issue)", text, re.IGNORECASE)
    if not am:
        return None
    action = am.group(1).lower()
    im = re.search(r"ISSUE:\s*(.*)", text)
    issue = im.group(1).strip() if im else ""
    cm = re.search(r"COMMENT:\s*(.+?)(?=\n---\s*FILE:|\Z)", text, re.DOTALL)
    comment = cm.group(1).strip() if cm else ""
    files = [{"path": m.group(1).strip(), "new_content": m.group(2)} for m in _FILE_RE.finditer(text)]
    # A 'propose' with no parseable file blocks is not actionable â€” reject so the
    # turn is retried rather than silently advancing an empty candidate.
    if action == "propose" and not files:
        return None
    return {"action": action, "issue": issue, "comment": comment, "files": files}


def candidate_key(files: list[dict]) -> str:
    return json.dumps(sorted((f["path"], f.get("new_content", "")) for f in files))


# â”€â”€â”€ evidence gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _ignore_copy(_dir, names):
    return [n for n in names if n in IGNORE or Path(n).suffix.lower() in IGNORE_SUFFIX]


def run_evidence(root: Path, files: list[dict], test_cmd: str) -> tuple[bool, str]:
    """Apply candidate to a sandbox copy and run the tests. Returns (passed, output)."""
    sandbox = Path(tempfile.mkdtemp(prefix="judicial_"))
    try:
        dst = sandbox / root.name
        shutil.copytree(root, dst, ignore=_ignore_copy, symlinks=True)
        for f in files:
            tp = dst / f["path"]
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(f.get("new_content", ""), encoding="utf-8")
        proc = subprocess.run(
            test_cmd, cwd=dst, shell=True, capture_output=True, text=True, timeout=600,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, out[-6000:]
    except subprocess.TimeoutExpired:
        return False, "[evidence] tests timed out after 600s"
    except Exception as e:
        return False, f"[evidence] sandbox error: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


# â”€â”€â”€ the mesh â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def diff_for(root: Path, files: list[dict]) -> str:
    chunks = []
    for f in files:
        old = ""
        op = root / f["path"]
        if op.exists():
            old = op.read_text(encoding="utf-8", errors="ignore")
        d = difflib.unified_diff(
            old.splitlines(True), f.get("new_content", "").splitlines(True),
            fromfile=f"a/{f['path']}", tofile=f"b/{f['path']}",
        )
        chunks.append("".join(d) or f"(no textual diff for {f['path']})\n")
    return "\n".join(chunks)


def run(path: Path, turns: int, test_cmd: str | None, focus: str | None,
        max_files: int, workspace: Path, provider: str = "groq", model: str | None = None,
        reviewer: str = "claude", reviewer_model: str | None = None, use_memory: bool = True) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = workspace / "agent_patch.md"

    proposer = make_agent(provider, model=model)              # proposer seat
    reviewer = make_agent(reviewer, model=reviewer_model)     # reviewer seat
    p_name, r_name = proposer.name, reviewer.name
    if p_name == r_name:                                      # same provider both seats: disambiguate
        p_name, r_name = p_name + "-A", r_name + "-B"
    p_sys = proposer_sys(p_name)
    r_sys = reviewer_sys(r_name, p_name)
    digest = build_digest(path, max_files=max_files, char_budget=80000, focus=focus)
    if test_cmd is None:
        test_cmd = detect_test_cmd(path)

    goal = f"Repository: {path.name}\n"
    if focus:
        goal += f"FOCUS (fix this specifically): {focus}\n"
    else:
        goal += "Find and fix the single highest-value concrete issue.\n"

    print(f"[judicial] path={path}")
    print(f"[judicial] seats: proposer={p_name} vs reviewer={r_name}")
    print(f"[judicial] test gate: {test_cmd or 'NONE FOUND â€” patch will be UNVERIFIED'}")
    print(f"[judicial] digest: {len(digest)} chars, focus={focus!r}\n")

    transcript = [f"# Judicial mesh â€” {ts}\n\n{goal}\n**Test gate:** `{test_cmd or 'none'}`\n\n---\n"]
    candidate: list[dict] = []
    approvals: set[str] = set()
    last_evidence = ""
    signed = False
    rejected_reason = ""
    no_issue_votes: set[str] = set()
    verdict = ""
    last_issue = last_comment = ""

    # Long-horizon memory: pull prior experience for similar issues in this repo.
    mem_ctx = ""
    if use_memory and memory_store is not None:
        try:
            sample = [str(p.relative_to(path)) for p in list(_iter_files(path))[:8]]
            mem_ctx = memory_store.format_context(path.name, focus or "", focus, sample)
            if mem_ctx:
                print(f"[judicial] memory: injected prior experience ({memory_store.stats()['total']} runs on file)")
        except Exception:
            mem_ctx = ""

    # Proposer opens by scanning + proposing; then alternate with the reviewer.
    order = [p_name, r_name]
    for turn in range(1, turns + 1):
        who = order[(turn - 1) % 2]
        agent, agent_sys = (proposer, p_sys) if who == p_name else (reviewer, r_sys)

        ctx = [goal]
        if turn <= 2:
            ctx.append("\n## REPOSITORY SNAPSHOT\n" + digest)
        if mem_ctx and turn <= 2:
            ctx.append("\n## PRIOR EXPERIENCE (long-horizon memory)\n" + mem_ctx)
        if candidate:
            ctx.append("\n## CURRENT CANDIDATE PATCH\n" + diff_for(path, candidate))
            ctx.append(f"\nApproved so far by: {', '.join(sorted(approvals)) or 'nobody'}")
        if last_evidence:
            ctx.append("\n## LATEST TEST RESULT (ground truth â€” must pass)\n```\n" + last_evidence + "\n```")
        ctx.append(f"\n## TRANSCRIPT\n{''.join(transcript)[-6000:]}")
        ctx.append(f"\nYou are {who}. Give your next move per the protocol.")
        user = "\n".join(ctx)

        raw = agent.ask(agent_sys, user)
        move = parse_turn(raw)
        if move is None:
            transcript.append(f"\n## Turn {turn} â€” {who} (unparseable, skipped)\n{raw[:800]}\n")
            print(f"[turn {turn}] {who}: unparseable response, skipping")
            continue

        action, issue, comment = move["action"], move["issue"], move["comment"]
        last_issue, last_comment = issue or last_issue, comment or last_comment
        transcript.append(f"\n## Turn {turn} â€” {who}: {action.upper()}\n**Issue:** {issue}\n\n{comment}\n")
        print(f"[turn {turn}] {who}: {action} â€” {issue[:70]}")

        if action == "propose" and move["files"]:
            new_key = candidate_key(move["files"])
            if not candidate or new_key != candidate_key(candidate):
                candidate = move["files"]
                approvals = {who}            # a fresh proposal resets agreement
                last_evidence = ""           # stale evidence no longer applies
                no_issue_votes.clear()       # someone now sees a real issue
            else:
                approvals.add(who)
        elif action == "approve" and candidate:
            approvals.add(who)
        elif action == "no_issue" and not candidate:
            no_issue_votes.add(who)
            if no_issue_votes >= {p_name, r_name}:   # both judges: nothing to fix
                verdict = "no_issue"
                print("[judicial] both agents judged the code correct â€” no change needed.")
                break

        # Consensus reached? Then run the evidence gate.
        if candidate and approvals >= {p_name, r_name}:
            if not test_cmd:
                rejected_reason = "Consensus reached but NO test suite found â€” patch is UNVERIFIED."
                print(f"[judicial] {rejected_reason}")
                break
            print(f"[judicial] consensus on candidate â†’ running evidence gate: {test_cmd}")
            passed, last_evidence = run_evidence(path, candidate, test_cmd)
            transcript.append(f"\n## Evidence gate (turn {turn})\n`{test_cmd}` â†’ "
                              f"{'PASS' if passed else 'FAIL'}\n```\n{last_evidence}\n```\n")
            if passed:
                signed = True
                print("[judicial] âœ… tests pass â€” patch SIGNED.")
                break
            else:
                print("[judicial] âŒ tests fail â€” sending failure back for revision.")
                approvals.clear()            # must re-earn consensus on a fixed candidate

    _write_patch(out_md, ts, goal, test_cmd, path, candidate, transcript,
                 signed, rejected_reason, last_evidence, verdict)

    # Long-horizon memory: record this run's outcome for future retrieval.
    if use_memory and memory_store is not None:
        final = ("SIGNED" if signed else "NO_ISSUE" if verdict == "no_issue"
                 else "UNVERIFIED" if (candidate and not test_cmd) else "NOT_SIGNED")
        outcome = "pass" if signed else ("none" if not test_cmd else "fail" if last_evidence else "n/a")
        memory_store.record_run(
            path.name, last_issue, focus or "",
            [f["path"] for f in candidate] if candidate else [],
            outcome, final, f"{p_name} vs {r_name}", turn, comment=last_comment)

    tag = "SIGNED âœ…" if signed else ("NO ISSUE FOUND âœ…" if verdict == "no_issue" else "NOT SIGNED âŒ")
    print(f"\n[judicial] {tag} â€” written to {out_md}")
    return out_md


def _write_patch(out_md, ts, goal, test_cmd, path, candidate, transcript,
                 signed, rejected_reason, last_evidence, verdict=""):
    if signed:
        status = "âœ… SIGNED (consensus + tests green)"
    elif verdict == "no_issue":
        status = "âœ… NO ISSUE â€” both judges found the code already correct; no change made"
    else:
        status = "âŒ NOT SIGNED â€” " + (rejected_reason or "no consensus reached within turn limit")
    body = [f"# Agent Patch â€” {ts}\n", f"**Status:** {status}\n", f"\n{goal}\n",
            f"**Test gate:** `{test_cmd or 'none'}`\n"]
    if candidate:
        body.append("\n## Proposed changes (unified diff)\n```diff\n" + diff_for(path, candidate) + "\n```\n")
        if signed:
            body.append("\n## Apply\nThese files passed the test gate. To apply:\n")
            for f in candidate:
                body.append(f"- overwrite `{f['path']}`\n")
    if last_evidence and not signed:
        body.append("\n## Last test output\n```\n" + last_evidence + "\n```\n")
    body.append("\n---\n## Full transcript\n" + "".join(transcript))
    out_md.write_text("".join(body), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Claude<->Gemini judicial code mesh (local).")
    ap.add_argument("--path", default=str(BRIDGE_DIR.parent),
                    help="Repo/dir to improve (default: GH05T3 root).")
    _seats = ["groq", "mistral", "openrouter", "nvidia", "gemini", "ollama", "claude"]
    ap.add_argument("--provider", default="groq", choices=_seats,
                    help="Proposer seat (default groq).")
    ap.add_argument("--model", help="Override the proposer model (provider default otherwise).")
    ap.add_argument("--reviewer", default="claude", choices=_seats,
                    help="Reviewer seat (default claude).")
    ap.add_argument("--reviewer-model", help="Override the reviewer model.")
    ap.add_argument("--focus", help="Steer both agents at a specific issue (optional).")
    ap.add_argument("--turns", type=int, default=8, help="Max agent turns (default 8).")
    ap.add_argument("--test-cmd", help="Override test command (auto-detected otherwise).")
    ap.add_argument("--max-files", type=int, default=25, help="Max source files in the scan digest.")
    ap.add_argument("--no-memory", action="store_true",
                    help="Disable long-horizon memory (for clean A/B baselines).")
    ap.add_argument("--workspace", default=str(BRIDGE_DIR / "workspace"),
                    help="Where agent_patch.md is written.")
    args = ap.parse_args()

    root = Path(args.path).resolve()
    if not root.is_dir():
        sys.exit(f"[judicial] not a directory: {root}")
    try:
        run(root, args.turns, args.test_cmd, args.focus, args.max_files,
            Path(args.workspace), provider=args.provider, model=args.model,
            reviewer=args.reviewer, reviewer_model=args.reviewer_model,
            use_memory=not args.no_memory)
    except KeyboardInterrupt:
        sys.exit("\n[judicial] interrupted.")


if __name__ == "__main__":
    main()
