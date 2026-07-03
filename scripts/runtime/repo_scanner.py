"""
repo_scanner.py — Scans leerobber repos and extracts capability map for GH05T3 business training.
Writes: data/repo_capabilities.json
Run standalone: python repo_scanner.py
Or import: from repo_scanner import scan_all_repos
"""
import json, os, re
from pathlib import Path

# All repos to scan (relative to user home or absolute)
HOME = Path.home()
REPOS = {
    "sovereign-core": HOME / "sovereign-core",
    "hyper-agent":    HOME / "hyper-agent",
    "openclaw":       HOME / "openclaw",
    "verelene_v5":    HOME / "verelene_v5",
    "MYTHOS":         HOME / "MYTHOS",
    "Jarvis":         HOME / "Jarvis",
    "my_agent":       HOME / "my_agent",
    "GH05T3":         HOME / "GH05T3",
    "avery":          HOME / "avery",
}

README_FILES = ["README.md", "readme.md", "README.txt", "VISION.md", "AGENTS.md"]
CODE_EXTS    = {".py", ".js", ".ts", ".mjs"}
SKIP_DIRS    = {"__pycache__", "node_modules", ".git", ".venv", "venv",
                "dist", "build", ".next", "coverage"}


def _read_safe(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]
    except Exception:
        return ""


def _sniff_readme(repo_path: Path) -> str:
    for name in README_FILES:
        p = repo_path / name
        if p.exists():
            return _read_safe(p, 1200)
    return ""


def _sniff_deps(repo_path: Path) -> list[str]:
    deps = []
    for fname in ["requirements.txt", "package.json", "pyproject.toml"]:
        p = repo_path / fname
        if not p.exists():
            continue
        text = _read_safe(p, 3000)
        if fname == "package.json":
            try:
                pkg = json.loads(text)
                deps += list((pkg.get("dependencies") or {}).keys())[:20]
                deps += list((pkg.get("devDependencies") or {}).keys())[:10]
            except Exception:
                pass
        else:
            # extract package names from lines like "fastapi>=0.110" or "  fastapi"
            for line in text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    name = re.split(r"[>=<!\[;]", line)[0].strip()
                    if name:
                        deps.append(name)
    return list(dict.fromkeys(deps))[:30]  # deduplicate, cap at 30


def _sniff_key_files(repo_path: Path, max_files: int = 8) -> dict[str, str]:
    """Find the most important source files and grab first 60 lines each."""
    priority_names = {
        "main.py", "agent.py", "server.py", "gateway.py", "app.py",
        "orchestrator.py", "continuous_runner.py", "curriculum.py",
        "main.js", "main.ts", "index.ts",
    }
    found = {}
    all_code = []

    for p in repo_path.rglob("*"):
        if any(skip in p.parts for skip in SKIP_DIRS):
            continue
        if p.suffix in CODE_EXTS and p.is_file():
            priority = 0 if p.name in priority_names else 1
            all_code.append((priority, p))

    all_code.sort(key=lambda x: (x[0], x[1].stat().st_size if x[1].exists() else 0),
                  reverse=False)

    for _, p in all_code[:max_files]:
        rel = str(p.relative_to(repo_path))
        lines = _read_safe(p, 2000).splitlines()[:60]
        snippet = "\n".join(lines).strip()
        if snippet:
            found[rel] = snippet

    return found


def _summarize_repo(name: str, repo_path: Path) -> dict:
    if not repo_path.exists():
        return {"name": name, "exists": False}

    readme = _sniff_readme(repo_path)
    deps   = _sniff_deps(repo_path)
    files  = _sniff_key_files(repo_path)

    # Pull first paragraph of README as the short description
    desc = ""
    for line in readme.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("!"):
            desc = line[:200]
            break

    # Detect tech stack from deps + file extensions
    stack = []
    dep_set = set(d.lower() for d in deps)
    if any(d in dep_set for d in ["fastapi", "uvicorn", "starlette"]):
        stack.append("FastAPI")
    if "torch" in dep_set or "transformers" in dep_set:
        stack.append("PyTorch/HuggingFace")
    if any(d in dep_set for d in ["react", "next", "vite"]):
        stack.append("React")
    if "mongodb" in dep_set or "motor" in dep_set:
        stack.append("MongoDB")
    if "sqlite" in " ".join(dep_set):
        stack.append("SQLite")
    if "ollama" in " ".join(dep_set):
        stack.append("Ollama")
    if "unsloth" in dep_set:
        stack.append("Unsloth/LoRA")

    return {
        "name":        name,
        "exists":      True,
        "path":        str(repo_path),
        "description": desc,
        "readme_excerpt": readme[:600],
        "tech_stack":  stack,
        "key_deps":    deps[:15],
        "key_files":   {k: v[:400] for k, v in list(files.items())[:5]},
    }


def scan_all_repos(output_path: str | Path = "data/repo_capabilities.json") -> dict:
    Path("data").mkdir(exist_ok=True)
    cap_map = {}
    for name, path in REPOS.items():
        print(f"  scanning {name}...", end=" ", flush=True)
        info = _summarize_repo(name, path)
        cap_map[name] = info
        status = "ok" if info["exists"] else "missing"
        print(status)

    # Build a compact summary string for injection into prompts
    lines = []
    for name, info in cap_map.items():
        if not info.get("exists"):
            continue
        stack = ", ".join(info.get("tech_stack", [])) or "Python"
        desc  = info.get("description", "")[:120]
        lines.append(f"• {name}: {desc} [{stack}]")
    cap_map["_summary"] = "\n".join(lines)

    out = Path(output_path)
    out.write_text(json.dumps(cap_map, indent=2), encoding="utf-8")
    print(f"\n  capability map -> {out}  ({len(lines)} repos active)")
    return cap_map


def load_capability_summary(path: str | Path = "data/repo_capabilities.json") -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("_summary", "")
    except Exception:
        return ""


if __name__ == "__main__":
    print("\n=== Repo Scanner ===")
    result = scan_all_repos()
    print("\n--- Capability Summary ---")
    print(result["_summary"])
    print()
