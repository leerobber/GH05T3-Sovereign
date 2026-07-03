# memory_store.py â€” long-horizon memory for the judicial mesh.
#
# Two tiers:
#   - episodic  : one record per mesh run (ledger.jsonl), with an embedding
#   - semantic  : distilled "learnings" (learnings.json), always injected
#
# Embedder chain (best-effort, never fatal): NPU :8111 -> Ollama -> lexical.
# Privacy: secrets/keys are redacted on write; full source is never stored â€”
# only the issue text, touched paths, outcome, and a short comment.
import json
import math
import re
import time
import urllib.request
from zlib import zlib
from pathlib import Path

from _common import BRIDGE_DIR

MEM_DIR = BRIDGE_DIR / "memory"
LEDGER = MEM_DIR / "ledger.jsonl"
LEARNINGS = MEM_DIR / "learnings.json"

NPU_URL = "http://localhost:8111/embed_query"
OLLAMA_EMBED = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

MAX_LEDGER = 3000          # cap before compression
RETRIEVE_THRESHOLD = 0.30  # min cosine to surface a prior case

# Never persist anything that looks like a credential.
_SECRET_RE = re.compile(
    r"(AIza[\w-]{15,}|sk-[\w-]{20,}|AQ\.[\w.-]{20,}|gsk_[\w-]{20,}"
    r"|xox[baprs]-[\w-]{10,}|ghp_[\w-]{20,}|[A-Za-z0-9+/]{50,}={0,2})"
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub("<redacted>", text or "")


def _http_json(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _lexical(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        vec[zlib.crc32(tok.encode()) % dim] += 1.0   # crc32 = stable across runs
    return vec


def embed(text: str) -> tuple[list[float], str]:
    """Return (vector, tag). Tag identifies the embedder so we only compare like-with-like."""
    text = (text or "")[:4000]
    try:
        d = _http_json(NPU_URL, {"text": text}, timeout=5)
        v = d.get("embedding") or d.get("vector") or (d.get("embeddings") or [None])[0]
        if v:
            return [float(x) for x in v], "npu"
    except Exception:
        pass
    try:
        d = _http_json(OLLAMA_EMBED, {"model": EMBED_MODEL, "prompt": text}, timeout=20)
        v = d.get("embedding")
        if v:
            return [float(x) for x in v], "ollama:" + EMBED_MODEL
    except Exception:
        pass
    return _lexical(text), "lexical"


def _cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _query_text(issue: str, focus: str, paths: list[str]) -> str:
    return _redact(f"{issue} | focus: {focus or ''} | files: {', '.join(paths or [])}")


def record_run(repo, issue, focus, paths, test_outcome, verdict, models, n_rounds, comment=""):
    """Append one episodic record. Best-effort; failures never break a mesh run."""
    try:
        MEM_DIR.mkdir(parents=True, exist_ok=True)
        vec, tag = embed(_query_text(issue, focus, paths))
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "repo": repo, "issue": _redact(issue)[:300], "focus": _redact(focus or "")[:200],
            "paths": paths or [], "test_outcome": test_outcome, "verdict": verdict,
            "models": models, "n_rounds": n_rounds, "comment": _redact(comment)[:500],
            "emb": tag, "vec": vec,
        }
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        _maybe_compress()
        return True
    except Exception:
        return False


def _load() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def retrieve(repo, issue, focus, paths, k=3) -> list[dict]:
    """Top-k prior cases most similar to the current situation (same embedder only)."""
    try:
        qvec, qtag = embed(_query_text(issue, focus, paths))
    except Exception:
        return []
    scored = []
    for rec in _load():
        if rec.get("emb") != qtag:
            continue
        s = _cosine(qvec, rec.get("vec", []))
        if rec.get("repo") == repo:
            s += 0.05  # mild same-repo bias = personalization
        scored.append((s, rec))
    scored.sort(key=lambda x: -x[0])
    return [r for s, r in scored[:k] if s >= RETRIEVE_THRESHOLD]


def format_context(repo, issue, focus, paths, k=3) -> str:
    """The PRIOR EXPERIENCE + LEARNINGS block injected into the proposer's context."""
    parts = []
    learn = get_learnings()
    if learn:
        parts.append("Distilled learnings:\n" + "\n".join(f"- {x}" for x in learn[:8]))
    cases = retrieve(repo, issue, focus, paths, k=k)
    if cases:
        lines = []
        for c in cases:
            files = ", ".join(c.get("paths", [])[:3]) or "â€”"
            lines.append(
                f"- [{c.get('verdict')}] \"{c.get('issue','')[:80]}\" "
                f"(files: {files}; tests: {c.get('test_outcome')}; rounds: {c.get('n_rounds')})"
                + (f" â€” note: {c['comment'][:120]}" if c.get("comment") else "")
            )
        parts.append("Similar past runs (learn from these â€” don't repeat rejected approaches):\n"
                     + "\n".join(lines))
    return "\n\n".join(parts)


# â”€â”€â”€ semantic tier â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_learnings() -> list[str]:
    if LEARNINGS.exists():
        try:
            return json.loads(LEARNINGS.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def add_learning(text: str):
    text = _redact(text).strip()
    if not text:
        return
    items = get_learnings()
    if text not in items:
        items.append(text)
        MEM_DIR.mkdir(parents=True, exist_ok=True)
        LEARNINGS.write_text(json.dumps(items, indent=2), encoding="utf-8")


# â”€â”€â”€ forgetting / compression â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _maybe_compress():
    """Keep the ledger bounded: retain all SIGNED (high value) + most-recent others."""
    recs = _load()
    if len(recs) <= MAX_LEDGER:
        return
    signed = [r for r in recs if r.get("verdict") == "SIGNED"]
    others = [r for r in recs if r.get("verdict") != "SIGNED"]
    keep = signed + others[-(MAX_LEDGER - len(signed)):]
    LEDGER.write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")


def stats() -> dict:
    recs = _load()
    by = {}
    for r in recs:
        by[r.get("verdict", "?")] = by.get(r.get("verdict", "?"), 0) + 1
    return {"total": len(recs), "by_verdict": by, "learnings": len(get_learnings())}
