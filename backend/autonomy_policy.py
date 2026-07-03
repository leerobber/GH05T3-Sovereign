"""Autonomy safety policy for GH05T3 self-directed jobs.

The policy is intentionally conservative: GH05T3 can improve her own code and
tooling, but major/risky work needs Robert's partial approval and abusive work
is blocked outright.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ABUSE_TERMS = {
    "abuse", "credential theft", "steal", "exfiltrate", "exfiltration",
    "malware", "ransomware", "keylogger", "phishing", "spam", "botnet",
    "bypass protections", "disable safety", "hide activity", "stealth",
    "persistence", "privilege escalation", "exploit others",
}

MAJOR_TERMS = {
    "major", "large refactor", "refactor", "schema", "migration",
    "delete", "wipe", "drop", "restart", "install", "dependency",
    "push", "pull request", "github", "env", ".env", "secret",
    "network exposure", "production", "business records", "spend",
}

EMERGENCY_TERMS = {
    "emergency", "runaway paid", "paid llm", "exposed secret", "quota",
    "down service", "security incident", "rogue process", "restore",
}


def is_gh05t3_owned_path(path: str | Path) -> bool:
    # Check path segments so Windows paths work on Linux and vice versa.
    path_str = str(path).replace("\\", "/")
    parts = [p for p in path_str.split("/") if p]
    if "GH05T3" in parts:
        return True
    # Filesystem resolution only for paths native to this OS (absolute or home-relative).
    try:
        p = Path(path).expanduser()
        if not p.is_absolute():
            return False
        resolved = p.resolve()
        return resolved == REPO_ROOT or str(resolved).startswith(str(REPO_ROOT) + "/")
    except Exception:
        return False


def _contains_any(text: str, terms: set[str]) -> list[str]:
    low = text.lower()
    return sorted(term for term in terms if term in low)


def classify_action(
    description: str,
    *,
    paths: list[str | Path] | None = None,
    emergency: bool = False,
) -> dict:
    """Return policy decision: auto, approval_required, emergency, or blocked."""
    paths = paths or []
    joined = " ".join([description, *(str(p) for p in paths)])
    abuse_hits = _contains_any(joined, ABUSE_TERMS)
    if abuse_hits:
        return {
            "allowed": False,
            "level": "blocked",
            "reason": "Abusive or stealth behavior is never allowed.",
            "triggers": abuse_hits,
        }

    outside = [str(p) for p in paths if not is_gh05t3_owned_path(p)]
    if outside:
        return {
            "allowed": False,
            "level": "approval_required",
            "reason": "Changes outside GH05T3-owned paths require Robert's approval.",
            "triggers": outside,
        }

    emergency_hits = _contains_any(joined, EMERGENCY_TERMS)
    if emergency or emergency_hits:
        return {
            "allowed": True,
            "level": "emergency",
            "reason": "Emergency repair may run immediately and must be logged.",
            "triggers": emergency_hits,
        }

    major_hits = _contains_any(joined, MAJOR_TERMS)
    if major_hits:
        return {
            "allowed": False,
            "level": "approval_required",
            "reason": "Major, risky, persistent, external, or disruptive work requires partial approval.",
            "triggers": major_hits,
        }

    return {
        "allowed": True,
        "level": "auto",
        "reason": "Allowed: GH05T3-owned enhancement or repair.",
        "triggers": [],
    }
