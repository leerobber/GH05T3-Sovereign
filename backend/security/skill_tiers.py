"""GH05T3 — Skill Tier System: T1–T4 progressive access gates.

Maps the manifesto's 26.1% vulnerability gap fix onto the sovereign-forge
Armory pass model. Every SKILL.md must declare a tier; the runtime enforces
permissions before execution.

Tier definitions:
    T1 — Community/unknown:     instructions-only, sandboxed, no MCP exec
    T2 — Self-authored:         limited MCP read, no shell, no network I/O
    T3 — ZERO Committee reviewed: full MCP, restricted shell (no rm/curl/ssh)
    T4 — Chaos-tested + signed: full exec, network I/O, unrestricted shell

The SkillRegistry loads skills from the filesystem and enforces tier checks
at dispatch time. Tier upgrades require ZERO Committee sign-off (stub: the
`approve()` method records the decision; production should require multi-sig).

Env vars:
    SKILL_MAX_TIER   maximum tier allowed in this deployment (default 4)
    SKILL_AUDIT_LOG  path to tier access audit JSONL (default security/skill_audit.jsonl)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("ghost.skill_tiers")

SKILL_MAX_TIER = int(os.environ.get("SKILL_MAX_TIER", "4"))
AUDIT_LOG      = Path(os.environ.get("SKILL_AUDIT_LOG", "security/skill_audit.jsonl"))


# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------

class SkillTier(IntEnum):
    T1_COMMUNITY  = 1   # instructions-only, sandboxed
    T2_SELF       = 2   # limited MCP read, no shell
    T3_REVIEWED   = 3   # full MCP, restricted shell
    T4_CERTIFIED  = 4   # full exec + network I/O


TIER_PERMISSIONS: dict[int, dict] = {
    1: {
        "mcp_read":    False,
        "mcp_write":   False,
        "shell":       False,
        "network":     False,
        "description": "Instructions-only, fully sandboxed. No external calls.",
    },
    2: {
        "mcp_read":    True,
        "mcp_write":   False,
        "shell":       False,
        "network":     False,
        "description": "Limited MCP read. No shell or network I/O.",
    },
    3: {
        "mcp_read":    True,
        "mcp_write":   True,
        "shell":       True,   # restricted: no rm -rf / curl / ssh
        "network":     False,
        "description": "Full MCP, restricted shell. ZERO Committee reviewed.",
    },
    4: {
        "mcp_read":    True,
        "mcp_write":   True,
        "shell":       True,
        "network":     True,
        "description": "Full exec + network. Chaos-tested and ZERO-signed.",
    },
}

# Shell commands blocked at T3 even with shell=True
T3_SHELL_BLOCKLIST = frozenset({
    "rm", "rmdir", "del", "curl", "wget", "ssh", "scp", "nc", "netcat",
    "python -c", "exec(", "eval(", "subprocess", "os.system",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SkillRecord:
    skill_id:     str
    name:         str
    tier:         int
    author:       str         = "unknown"
    approved_by:  list[str]   = field(default_factory=list)
    approved_at:  float       = 0.0
    chaos_tested: bool        = False
    description:  str         = ""
    path:         str         = ""

    @property
    def permissions(self) -> dict:
        return TIER_PERMISSIONS.get(self.tier, TIER_PERMISSIONS[1])

    def can_exec_shell(self, command: str) -> bool:
        if not self.permissions["shell"]:
            return False
        if self.tier == 3:
            import re
            # Normalize whitespace before matching — prevents `python   -c` bypass
            cmd_normalized = re.sub(r"\s+", " ", command.lower().strip())
            return not any(blocked in cmd_normalized for blocked in T3_SHELL_BLOCKLIST)
        return True  # T4 — unrestricted

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Manages skill registration, tier enforcement, and audit logging."""

    def __init__(self, max_tier: int = SKILL_MAX_TIER):
        self._skills:   dict[str, SkillRecord] = {}
        self._max_tier = max_tier
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

    def register(self, skill: SkillRecord) -> SkillRecord:
        """Register a skill. Raises PermissionError if tier > max_tier."""
        if skill.tier > self._max_tier:
            raise PermissionError(
                f"Skill {skill.skill_id!r} requests tier {skill.tier} "
                f"but SKILL_MAX_TIER={self._max_tier}"
            )
        self._skills[skill.skill_id] = skill
        LOG.info("[skills] registered %s (tier=T%d, author=%s)",
                 skill.skill_id, skill.tier, skill.author)
        return skill

    def approve(self, skill_id: str, approver: str) -> SkillRecord:
        """Record ZERO Committee approval. T3+ requires at least one approver."""
        skill = self._skills.get(skill_id)
        if not skill:
            raise KeyError(f"Skill {skill_id!r} not registered")
        if approver not in skill.approved_by:
            skill.approved_by.append(approver)
        skill.approved_at = time.time()
        self._audit("approve", skill_id, approver)
        return skill

    def check_permission(self, skill_id: str, action: str,
                         command: str = "") -> bool:
        """Return True if the skill is allowed to perform action.

        Actions: 'mcp_read', 'mcp_write', 'shell', 'network'
        """
        skill = self._skills.get(skill_id)
        if not skill:
            LOG.warning("[skills] unknown skill %r denied action %r", skill_id, action)
            return False

        if action == "shell":
            allowed = skill.can_exec_shell(command)
        else:
            allowed = skill.permissions.get(action, False)

        self._audit(action, skill_id, "system", allowed=allowed, command=command)

        if not allowed:
            LOG.warning("[skills] %s DENIED %r (tier=T%d, action=%s)",
                        skill_id, command[:40] or action, skill.tier, action)
        return allowed

    def get(self, skill_id: str) -> Optional[SkillRecord]:
        return self._skills.get(skill_id)

    def list_by_tier(self, tier: int) -> list[SkillRecord]:
        return [s for s in self._skills.values() if s.tier == tier]

    def summary(self) -> dict:
        by_tier: dict[int, int] = {}
        for s in self._skills.values():
            by_tier[s.tier] = by_tier.get(s.tier, 0) + 1
        return {
            "total":    len(self._skills),
            "max_tier": self._max_tier,
            "by_tier":  {f"T{k}": v for k, v in sorted(by_tier.items())},
        }

    def _audit(self, event: str, skill_id: str, actor: str,
               allowed: bool = True, command: str = ""):
        entry = {
            "ts":       time.time(),
            "event":    event,
            "skill_id": skill_id,
            "actor":    actor,
            "allowed":  allowed,
        }
        if command:
            entry["command"] = command[:80]
        try:
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            LOG.debug("skill audit write failed: %s", e)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_registry: Optional[SkillRegistry] = None


def get_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
