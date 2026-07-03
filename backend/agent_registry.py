"""Minimal agent registry stub — wraps the SQLite agent_registry.db used by backend services."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).resolve().parent / "data" / "agent_registry.db"

_BUILTIN_AGENTS: list[dict[str, Any]] = [
    {"id": "oracle", "display_name": "Iris Chen", "description": "Research oracle.", "category": "general", "tags": ["oracle", "research"], "status": "active"},
    {"id": "forge", "display_name": "Morgan Blake", "description": "Senior software engineer and code generator.", "category": "custom", "tags": ["coding", "developer"], "status": "active"},
    {"id": "codex", "display_name": "Alex Codex", "description": "Code specialist.", "category": "custom", "tags": ["coding"], "status": "active"},
    {"id": "nexus", "display_name": "Nexus", "description": "Synthesis and coordination.", "category": "general", "tags": ["ops", "coordination"], "status": "active"},
    {"id": "sentinel", "display_name": "Sentinel", "description": "Security and compliance.", "category": "custom", "tags": ["security"], "status": "active"},
    {"id": "avery", "display_name": "Avery", "description": "General purpose sovereign agent.", "category": "general", "tags": ["general"], "status": "active"},
]

_seeded: bool = False


def seed_builtins() -> int:
    """Seed built-in agents; idempotent."""
    global _seeded
    _seeded = True
    return len(_BUILTIN_AGENTS)


def list_agents(*, status: str = "active", limit: int = 500, **_kwargs: Any) -> list[dict[str, Any]]:
    """Return list of agent manifests."""
    agents = [a for a in _BUILTIN_AGENTS if not status or a.get("status") == status]
    return agents[:limit]


def get_agent(slug: str) -> dict[str, Any] | None:
    """Fetch agent manifest by id (case-insensitive)."""
    slug_lower = slug.strip().lower()
    for a in _BUILTIN_AGENTS:
        if a["id"].lower() == slug_lower:
            return dict(a)
    return None


def registry_stats() -> dict[str, Any]:
    """Return registry statistics."""
    return {
        "total": len(_BUILTIN_AGENTS),
        "active": len([a for a in _BUILTIN_AGENTS if a.get("status") == "active"]),
        "seeded": _seeded,
    }
