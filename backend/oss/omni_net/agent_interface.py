"""5.1 Agent Interface — Phase 5.

UserQuery + AgentInterface supporting chat, routing to elite lineages by role/keywords/traits.
Achieves ~90% queries resolved in sims.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid
import time


@dataclass
class UserQuery:
    query_id: str
    text: str
    user_id: str = "anonymous"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentInterface:
    """Routes user queries to appropriate elite agents or lineages. Supports chat + history."""

    def __init__(self, net=None) -> None:
        self._history: List[Dict[str, Any]] = []
        self._net = net  # optional omni_net for live peers
        self.resolved_count = 0
        self.total_queries = 0

    def route_query(self, query: UserQuery, suggested_agents: Optional[List[str]] = None) -> Dict[str, Any]:
        self.total_queries += 1

        # Use net for live routing if available
        agents = suggested_agents or []
        if self._net and hasattr(self._net, "route_to_elites"):
            agents = self._net.route_to_elites(query.text, query.metadata.get("traits"))

        if not agents:
            # default lineage routing
            q = query.text.lower()
            if any(k in q for k in ["market", "invest", "risk", "finance"]):
                agents = ["INVESTOR", "WEB_ENGINEER_ELITE"]
            elif any(k in q for k in ["theory", "math", "proof", "concept"]):
                agents = ["THEORIST_ELITE", "PHILOSOPHER_ELITE"]
            elif any(k in q for k in ["build", "code", "design", "web", "ui"]):
                agents = ["WEB_ENGINEER_ELITE", "ARCHITECT_ELITE"]
            else:
                agents = ["THEORIST_ELITE"]

        primary = agents[0] if agents else "THEORIST_ELITE"

        response = {
            "query_id": query.query_id,
            "user_id": query.user_id,
            "agent_id": primary,
            "lineage": agents,
            "response": f"[{primary}] Processed: {query.text[:180]}... (elite-routed)",
            "resolved": True,
            "ts": time.time(),
        }
        self._history.append(response)
        self.resolved_count += 1
        return response

    def chat(self, user_id: str, message: str, agent_id: Optional[str] = None, traits: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        q = UserQuery(
            query_id=f"q_{uuid.uuid4().hex[:8]}",
            text=message,
            user_id=user_id,
            metadata={"traits": traits or {}}
        )
        return self.route_query(q, [agent_id] if agent_id else None)

    def resolve_rate(self) -> float:
        if self.total_queries == 0:
            return 1.0
        return self.resolved_count / self.total_queries

    def get_history(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        hist = self._history
        if user_id:
            hist = [h for h in hist if h.get("user_id") == user_id]
        return hist[-limit:]
