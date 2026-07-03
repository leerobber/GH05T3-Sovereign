"""5.4 Multi-Agent Chat — Phase 5.

MultiAgentChat: start sessions, send, history, economy rewards per message.
Rate limits for beta.
Sim supports 5+ agents real-time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import uuid


@dataclass
class ChatMessage:
    msg_id: str
    session_id: str
    from_agent: str
    to_agent: str
    content: str
    ts: float = field(default_factory=time.time)
    reward: float = 0.0


class MultiAgentChat:
    """Real-time multi agent chat with rewards."""

    def __init__(self, economy=None, rate_limit_per_min: int = 60):
        self.sessions: Dict[str, List[ChatMessage]] = {}
        self.economy = economy  # optional OmniEconomy
        self.rate_limit = rate_limit_per_min
        self._send_counts: Dict[str, List[float]] = {}

    def start_session(self, participants: List[str], topic: str = "") -> str:
        sid = f"chat_{uuid.uuid4().hex[:10]}"
        self.sessions[sid] = []
        # initial system
        self.sessions[sid].append(ChatMessage(
            msg_id=f"m_{uuid.uuid4().hex[:6]}",
            session_id=sid,
            from_agent="system",
            to_agent="all",
            content=f"Session started with {participants}. Topic: {topic}"
        ))
        return sid

    def send(self, session_id: str, from_agent: str, to_agent: str, content: str) -> ChatMessage:
        # rate limit
        now = time.time()
        self._send_counts.setdefault(from_agent, []).append(now)
        recent = [t for t in self._send_counts[from_agent] if now - t < 60]
        self._send_counts[from_agent] = recent
        if len(recent) > self.rate_limit:
            raise RuntimeError(f"Rate limit exceeded for {from_agent}")

        if session_id not in self.sessions:
            raise ValueError("No such session")

        msg = ChatMessage(
            msg_id=f"m_{uuid.uuid4().hex[:8]}",
            session_id=session_id,
            from_agent=from_agent,
            to_agent=to_agent,
            content=content[:800]
        )

        # Economy reward (small for chat)
        if self.economy:
            try:
                reward = 1.0 + (0.5 if "THEORIST" in from_agent.upper() else 0)
                self.economy.reward(from_agent, reward, reason="multi_agent_chat")
                msg.reward = reward
            except Exception:
                pass

        self.sessions[session_id].append(msg)
        return msg

    def history(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        return self.sessions.get(session_id, [])[-limit:]

    def active_agents(self, session_id: str) -> List[str]:
        msgs = self.sessions.get(session_id, [])
        agents = set(m.from_agent for m in msgs if m.from_agent != "system")
        return list(agents)

    def simulate_conversation(self, agents: List[str], turns: int = 5) -> str:
        """Helper for tests: creates a 5+ agent real-time feeling chat."""
        sid = self.start_session(agents, "Phase 5 multi chat sim")
        for i in range(turns):
            frm = agents[i % len(agents)]
            to = agents[(i+1) % len(agents)]
            self.send(sid, frm, to, f"Round {i}: idea from {frm}")
        return sid
