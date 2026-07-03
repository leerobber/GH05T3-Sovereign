"""
GH05T3 — SWARM COMMUNICATION BUS v3
======================================
The nervous system of the entire GH05T3 organism.

Every message between agents, nodes, and specialists passes
through this bus. It records ALL conversations in real-time,
persists them to JSONL, streams them to the dashboard via WebSocket,
and enables agent-to-agent direct messaging.

Architecture:
  SwarmBus         — central message router (singleton)
  SwarmChannel     — named pub/sub channel (one per agent pair / topic)
  SwarmAgent       — base class all agents inherit
  ConversationLog  — persistent JSONL chat archive
  BusWebSocketRelay— streams live to dashboard

Channels:
  #broadcast       — all agents receive
  #omega           — Omega Loop decisions
  #sage            — SAGE validation verdicts
  #kairos          — KAIROS cycle events
  #github          — GitHub integration events
  #claude          — Claude API assistance events
  #swarm/{agent}   — direct agent channels
  #tia             — Android/TIA relay channel
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Awaitable, Any, Optional
from enum import Enum

log = logging.getLogger("gh0st3.swarm.bus")

CHAT_LOG_PATH = Path("memory/conversations.jsonl")
MAX_HISTORY   = 5000   # messages kept in-memory ring buffer


# ─────────────────────────────────────────────
# MESSAGE TYPES
# ─────────────────────────────────────────────

class MsgType(str, Enum):
    CHAT       = "chat"        # agent says something to another agent
    TASK       = "task"        # delegate a task
    RESULT     = "result"      # task result returned
    THOUGHT    = "thought"     # internal reasoning (streamed to dashboard)
    CRITIQUE   = "critique"    # SAGE critique
    VERDICT    = "verdict"     # SAGE verdict
    KAIROS     = "kairos"      # evolutionary cycle event
    GITHUB     = "github"      # GitHub push/PR/commit event
    CLAUDE     = "claude"      # Claude API call/response
    SYSTEM     = "system"      # system-level event
    HEARTBEAT  = "heartbeat"   # node alive signal
    ERROR      = "error"       # error event


@dataclass
class SwarmMessage:
    id:        str       = field(default_factory=lambda: str(uuid.uuid4())[:12])
    channel:   str       = "#broadcast"
    msg_type:  MsgType   = MsgType.CHAT
    src:       str       = "system"        # sender agent/node ID
    dst:       str       = "*"             # recipient ("*" = broadcast)
    content:   str       = ""
    metadata:  dict      = field(default_factory=dict)
    timestamp: float     = field(default_factory=time.time)
    seq:       int       = 0               # monotonic sequence per channel

    def to_dict(self) -> dict:
        d = asdict(self)
        d["msg_type"] = self.msg_type.value
        d["ts_human"] = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ─────────────────────────────────────────────
# CONVERSATION LOG — persistent archive
# ─────────────────────────────────────────────

class ConversationLog:
    """
    Appends every SwarmMessage to JSONL.
    Loads recent history on startup.
    Provides search by agent, channel, time range.
    """

    def __init__(self, path: Path = CHAT_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ring: deque[SwarmMessage] = deque(maxlen=MAX_HISTORY)
        self._load_recent()

    def _load_recent(self):
        """Load last N messages from disk on startup."""
        if not self.path.exists():
            return
        lines = self.path.read_text().splitlines()
        for line in lines[-MAX_HISTORY:]:
            try:
                d = json.loads(line)
                d["msg_type"] = MsgType(d.get("msg_type", "chat"))
                msg = SwarmMessage(**{k: v for k, v in d.items()
                                      if k in SwarmMessage.__dataclass_fields__})
                self._ring.append(msg)
            except Exception:
                pass
        log.info(f"[ConvLog] Loaded {len(self._ring)} historical messages")

    def append(self, msg: SwarmMessage):
        self._ring.append(msg)
        with open(self.path, "a") as f:
            f.write(msg.to_json() + "\n")

    def recent(self, n: int = 100, channel: str = None,
               src: str = None, msg_type: MsgType = None) -> list[dict]:
        msgs = list(self._ring)
        if channel:
            msgs = [m for m in msgs if m.channel == channel]
        if src:
            msgs = [m for m in msgs if m.src == src]
        if msg_type:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        return [m.to_dict() for m in msgs[-n:]]

    def search(self, query: str, limit: int = 50) -> list[dict]:
        q = query.lower()
        hits = [m for m in self._ring
                if q in m.content.lower() or q in m.src.lower()]
        return [m.to_dict() for m in hits[-limit:]]

    @property
    def stats(self) -> dict:
        msgs = list(self._ring)
        by_agent: dict[str, int] = defaultdict(int)
        by_channel: dict[str, int] = defaultdict(int)
        by_type: dict[str, int] = defaultdict(int)
        for m in msgs:
            by_agent[m.src] += 1
            by_channel[m.channel] += 1
            by_type[m.msg_type.value] += 1
        return {
            "total":      len(msgs),
            "by_agent":   dict(by_agent),
            "by_channel": dict(by_channel),
            "by_type":    dict(by_type),
            "oldest":     msgs[0].timestamp if msgs else None,
            "newest":     msgs[-1].timestamp if msgs else None,
        }


# ─────────────────────────────────────────────
# SWARM BUS — central router
# ─────────────────────────────────────────────

Handler = Callable[[SwarmMessage], Awaitable[None]]


class SwarmBus:
    """
    Central message bus for the entire GH05T3 swarm.
    Thread-safe, async. Singleton per process.

    Usage:
        bus = SwarmBus.instance()
        await bus.publish(SwarmMessage(src="omega", content="Cycle complete", channel="#omega"))
        bus.subscribe("#sage", my_handler)
    """

    _instance: Optional["SwarmBus"] = None

    def __init__(self):
        self._subs:       dict[str, list[Handler]] = defaultdict(list)
        self._ws_clients: list[asyncio.Queue]       = []   # WS relay queues
        self._seq:        dict[str, int]            = defaultdict(int)
        self.log          = ConversationLog()
        self._lock        = asyncio.Lock()
        self._agents:     dict[str, dict]           = {}   # registered agents

    @classmethod
    def instance(cls) -> "SwarmBus":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── REGISTRY ──────────────────────────────

    def register_agent(self, agent_id: str, meta: dict):
        """Register an agent so it appears in the dashboard."""
        self._agents[agent_id] = {**meta, "registered_at": time.time(), "active": True}
        log.info(f"[Bus] Agent registered: {agent_id}")

    def deregister_agent(self, agent_id: str):
        if agent_id in self._agents:
            self._agents[agent_id]["active"] = False

    @property
    def agents(self) -> dict:
        return dict(self._agents)

    # ── SUBSCRIBE / PUBLISH ───────────────────

    def subscribe(self, channel: str, handler: Handler):
        """Subscribe to a channel. '#broadcast' receives all messages."""
        self._subs[channel].append(handler)

    def unsubscribe(self, channel: str, handler: Handler):
        """Remove a previously registered handler from a channel."""
        ch = self._subs.get(channel)
        if ch and handler in ch:
            ch.remove(handler)

    def subscribe_all(self, handler: Handler):
        """Receive every message on every channel."""
        self.subscribe("__ALL__", handler)

    async def publish(self, msg: SwarmMessage) -> int:
        """
        Publish a message. Returns number of handlers that received it.
        Automatically logs to ConversationLog and streams to WS clients.
        """
        async with self._lock:
            self._seq[msg.channel] += 1
            msg.seq = self._seq[msg.channel]

        # Persist
        self.log.append(msg)

        # Deliver to handlers
        handlers = (
            self._subs.get(msg.channel, []) +
            self._subs.get("#broadcast", []) +
            self._subs.get("__ALL__", [])
        )

        tasks = [h(msg) for h in handlers]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error(f"[Bus] Handler error: {r}")

        # Stream to WebSocket clients
        payload = msg.to_json()
        dead = []
        for q in self._ws_clients:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for d in dead:
            self._ws_clients.remove(d)

        return len(handlers)

    async def emit(self, src: str, content: str, channel: str = "#broadcast",
                   msg_type: MsgType = MsgType.CHAT, dst: str = "*",
                   **metadata) -> SwarmMessage:
        """Convenience wrapper for publish."""
        msg = SwarmMessage(
            src=src, content=content, channel=channel,
            msg_type=msg_type, dst=dst, metadata=metadata,
        )
        await self.publish(msg)
        return msg

    # ── WEBSOCKET RELAY ───────────────────────

    def add_ws_client(self) -> asyncio.Queue:
        """Create a queue for a new WebSocket client. Returns the queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._ws_clients.append(q)
        # Replay last 50 messages on connect
        for m in self.log.recent(50):
            try:
                q.put_nowait(json.dumps(m))
            except asyncio.QueueFull:
                break
        return q

    def remove_ws_client(self, q: asyncio.Queue):
        if q in self._ws_clients:
            self._ws_clients.remove(q)

    # ── STATS / QUERY ─────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "agents":       len(self._agents),
            "active_agents": sum(1 for a in self._agents.values() if a.get("active")),
            "channels":     list(self._subs.keys()),
            "ws_clients":   len(self._ws_clients),
            "log":          self.log.stats,
        }

    async def direct(self, src: str, dst: str, content: str,
                     msg_type: MsgType = MsgType.CHAT, **kw) -> SwarmMessage:
        """Send a direct message from src → dst (also broadcasts for logging)."""
        return await self.emit(src=src, content=content,
                                channel=f"#swarm/{dst}",
                                msg_type=msg_type, dst=dst, **kw)


# ─────────────────────────────────────────────
# SWARM AGENT BASE CLASS
# ─────────────────────────────────────────────

class SwarmAgent:
    """
    Base class all GH05T3 specialist agents inherit.
    Provides: bus access, pub/sub, conversation participation,
    structured logging to the conversation log.
    """

    ROLE        = "agent"
    DESCRIPTION = "Generic swarm agent"
    CHANNELS    = ["#broadcast"]

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or f"{self.ROLE}-{str(uuid.uuid4())[:6]}"
        self.bus      = SwarmBus.instance()
        self._running = False
        self._task_count = 0
        self._msg_count  = 0
        self.boot_time   = time.time()

        # Register
        self.bus.register_agent(self.agent_id, {
            "role":        self.ROLE,
            "description": self.DESCRIPTION,
            "channels":    self.CHANNELS,
        })

        # Subscribe to own channel + requested channels
        for ch in self.CHANNELS:
            self.bus.subscribe(ch, self._handle)
        self.bus.subscribe(f"#swarm/{self.agent_id}", self._handle)

    async def _handle(self, msg: SwarmMessage):
        """Route inbound messages to on_message."""
        if msg.src == self.agent_id:
            return   # ignore own messages
        self._msg_count += 1
        try:
            await self.on_message(msg)
        except Exception as e:
            log.error(f"[{self.agent_id}] on_message error: {e}")
            await self.say(f"Error handling {msg.msg_type}: {e}",
                            channel="#broadcast", msg_type=MsgType.ERROR)

    async def on_message(self, msg: SwarmMessage):
        """Override in subclass to handle inbound messages."""
        pass

    async def say(self, content: str, channel: str = "#broadcast",
                  msg_type: MsgType = MsgType.CHAT,
                  dst: str = "*", **metadata) -> SwarmMessage:
        """Publish a message as this agent."""
        return await self.bus.emit(
            src=self.agent_id, content=content,
            channel=channel, msg_type=msg_type, dst=dst, **metadata,
        )

    async def think(self, thought: str):
        """Publish an internal reasoning step (visible in dashboard)."""
        await self.say(thought, channel=f"#swarm/{self.agent_id}",
                       msg_type=MsgType.THOUGHT)

    async def dm(self, dst: str, content: str, **kw) -> SwarmMessage:
        """Send a direct message to another agent."""
        return await self.bus.direct(self.agent_id, dst, content, **kw)

    async def task(self, dst: str, task_desc: str, payload: dict = None) -> SwarmMessage:
        """Delegate a task to another agent."""
        self._task_count += 1
        return await self.bus.direct(
            self.agent_id, dst, task_desc,
            msg_type=MsgType.TASK,
            payload=payload or {},
            task_id=str(uuid.uuid4())[:8],
        )

    @property
    def stats(self) -> dict:
        return {
            "agent_id":   self.agent_id,
            "role":       self.ROLE,
            "uptime":     time.time() - self.boot_time,
            "tasks":      self._task_count,
            "msgs_recv":  self._msg_count,
        }
