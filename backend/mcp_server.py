"""GH05T3 MCP server — exposes the gateway as Model Context Protocol tools.

Mount via gateway_v3.py:
    from mcp_server import get_mcp_asgi, wire_gateway, MCP_AVAILABLE
    app.mount("/mcp", get_mcp_asgi())

Claude Code settings.json (remote via Tailscale):
    {
      "mcpServers": {
        "gh05t3": {
          "type": "sse",
          "url": "http://<tailscale-ip>:8002/mcp/sse",
          "headers": { "Authorization": "Bearer <GH05T3_API_TOKEN>" }
        }
      }
    }
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

log = logging.getLogger("gh0st3.mcp")

# ── Late-bound gateway state (set by wire_gateway at startup) ─────────────────
_bus            = None
_memory         = None
_omega          = None
_swarm          = None
_ghost          = None
_peer_registry  = None


def wire_gateway(bus, memory, omega, swarm, ghost, peer_registry) -> None:
    """Called from gateway_v3 lifespan after all subsystems are ready."""
    global _bus, _memory, _omega, _swarm, _ghost, _peer_registry
    _bus           = bus
    _memory        = memory
    _omega         = omega
    _swarm         = swarm
    _ghost         = ghost
    _peer_registry = peer_registry
    log.info("MCP server wired to gateway state")


# ── MCP tool definitions ──────────────────────────────────────────────────────
try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        "GH05T3",
        instructions=(
            "GH05T3 — sovereign AI mesh on TatorTot (Lenovo LOQ, RTX 5050). "
            "Five swarm agents: ARCHITECT, SENTINEL, SAGE, EXECUTOR, GITHUB. "
            "Tools: chat, status, memory recall, swarm delegation, GitHub push, "
            "GhostScript execution, peer mesh control."
        ),
    )

    # ── CHAT ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def chat(message: str) -> str:
        """Send a message through GH05T3's Omega Loop and get the swarm response.

        Passes through Ghost Protocol screening, then routes through SAGE/KAIROS
        and the active LLM backend. Returns JSON with response, mode, sage_score,
        latency_ms, and backend_used.
        """
        from swarm.bus import MsgType

        trap = await _ghost.process_input(message)
        if trap:
            return trap

        state = await _omega.run(message, None)
        await _bus.emit(
            src="CLAUDE",
            content=message,
            channel="#omega",
            msg_type=MsgType.CHAT,
        )
        return json.dumps({
            "response":     state.response,
            "mode":         state.mode.value,
            "sage_score":   state.sage_score,
            "sage_verdict": state.sage_verdict,
            "latency_ms":   state.latency_ms,
            "backend_used": state.backend_used,
            "cycle_id":     state.cycle_id,
        })

    # ── STATUS ────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def get_status() -> str:
        """Get full GH05T3 system status.

        Returns: omega_loop stats, swarm bus stats, active agent list,
        memory palace shard count, KAIROS cycle totals.
        """
        return json.dumps({
            "omega_loop": _omega.stats,
            "swarm":      _bus.stats,
            "agents":     list(_bus.agents.keys()),
            "memory":     _memory.stats(),
        })

    # ── MEMORY ────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def memory_recall(query: str, top_k: int = 5) -> str:
        """Search GH05T3's Memory Palace for semantically relevant memories.

        Args:
            query: Natural language query to search.
            top_k: Number of results to return (default 5).

        Returns JSON list of matching memory shards.
        """
        results = await _memory.recall(query=query, top_k=top_k)
        return json.dumps({"results": results, "count": len(results)})

    @mcp.tool()
    async def memory_store(content: str, room: str = "general", tags: str = "") -> str:
        """Store a new memory shard in GH05T3's Memory Palace.

        Args:
            content: The memory content to store.
            room:    Memory room/namespace (default 'general').
            tags:    Comma-separated tags for retrieval.

        Returns confirmation with shard ID.
        """
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        shard_id = await _memory.store(content=content, room=room, tags=tag_list)
        return json.dumps({"ok": True, "shard_id": shard_id, "room": room})

    # ── SWARM ─────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def swarm_delegate(task: str, agent: Optional[str] = None) -> str:
        """Delegate a task to GH05T3's swarm agents.

        Args:
            task:  The task description to delegate.
            agent: Optional target agent name: ARCHITECT, SENTINEL, SAGE,
                   EXECUTOR, or GITHUB. Omit to let the swarm auto-route.

        Returns JSON with routed_to agent name and task confirmation.
        """
        if not _swarm:
            return json.dumps({"ok": False, "error": "Swarm not initialized"})
        target = await _swarm.delegate(task, preferred_agent=agent)
        return json.dumps({"ok": True, "task": task, "routed_to": target})

    @mcp.tool()
    async def emit_to_bus(content: str, channel: str = "#broadcast", src: str = "CLAUDE") -> str:
        """Emit a message directly onto GH05T3's SwarmBus.

        All connected WebSocket clients (dashboard) and swarm agents will
        receive this message in real time.

        Args:
            content: Message content.
            channel: Bus channel (e.g. '#broadcast', '#omega', '#github').
            src:     Sender label shown in dashboard (default 'CLAUDE').
        """
        from swarm.bus import MsgType

        await _bus.emit(
            src=src,
            content=content,
            channel=channel,
            msg_type=MsgType.CHAT,
        )
        return json.dumps({"ok": True, "channel": channel, "src": src})

    # ── CONVERSATIONS ─────────────────────────────────────────────────────────

    @mcp.tool()
    async def get_conversations(n: int = 50, channel: Optional[str] = None) -> str:
        """Get recent swarm bus message history.

        Args:
            n:       Number of messages to return (default 50, max 500).
            channel: Filter to a specific channel (e.g. '#omega').

        Returns JSON with messages list and bus stats.
        """
        n = min(n, 500)
        return json.dumps({
            "messages": _bus.log.recent(n=n, channel=channel),
            "stats":    _bus.log.stats,
        })

    # ── GITHUB ────────────────────────────────────────────────────────────────

    @mcp.tool()
    async def github_push(files_json: str, message: str = "🖤 GH05T3 MCP push", branch: str = "main") -> str:
        """Push files to GH05T3's GitHub repository.

        Args:
            files_json: JSON object mapping file paths to content strings.
                        Example: '{"src/foo.py": "print(1)"}'
            message:    Commit message (default: GH05T3 MCP push).
            branch:     Target branch (default: main).

        Returns JSON with commit_url and files_pushed count.
        """
        from integrations.github_integration import GitHubAgent

        try:
            files = json.loads(files_json)
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"Invalid files_json: {e}"})

        gh = GitHubAgent()
        try:
            url = await gh._gh.push_files(files, message, branch)
            await gh.close()
            return json.dumps({"ok": True, "commit_url": url, "files_pushed": len(files)})
        except Exception as exc:
            await gh.close()
            return json.dumps({"ok": False, "error": str(exc)})

    @mcp.tool()
    async def github_status() -> str:
        """Get GH05T3's GitHub repository status (latest commits, branches, etc.)."""
        from integrations.github_integration import GitHubAgent

        gh = GitHubAgent()
        try:
            info = await gh._gh.repo_info()
            await gh.close()
            return json.dumps(info)
        except Exception as exc:
            await gh.close()
            return json.dumps({"ok": False, "error": str(exc)})

    # ── GHOSTSCRIPT ───────────────────────────────────────────────────────────

    @mcp.tool()
    async def ghostscript_run(src: str) -> str:
        """Execute a GhostScript program in GH05T3's AI scripting runtime.

        GhostScript is GH05T3's domain language — supports variables, async/await,
        pipeline operators, LLM calls, memory ops, and multi-agent routing.

        Example:
            let x = llm.chat("Summarize quantum computing in one line")
            memory.store(x, room="research")
            print(x)

        Returns JSON execution trace with output, proposals, and emits.
        """
        from ghostscript import run_async as _gs_run
        from ghost_llm import chat_once as _chat_once

        async def _llm_fn(prompt: str) -> str:
            text, _ = await _chat_once(session="mcp-gs", system="", user=prompt)
            return text

        result = await _gs_run(
            src,
            llm_fn=_llm_fn,
            memory_engine=_memory,
            agent_id="mcp-gs",
        )
        return json.dumps(result)

    # ── PEER MESH ─────────────────────────────────────────────────────────────

    @mcp.tool()
    async def list_peers() -> str:
        """List all GH05T3 peer instances discovered via Tailscale auto-discovery.

        Returns JSON with each peer's label, Tailscale IP, gateway URL,
        status, and active agent count.
        """
        if not _peer_registry:
            return json.dumps({"peers": [], "note": "Peer registry not initialized"})
        return json.dumps({
            "peers": _peer_registry.peers,
            "count": len(_peer_registry.peers),
        })

    @mcp.tool()
    async def refresh_peers() -> str:
        """Trigger immediate re-discovery of Tailscale peer instances.

        Calls Tailscale API and probes each device. Returns updated peer list.
        """
        if not _peer_registry:
            return json.dumps({"peers": [], "note": "Peer registry not initialized"})
        peers = await _peer_registry.refresh()
        return json.dumps({"peers": peers, "count": len(peers)})

    @mcp.tool()
    async def peer_delegate(peer_url: str, task: str, agent: Optional[str] = None) -> str:
        """Delegate a task to a specific GH05T3 peer instance.

        Args:
            peer_url: Base URL of the peer gateway (e.g. 'http://100.x.x.x:8002').
            task:     Task description to delegate.
            agent:    Optional preferred agent on the peer.

        Returns JSON response from the peer's swarm.
        """
        import httpx

        api_token = os.environ.get("GH05T3_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

        payload: dict = {"task": task}
        if agent:
            payload["agent"] = agent

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.post(
                    f"{peer_url.rstrip('/')}/swarm/delegate",
                    json=payload,
                    headers=headers,
                )
                return json.dumps(r.json())
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc), "peer": peer_url})

    @mcp.tool()
    async def peer_chat(peer_url: str, message: str) -> str:
        """Send a chat message to a specific GH05T3 peer instance.

        Args:
            peer_url: Base URL of the peer gateway (e.g. 'http://100.x.x.x:8002').
            message:  Message to send through the peer's Omega Loop.

        Returns JSON with the peer's response.
        """
        import httpx

        api_token = os.environ.get("GH05T3_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                r = await client.post(
                    f"{peer_url.rstrip('/')}/chat",
                    json={"message": message},
                    headers=headers,
                )
                return json.dumps(r.json())
            except Exception as exc:
                return json.dumps({"ok": False, "error": str(exc), "peer": peer_url})

    @mcp.tool()
    async def mesh_broadcast(content: str, channel: str = "#broadcast") -> str:
        """Broadcast a message to ALL known GH05T3 peer instances simultaneously.

        Emits on the local bus AND POSTs to every live peer's /swarm/broadcast.
        Returns per-peer delivery status.
        """
        import httpx
        from swarm.bus import MsgType

        # Emit locally
        await _bus.emit(
            src="CLAUDE",
            content=content,
            channel=channel,
            msg_type=MsgType.CHAT,
        )

        if not _peer_registry or not _peer_registry.peers:
            return json.dumps({"ok": True, "local": True, "peers": [], "note": "No remote peers"})

        api_token = os.environ.get("GH05T3_API_TOKEN", "")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}

        results = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for peer in _peer_registry.peers:
                try:
                    r = await client.post(
                        f"{peer['url']}/swarm/broadcast",
                        params={"content": content, "src": "MESH"},
                        headers=headers,
                    )
                    results.append({"peer": peer["label"], "ok": r.status_code == 200})
                except Exception as exc:
                    results.append({"peer": peer["label"], "ok": False, "error": str(exc)})

        return json.dumps({"ok": True, "local": True, "peers": results})

    # ── ASGI factory ─────────────────────────────────────────────────────────

    def get_mcp_asgi():
        """Return the Starlette ASGI app for SSE MCP transport."""
        return mcp.sse_app()

    MCP_AVAILABLE = True
    log.info(
        "MCP server initialised — 14 tools: chat, get_status, memory_recall, "
        "memory_store, swarm_delegate, emit_to_bus, get_conversations, "
        "github_push, github_status, ghostscript_run, list_peers, "
        "refresh_peers, peer_delegate, peer_chat, mesh_broadcast"
    )

except ImportError:
    MCP_AVAILABLE = False

    def get_mcp_asgi():
        return None

    log.warning(
        "mcp package not installed — MCP server disabled. "
        "Run: pip install mcp"
    )
