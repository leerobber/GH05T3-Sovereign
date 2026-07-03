"""
GH05T3 â€” GATEWAY v3 (Extended)
================================
Drop-in replacement for integrations/api_server.py.
Extends the original with:
  â€¢ WebSocket /ws â€” streams ALL swarm bus messages live
  â€¢ GET /conversations â€” paginated conversation log
  â€¢ GET /conversations/search â€” full-text search
  â€¢ GET /swarm/agents â€” live agent registry + stats
  â€¢ POST /swarm/delegate â€” delegate task to swarm
  â€¢ POST /claude/train â€” trigger Claude training batch
  â€¢ POST /claude/review â€” Claude architecture review
  â€¢ GET /github/status â€” repo info
  â€¢ POST /github/push â€” push files
  â€¢ POST /github/sync-memory â€” push memory to GitHub
  â€¢ WS /ws â€” unified live event stream (swarm bus relay)
  â€¢ All original Omega Loop, KAIROS, Memory, KillSwitch endpoints preserved

Mount at the same port as the original (8000).
"""

# â”€â”€ Aethyro license gate â€” GH05T3 will not start without an active trial/subscription â”€â”€
import os as _aeos
if _aeos.environ.get("AETHYRO_SKIP_LICENSE") != "1":
    try:
        from aethyro_license import gate as _ae_gate
    except Exception:
        try:
            from backend.aethyro_license import gate as _ae_gate
        except Exception:
            _ae_gate = None
    if _ae_gate:
        _ae_gate()
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from string import Template

# Pre-compiled templates for log content to reduce .format() overhead in KAIROS (892+ cycles) and swarm paths (~31s inference window)
KAIROS_EMIT_TMPL = Template("Cycle #${id} recorded â€” score=${score:.2f} verdict=${verdict}")
DISCOVERY_EMIT_TMPL = Template("DISCOVERY complete - slip ${slip:.5f} (improve ${improve}%) shadow=${shadow} p95=${p95}ms expert=${expert}")
SHADOW_EMIT_TMPL = Template("SHADOW_LOOP completed: max_drift=${max_d} hard_stops=${hard_stops}")

from core.config import (BACKENDS, GATEWAY_HOST, GATEWAY_PORT,
                          GITHUB_PAT, GITHUB_REPO, GITHUB_BRANCH,
                          OLLAMA_BASE, GH05T3_MODEL_URL)
from core.omega_loop import OmegaLoop
try:
    from memory.memory_palace import MemoryPalace
except (ImportError, ModuleNotFoundError):
    import importlib.util as _ilu, os as _gw_os
    _mp_path = _gw_os.path.join(_gw_os.path.dirname(__file__), "memory", "memory_palace.py")
    _mp_spec = _ilu.spec_from_file_location("backend.memory.memory_palace", _mp_path)
    _mp_mod = _ilu.module_from_spec(_mp_spec)  # type: ignore[arg-type]
    _mp_spec.loader.exec_module(_mp_mod)  # type: ignore[union-attr]
    MemoryPalace = _mp_mod.MemoryPalace
from evolution.kairos import KAIROS
# AUTO-DISABLED by GH05T3 aggressive engine: from backend.oss.financial.liquidity_routing import get_router, RouteStrategy
pass  # safe placeholder
from evolution.sage import SAGE
from security.ghost_protocol import GhostProtocol, KillSwitchMode

# v3 additions
from swarm.bus import SwarmBus, SwarmMessage, MsgType
from swarm.agents import GH05T3Swarm
from swarm.peer_registry import PeerRegistry
from integrations.claude_integration import ClaudeSwarmAgent
from integrations.github_integration import GitHubAgent, create_github_webhook_router
from mcp_server import get_mcp_asgi, wire_gateway, MCP_AVAILABLE
from integrations.stripe_integration import (
    verify_stripe_signature, process_stripe_event,
    all_subscribers, subscriber_count, STRIPE_WEBHOOK_SECRET,
)
from personas import team_roster, get_persona
from integrations.story_editor import (
    story_editor_greeting, story_editor_turn,
    get_session, reset_session, list_sessions,
)
from agent_marketplace import (
    JobQueue, ingest_github_event, ingest_stripe_event, ingest_cve_feed,
)
from integrations.lemonade_integration import (
    lemonade_available as _lemonade_ok,
    transcribe  as _lemonade_transcribe,
    speak       as _lemonade_speak,
    generate_image as _lemonade_generate_image,
    lemonade_status as _lemonade_status,
)
import v1_router

# â”€â”€ Marketplace API key â€” protects internal write endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_MARKETPLACE_KEY = os.environ.get("MARKETPLACE_API_KEY", "")


def _require_marketplace_auth(request: Request):
    """Reject requests missing a valid X-API-Key header.

    Set MARKETPLACE_API_KEY in backend/.env to enable enforcement.
    Empty key = dev-mode (accept all) â€” never leave empty in production.
    """
    if not _MARKETPLACE_KEY:
        return  # dev mode â€” no key configured
    provided = request.headers.get("X-API-Key", "")
    if not provided or provided != _MARKETPLACE_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing X-API-Key")

log = logging.getLogger("gh0st3.gateway_v3")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# BEARER AUTH MIDDLEWARE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_PUBLIC_PATHS = {"/", "/health", "/setup/secrets", "/setup/secrets/status", "/metrics"}
_PUBLIC_PREFIXES = ("/ws", "/mcp/sse", "/mcp/messages")


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require Authorization: Bearer <GH05T3_API_TOKEN> on all non-public endpoints.

    When GH05T3_API_TOKEN is empty the gateway runs in open/dev mode (no auth).
    Set the token in backend/.env or via install.ps1 for remote-access security.
    """

    async def dispatch(self, request: Request, call_next):
        # CORS preflight â€” always allow
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        # Public endpoints â€” no auth needed
        if path in _PUBLIC_PATHS:
            return await call_next(request)
        if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
            return await call_next(request)

        token = os.environ.get("GH05T3_API_TOKEN", "").strip()
        if not token:
            # Dev mode â€” no token configured, allow everything
            return await call_next(request)

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:].strip() != token:
            return JSONResponse(
                {"error": "Unauthorized", "hint": "Set Authorization: Bearer <GH05T3_API_TOKEN>"},
                status_code=401,
            )

        return await call_next(request)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SYSTEM INIT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

bus            = SwarmBus.instance()
memory         = MemoryPalace()
kairos         = KAIROS()
sage           = SAGE()
omega          = OmegaLoop(memory=memory, kairos=kairos, sage=sage)
ghost          = GhostProtocol()
peer_registry  = PeerRegistry()
swarm          = None   # initialized in lifespan
github         = None
claude         = None

# Cached backend health â€” updated by background task, never blocks /health
_backend_cache: dict = {
    **{name: "unknown" for name in BACKENDS},
    "ollama": "unknown",
    "gh05t3": "unknown",
}
_boot_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global swarm, github, claude

    # Boot swarm agents
    swarm  = GH05T3Swarm()
    github = GitHubAgent()
    claude = ClaudeSwarmAgent(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    await swarm.boot_announcement()

    # Start Tailscale peer discovery
    await peer_registry.start()

    # Wire gateway state into MCP server tools
    wire_gateway(bus, memory, omega, swarm, ghost, peer_registry)

    await bus.emit(
        src="GATEWAY",
        content=(
            "ðŸ–¤ GH05T3 v3 GATEWAY ONLINE â€” "
            f"Omega Loop + Swarm + Claude + GitHub + MCP({'âœ“' if MCP_AVAILABLE else 'âœ—'}) ACTIVE"
        ),
        channel="#broadcast",
        msg_type=MsgType.SYSTEM,
    )

    # Background backend health probe (non-blocking, updates cache every 30s)
    async def _probe_backends():
        import asyncio as _asyncio
        while True:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    async def _check_vllm(name, url):
                        try:
                            r = await client.get(f"{url}/health", timeout=1.5)
                            _backend_cache[name] = "online" if r.status_code == 200 else "degraded"
                        except Exception:
                            _backend_cache[name] = "offline"

                    async def _check_ollama():
                        try:
                            r = await client.get(f"{OLLAMA_BASE}/api/tags", timeout=1.5)
                            _backend_cache["ollama"] = "online" if r.status_code == 200 else "degraded"
                        except Exception:
                            _backend_cache["ollama"] = "offline"

                    async def _check_gh05t3():
                        try:
                            r = await client.get(f"{GH05T3_MODEL_URL}/health", timeout=1.5)
                            if r.status_code == 200:
                                _backend_cache["gh05t3"] = "online"
                                return
                        except Exception:
                            pass
                        try:
                            r = await client.get(f"{GH05T3_MODEL_URL}/v1/models", timeout=1.5)
                            _backend_cache["gh05t3"] = "online" if r.status_code == 200 else "degraded"
                        except Exception:
                            _backend_cache["gh05t3"] = "offline"

                    await _asyncio.gather(
                        *[_check_vllm(n, u) for n, u in BACKENDS.items()],
                        _check_ollama(),
                        _check_gh05t3(),
                    )
            except Exception:
                pass
            await _asyncio.sleep(30)

    asyncio.create_task(_probe_backends(), name="backend-probe")

    log.info("GH05T3 v3 gateway online (MCP=%s)", "enabled" if MCP_AVAILABLE else "disabled")
    yield

    await peer_registry.stop()
    await swarm.shutdown()
    await github.close()
    await claude.close()
    await omega.close()
    await sage.close()
    log.info("GH05T3 v3 shutdown complete")


app = FastAPI(
    title="GH05T3 v3",
    description="Unified swarm gateway â€” Omega Loop Â· SAGE Â· KAIROS Â· Claude Â· GitHub Â· MCP",
    version="3.0.0",
    lifespan=lifespan,
)

# Middleware order: CORS first, then auth (FastAPI applies in reverse registration order)
app.add_middleware(BearerAuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount GitHub webhook router
app.include_router(create_github_webhook_router())

# Mount Aethyro AIOS v1 API (agents, sessions, memory cortex) â€” same
# BearerAuthMiddleware protection as the rest of this app, no separate auth needed
app.include_router(v1_router.router)

# Mount OSS Omni-Sentient Singularity API (Phase 1 â€” DNA + Mind swarm)
try:
    import sys as _oss_sys
    from pathlib import Path as _OssPath
    _oss_root = str(_OssPath(__file__).resolve().parent.parent)
    if _oss_root not in _oss_sys.path:
        _oss_sys.path.insert(0, _oss_root)
    from oss.api.router import router as _oss_router
    from oss.pact.provider_states import router as _pact_states_router
    app.include_router(_oss_router, prefix="/oss")
    app.include_router(_pact_states_router)
    log.info("OSS router mounted at /oss")
except Exception as _oss_err:
    log.warning("OSS router not mounted: %s", _oss_err)

# Mount MCP SSE server at /mcp (Claude Code connects to /mcp/sse)
_mcp_asgi = get_mcp_asgi()
if _mcp_asgi is not None:
    app.mount("/mcp", _mcp_asgi)
    log.info("MCP server mounted at /mcp/sse")
else:
    log.warning("MCP server not mounted (mcp package missing)")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SCHEMAS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ChatRequest(BaseModel):
    message: str
    context: Optional[dict] = None
    force_ego: bool = False
    ghost_veil: bool = True

class ChatResponse(BaseModel):
    response: str
    mode: str
    sage_score: float
    sage_verdict: str
    latency_ms: float
    cycle_id: int
    backend_used: str

class KAIROSCycleRequest(BaseModel):
    proposal: str
    verdict: str
    score: float

class LiquidityDiscoverRequest(BaseModel):
    size_usd: float = 250000.0
    risk_tolerance: float = 0.30
    assets: str = "USDC->ETH"
    max_pools: int = 4
    high_volatility: bool = False
    target_pool: Optional[str] = None   # e.g. "curve_usdc_eth" for stress
    force_low_depth: bool = False

class LiquidityPolicyRequest(BaseModel):
    mutation_rate: Optional[float] = None
    selection_pressure: Optional[float] = None
    slippage_weight: Optional[float] = None
    diversification_bonus: Optional[float] = None
    risk_tolerance: Optional[float] = None

class RecallRequest(BaseModel):
    query: str
    room: Optional[str] = None
    top_k: int = 5

class MemoryStoreRequest(BaseModel):
    content: str
    type: Optional[str] = None
    source: Optional[str] = None
    domain: Optional[str] = None
    confidence: Optional[float] = None
    tags: list = []

class BroadcastRequest(BaseModel):
    content: str
    src: str = "API"
    channel: str = "#broadcast"
    msg_type: str = "chat"

class KillSwitchRequest(BaseModel):
    mode: str
    key: str

class DelegateRequest(BaseModel):
    task: str
    agent: Optional[str] = None
    metadata: dict = {}

class TrainRequest(BaseModel):
    domain: str = "agent_systems"
    count: int = 5

class ReviewRequest(BaseModel):
    module: str
    source: str = ""

class PushRequest(BaseModel):
    files: dict    # {path: content}
    message: str = "ðŸ–¤ GH05T3 auto-push"
    branch: str = "main"

class GhostScriptRunRequest(BaseModel):
    src: str
    reply_timeout: float = 30.0

class GhostScriptFileRequest(BaseModel):
    path: str
    reply_timeout: float = 30.0

class SecretsRequest(BaseModel):
    anthropic_api_key: Optional[str] = None
    github_pat: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_ai_key: Optional[str] = None


def _peer_self_info() -> dict:
    """Return a stable identity block for the mesh dashboard."""
    label = os.environ.get("TAILSCALE_OWN_LABEL") or os.environ.get("INSTANCE_LABEL") or "GH05T3"
    role = os.environ.get("INSTANCE_ROLE", "peer")
    ip = os.environ.get("TAILSCALE_OWN_IP", "").strip()
    host = ip or "localhost"
    return {
        "label": label,
        "role": role,
        "url": f"http://{host}:{GATEWAY_PORT}",
        "ip": ip or None,
        "discovery": "tailscale",
    }


def _parse_msg_type(value: str) -> MsgType:
    """Map caller string (e.g. 'thought', 'chat') to SwarmBus MsgType."""
    try:
        return MsgType(value.lower())
    except ValueError:
        return MsgType.CHAT


def _mesh_contract() -> dict:
    return {
        "self": _peer_self_info(),
        "peers": peer_registry.peers,
        "mesh": {
            "discovery": {
                "mode": "tailscale",
                "refresh": "/peers/refresh",
                "ping": "/peers/ping",
            },
            "github_relay": {
                "push": "/github/mesh/push",
                "pull": "/github/mesh/pull",
                "sync": "/github/mesh/sync",
                "peers": "/github/mesh/peers",
            },
        },
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ORIGINAL ROUTES (preserved from v2)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/")
async def identity():
    return {
        "name":    "GH05T3",
        "version": "3.0.0",
        "owner":   "leerobber",
        "hardware":"TatorTot â€” Lenovo LOQ 15AHP10",
        "mesh":    {
            "primary":  "RTX 5050 â†’ vLLM/Qwen2.5-32B-AWQ :8001",
            "verifier": "Radeon 780M â†’ llama.cpp/ROCm :8002",
            "fallback": "Ryzen 7 CPU â†’ llama.cpp :8003",
        },
        "swarm":   [a for a in bus.agents.keys()],
        "kairos_cycles": kairos.stats["total_cycles"],
        "memory_shards": memory.stats()["total_shards"],
        "conv_log":      bus.log.stats["total"],
    }


@app.get("/health")
async def health():
    # Returns instantly â€” backend probes run in background and are cached
    any_online = any(v == "online" for v in _backend_cache.values())
    all_unknown = all(v == "unknown" for v in _backend_cache.values())
    status = "starting" if all_unknown else ("operational" if any_online else "degraded")
    online_backends = [name for name, state in _backend_cache.items() if state == "online"]
    return {
        "status": status,
        "backends": _backend_cache,
        "inference": {
            "online": online_backends,
            "any_online": any_online,
        },
        "swarm_agents": len(bus.agents),
        "ws_clients": bus.stats["ws_clients"],
        "uptime_s": round(time.time() - _boot_time, 1),
        "timestamp": int(time.time()),
    }


@app.get("/status")
async def full_status():
    return {
        "system":    "GH05T3 v3",
        "omega_loop":omega.stats,
        "kairos":    kairos.stats,
        "memory":    memory.stats(),
        "sage":      sage.stats,
        "ghost":     ghost.stats,
        "swarm":     bus.stats,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    trap = await ghost.process_input(req.message)
    if trap is not None:
        return ChatResponse(response=trap, mode="ghost", sage_score=0.0,
                            sage_verdict="TRAPPED", latency_ms=0.0,
                            cycle_id=omega.cycle_count, backend_used="ghost_protocol")

    state = await omega.run(req.message, req.context)

    # Publish to swarm bus for dashboard visibility
    await bus.emit(
        src="USER",
        content=req.message,
        channel="#omega",
        msg_type=MsgType.CHAT,
    )
    await bus.emit(
        src="OMEGA",
        content=state.response,
        channel="#omega",
        msg_type=MsgType.RESULT,
        mode=state.mode.value,
        sage_score=state.sage_score,
        sage_verdict=state.sage_verdict,
        cycle_id=state.cycle_id,
    )

    return ChatResponse(
        response=state.response,
        mode=state.mode.value,
        sage_score=state.sage_score,
        sage_verdict=state.sage_verdict,
        latency_ms=state.latency_ms,
        cycle_id=state.cycle_id,
        backend_used=state.backend_used,
    )


@app.post("/kairos/cycle")
async def record_kairos_cycle(req: KAIROSCycleRequest):
    cycle = kairos.record_cycle(proposal=req.proposal, verdict=req.verdict, score=req.score)
    content = KAIROS_EMIT_TMPL.substitute(id=cycle.id, score=req.score, verdict=req.verdict)
    await bus.emit(
        src="KAIROS",
        content=content,
        channel="#broadcast",
        msg_type=MsgType.KAIROS,
        cycle_id=cycle.id,
        score=req.score,
        verdict=req.verdict,
        is_elite=cycle.is_elite,
    )
    return cycle.to_dict()


@app.get("/kairos/elite")
async def get_elite_archive():
    return [c.to_dict() for c in kairos.elite_archive]


# Compatibility layer - Honcho + Termux + sovereign-core clients
# Every client points VITE_SOVEREIGN_GATEWAY_URL / sovereign_url at :8002.

class _InferenceReq(BaseModel):
    model: str = 'auto'
    prompt: str = ''
    messages: list = []
    options: dict = {}
    stream: bool = False
    timeout: int = 30
    prefer_backend: str = ''
    require_gpu: bool = False


@app.post('/inference')
async def inference_compat(req: _InferenceReq, request: Request):
    message = req.prompt
    if not message and req.messages:
        message = ' '.join(m.get('content', '') for m in req.messages if m.get('role') != 'system')
    if not message:
        raise HTTPException(status_code=400, detail='prompt or messages required')
    trap = await ghost.process_input(message)
    if trap is not None:
        return {'request_id': f'gw-{int(time.time()*1000)}', 'model': 'ghost_protocol',
                'backend_id': 'ghost', 'backend_label': 'Ghost Protocol', 'response': trap,
                'done': True, 'prompt_eval_count': 0, 'eval_count': 0,
                'latency_ms': 0.0, 'routed_at': time.time()}
    t0 = time.time()
    state = await omega.run(message, {})
    return {'request_id': f'gw-{int(t0*1000)}', 'model': state.backend_used or 'avery',
            'backend_id': state.backend_used or 'omega', 'backend_label': state.backend_used or 'Avery',
            'response': state.response, 'done': True, 'prompt_eval_count': 0, 'eval_count': 0,
            'latency_ms': state.latency_ms, 'routed_at': t0}


class _SAGEReq(BaseModel):
    task: str
    context: dict = {}
    max_cycles: int = 1
    require_verification: bool = False


@app.post('/kairos/sage')
async def kairos_sage(req: _SAGEReq):
    t0 = time.time()
    state = await omega.run(req.task, req.context)
    cycle = kairos.record_cycle(proposal=state.response,
                                verdict=state.sage_verdict or 'PASS',
                                score=state.sage_score or 0.5)
    return {'agent_id': f'avery-{cycle.id}',
            'generation': kairos.stats.get('total_cycles', 0),
            'score': state.sage_score or 0.5,
            'verification_verdict': state.sage_verdict or 'PASS',
            'proposals': [state.response],
            'latency_ms': (time.time() - t0) * 1000,
            'sage_log': [cycle.to_dict()]}


class _EvolveReq(BaseModel):
    cycles: int = 1
    agent_id: str = ''


@app.post('/kairos/evolve')
async def kairos_evolve(req: _EvolveReq):
    results = []
    for _ in range(max(1, min(req.cycles, 10))):
        t0 = time.time()
        state = await omega.run('KAIROS: run autonomous evolution cycle', {})
        cycle = kairos.record_cycle(proposal=state.response,
                                    verdict=state.sage_verdict or 'PASS',
                                    score=state.sage_score or 0.5)
        results.append({'agent_id': req.agent_id or f'avery-{cycle.id}',
                         'generation': kairos.stats.get('total_cycles', 0),
                         'score': state.sage_score or 0.5,
                         'verification_verdict': state.sage_verdict or 'PASS',
                         'elite_promoted': cycle.is_elite,
                         'latency_ms': (time.time() - t0) * 1000,
                         'arso_cycles': 1})
    return {'results': results, 'total_latency_ms': sum(r['latency_ms'] for r in results)}


@app.get('/kairos/leaderboard')
async def kairos_leaderboard(limit: int = 10):
    elite = sorted(kairos.elite_archive, key=lambda c: c.score, reverse=True)[:limit]
    return {'agents': [c.to_dict() for c in elite], 'total': len(kairos.elite_archive)}


@app.get('/status/backends')
async def status_backends():
    return {name: {'id': name, 'label': name.replace('_', ' ').title(),
                   'url': f'http://localhost:{8010 if "rtx" in name.lower() or "gpu" in name.lower() else 8011}',
                   'status': 'healthy' if state == 'online' else 'unhealthy',
                   'latency_ms': None,
                   'device_type': 'nvidia_gpu' if 'rtx' in name.lower() else 'cpu',
                   'vram_gib': 8 if 'rtx' in name.lower() else 0}
            for name, state in _backend_cache.items()}


@app.get('/status/kairos/{agent_id}')
async def status_kairos_agent(agent_id: str):
    for cycle in kairos.elite_archive:
        if str(cycle.id) in agent_id or agent_id in str(cycle.id):
            return cycle.to_dict()
    return {'agent_id': agent_id, 'status': 'not_found', 'kairos': kairos.stats}


class _AuctionBidReq(BaseModel):
    resource_type: str = 'inference'
    votes: int = 1
    agent_id: str = 'honcho'


@app.post('/auction/bid')
async def auction_bid(req: _AuctionBidReq):
    online = [name for name, state in _backend_cache.items() if state == 'online']
    winner = online[0] if online else 'avery'
    return {'winner': winner, 'resource_type': req.resource_type,
            'votes_cast': req.votes, 'agent_id': req.agent_id, 'ts': time.time()}


@app.get('/ledger/tail')
async def ledger_tail(n: int = 20):
    cycles = list(kairos.elite_archive)[-n:]
    entries = [{'entry_id': str(c.id), 'operation_type': 'kairos_cycle',
                'backend_id': 'avery', 'trust_score': c.score,
                'timestamp': time.time(),
                'integrity_ok': c.verdict in ('PASS', 'PARTIAL')}
               for c in reversed(cycles)]
    return {'entries': entries, 'total': len(kairos.elite_archive)}


class _BenchmarkReq(BaseModel):
    model_id: str = 'default'


@app.post('/benchmark/run')
async def benchmark_run(req: _BenchmarkReq):
    t0 = time.time()
    state = await omega.run('Benchmark: respond with OK', {})
    latency = (time.time() - t0) * 1000
    return {'model_id': req.model_id, 'backend_id': state.backend_used or 'avery',
            'latency_ms': latency, 'throughput_tps': round(50 / max(latency / 1000, 0.001), 1),
            'passed': latency < 30000, 'ts': t0}


@app.websocket('/ws/events')
async def ws_events(ws: WebSocket):
    """/ws/events alias for sovereign-core compatible clients."""
    await ws.accept()
    q = bus.add_ws_client()
    await ws.send_text(json.dumps({'type': 'connected', 'ts': time.time(),
                                   'data': {'node': 'GH05T3', 'version': '3.0.0'}}))
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=25.0)
                try:
                    inner = json.loads(payload)
                    await ws.send_text(json.dumps({'type': inner.get('type', 'swarm.message'),
                                                   'ts': inner.get('ts', time.time()),
                                                   'data': inner}))
                except Exception:
                    await ws.send_text(payload)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({'type': 'pong', 'ts': time.time(), 'data': {}}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        bus.remove_ws_client(q)




# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AETHYRO DEFI LIQUIDITY ROUTING â€” DISCOVERY GATE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_liq = None  # liquidity router disabled

@app.get("/aethyro/liquidity/baseline")
async def liq_baseline():
    return _liq.baseline_report()

@app.get("/aethyro/liquidity/heatmap")
async def liq_heatmap(size_usd: float = 250000.0):
    return _liq.get_heatmap(size_usd)

@app.post("/aethyro/liquidity/discover")
async def liq_discover(req: LiquidityDiscoverRequest, shadow: bool = False):
    """Core Discovery Gate. Evolutionary search over routing allocations. Records elite to KAIROS.
    Supports high_volatility + target_pool for $1M curve stress tests.
    """
    constraints = {
        "size_usd": req.size_usd,
        "risk_tolerance": req.risk_tolerance,
        "assets": req.assets,
        "max_pools": req.max_pools,
        "high_volatility": req.high_volatility,
        "target_pool": req.target_pool,
        "force_low_depth": req.force_low_depth,
    }
    result = _liq.discover(constraints, shadow=shadow)
    # Use pre-compiled style for perf (template sub if needed; here formatted for latency window)
    content = DISCOVERY_EMIT_TMPL.substitute(
        slip=result.get('evolved_slippage', 0),
        improve=result.get('improvement_pct', 0),
        shadow=shadow,
        p95=result.get('p95_eval_ms', 0),
        expert=result.get('requires_expert', False)
    )
    await bus.emit(
        src="AETHYRO-LIQ",
        content=content,
        channel="#kairos",
        msg_type=MsgType.KAIROS,
        score=result["strategy"]["score"],
    )
    return result

@app.post("/aethyro/liquidity/policy")
async def liq_policy(req: LiquidityPolicyRequest):
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    _liq.set_policy(patch)
    return {"ok": True, "policy": _liq.get_policy()}

@app.get("/aethyro/liquidity/policy")
async def liq_get_policy():
    return _liq.get_policy()

@app.get("/aethyro/liquidity/lineage")
async def liq_lineage(strategy_id: str = ""):
    # Return recent for UI; full ancestor chain future
    if not strategy_id:
        if _liq.recent_strategies:
            return {"lineage": [s.to_dict() for s in _liq.recent_strategies[-5:]]}
        return {"lineage": []}
    return {"lineage": _liq.get_lineage(strategy_id)}

@app.get("/aethyro/liquidity/sessions")
async def liq_sessions(limit: int = 10):
    """Recent discovery sessions for history / reports."""
    recent = [s.to_dict() for s in _liq.recent_strategies[-limit:]][::-1]
    return {"sessions": recent, "count": len(recent)}

@app.post("/aethyro/liquidity/recalibrate")
async def liq_recalibrate(obs: Optional[dict] = None):
    """Market vs Model drift correction. Triggers KAIROS_RECALIBRATE intent."""
    result = _liq.recalibrate(obs)
    await bus.emit(
        src="AETHYRO-LIQ",
        content="RECALIBRATE drift {}->{} | {}".format(result.get('drift_before'), result.get('drift_after'), result.get('trigger')),
        channel="#kairos",
        msg_type=MsgType.KAIROS,
    )
    return result

@app.post("/aethyro/liquidity/purge")
async def liq_purge(keep: int = 50):
    """Optional MEMORY_PURGE for liquidity sessions before heavy launch."""
    res = _liq.purge_old_sessions(keep)
    return {"ok": True, "liquidity_state": res, "note": "Liquidity state is isolated from general SovereignCore / MemoryPalace shards."}

@app.post("/aethyro/liquidity/archive")
async def liq_archive():
    """Seal IP: Archive current Strategy Lineage + Delta Logic for licensing."""
    path = _liq.archive_ip()
    return {"archived": True, "path": path}

@app.get("/aethyro/liquidity/policy_pack")
async def liq_policy_pack():
    """Product: Return Flash-Crash Resistant high-vol policy pack (premium module)."""
    return _liq.get_flash_crash_policy_pack()

@app.post("/aethyro/liquidity/shadow_loop")
async def liq_shadow_loop(hours: int = 48, cycles: int = 96, size_usd: float = 2000000):
    """Deployment: Start simulated 48h Live Shadow Execution (PURE DRY-RUN/SHADOW mode).
    SovereignCore bridge receives monitoring data/logs/alerts at most, but executes NO trades or adjustments.
    No Soft-Mode (small-cap execution on breach) is active. See liquidity_routing.py for details.
    """
    summary = _liq.run_shadow_loop(hours=hours, cycles=cycles, size_usd=size_usd)
    content = SHADOW_EMIT_TMPL.substitute(max_d=summary.get("max_drift", 0), hard_stops=summary.get("hard_stops", 0))
    await bus.emit(src="AETHYRO-LIQ", content=content, channel="#kairos", msg_type=MsgType.KAIROS)
    return summary

@app.post("/aethyro/liquidity/analyze_stable")
async def liq_analyze_stable():
    """Benchmark: After shadow data, analyze Delta Logic for stable-env mutation_rate tightening."""
    # Use recent or force a run
    rep = _liq.discover({"size_usd": 2000000, "high_volatility": False}, shadow=True)
    analysis = _liq.analyze_delta_for_stable_env(rep.get("lineage"))
    return analysis

@app.get("/aethyro/status")
async def aethyro_status():
    """Quick status for Aethyro.com launch health (DeFi priority)."""
    base = _liq.baseline_report()
    pol = _liq.get_policy()
    return {
        "domain": "aethyro.com",
        "focus": "Algorithmic Liquidity Routing",
        "baseline": base,
        "policy": pol,
        "recent_discoveries": len(_liq.recent_strategies),
        "engine": "KAIROS + evolutionary testnet router",
    }


@app.get("/memory/stats")
async def get_memory_stats():
    return memory.stats()


@app.post("/memory/recall")
async def recall_memory(req: RecallRequest):
    results = await memory.recall(query=req.query, room=req.room, top_k=req.top_k)
    return {"results": results, "count": len(results)}


@app.post("/memory/store")
async def store_memory(req: MemoryStoreRequest):
    room = req.type or req.domain or "general"
    shard = memory.store(content=req.content, room=room, tags=req.tags)
    return shard


@app.post("/killswitch")
async def killswitch(req: KillSwitchRequest):
    try:
        mode = KillSwitchMode(req.mode.lower())
    except ValueError:
        raise HTTPException(400, "Invalid mode")
    result = ghost.killswitch.execute(mode=mode, key=req.key)
    if result.get("status") == "denied":
        raise HTTPException(403, "Authentication failed")
    return result


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SWARM ROUTES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/swarm/agents")
async def get_agents():
    return {"agents": bus.agents, "stats": swarm.stats if swarm else {}}


@app.post("/swarm/delegate")
async def delegate_task(req: DelegateRequest):
    if not swarm:
        raise HTTPException(503, "Swarm not initialized")
    target = await swarm.delegate(req.task, preferred_agent=req.agent)
    return {"ok": True, "task": req.task, "routed_to": target}


@app.post("/swarm/broadcast")
async def broadcast(req: BroadcastRequest):
    await bus.emit(
        src=req.src,
        content=req.content,
        channel=req.channel,
        msg_type=_parse_msg_type(req.msg_type),
    )
    return {"ok": True}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONVERSATION LOG ROUTES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/conversations")
async def get_conversations(n: int = 100, channel: str = None, src: str = None):
    return {
        "messages": bus.log.recent(n=n, channel=channel, src=src),
        "stats":    bus.log.stats,
    }


@app.get("/conversations/search")
async def search_conversations(q: str, limit: int = 50):
    return {"results": bus.log.search(q, limit=limit)}


@app.get("/conversations/stats")
async def conv_stats():
    return bus.log.stats


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CLAUDE INTEGRATION ROUTES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/claude/train")
async def claude_train(req: TrainRequest):
    if not claude:
        raise HTTPException(503, "Claude not initialized")
    scenarios = await claude.trainer.generate_training_batch(req.domain, req.count)
    return {"scenarios": scenarios, "count": len(scenarios), "domain": req.domain}


@app.post("/claude/review")
async def claude_review(req: ReviewRequest):
    if not claude:
        raise HTTPException(503, "Claude not initialized")
    review = await claude.architect.review_module(req.module, req.source)
    return {"review": review, "module": req.module}


@app.post("/claude/upgrade")
async def claude_upgrade(topic: str):
    if not claude:
        raise HTTPException(503, "Claude not initialized")
    proposal = await claude.architect.propose_upgrade(topic)
    return {"proposal": proposal, "topic": topic}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GITHUB ROUTES
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/github/status")
async def github_status():
    if not github:
        raise HTTPException(503, "GitHub not initialized")
    try:
        info = await github._gh.repo_info()
        return info
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/github/push")
async def github_push(req: PushRequest):
    if not github:
        raise HTTPException(503, "GitHub not initialized")
    try:
        url = await github._gh.push_files(req.files, req.message, req.branch)
        return {"ok": True, "commit_url": url, "files": len(req.files)}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/github/sync-memory")
async def github_sync_memory():
    await bus.emit(
        src="API",
        content="sync memory to github",
        channel="#github",
        msg_type=MsgType.TASK,
    )
    return {"ok": True, "note": "Sync task delegated to GITHUB agent"}


@app.post("/github/mesh/push")
async def github_mesh_push():
    """Push this instance's memory to GitHub mesh/sync/<label>/ so peers can pull it."""
    gh_agent = next((a for a in bus._agents.values() if a.name == "GITHUB"), None)
    if not gh_agent:
        return {"ok": False, "error": "GITHUB agent not running"}
    result = await gh_agent.push_relay()
    return result


@app.post("/github/mesh/pull")
async def github_mesh_pull():
    """Pull peer memory from GitHub mesh/sync/ and merge into local data."""
    gh_agent = next((a for a in bus._agents.values() if a.name == "GITHUB"), None)
    if not gh_agent:
        return {"ok": False, "error": "GITHUB agent not running"}
    result = await gh_agent.pull_relay()
    return result


@app.post("/github/mesh/sync")
async def github_mesh_sync():
    """Push then pull â€” full bidirectional sync via GitHub relay."""
    gh_agent = next((a for a in bus._agents.values() if a.name == "GITHUB"), None)
    if not gh_agent:
        return {"ok": False, "error": "GITHUB agent not running"}
    push_result = await gh_agent.push_relay()
    pull_result = await gh_agent.pull_relay()
    return {"push": push_result, "pull": pull_result}


@app.get("/github/mesh/peers")
async def github_mesh_peers():
    """List all instances that have pushed to the GitHub mesh relay."""
    gh_agent = next((a for a in bus._agents.values() if a.name == "GITHUB"), None)
    if not gh_agent:
        return {"peers": [], "error": "GITHUB agent not running"}
    branch = os.environ.get("GITHUB_BRANCH", "main")
    try:
        r = await gh_agent._gh._http.get(
            f"/repos/{gh_agent._gh._repo}/contents/mesh/sync",
            params={"ref": branch}
        )
        if r.status_code != 200:
            return {"peers": [], "note": "mesh/sync not yet initialised"}
        import base64 as _b64, json as _json
        peers = []
        for item in r.json():
            if item["type"] != "dir":
                continue
            manifest = await gh_agent._gh.get_file(f"mesh/sync/{item['name']}/manifest.json", branch)
            if manifest:
                data = _json.loads(_b64.b64decode(manifest["content"]).decode())
                peers.append(data)
            else:
                peers.append({"label": item["name"]})
        return {"peers": peers, "count": len(peers)}
    except Exception as e:
        return {"peers": [], "error": str(e)}


@app.get("/peers")
async def get_peers():
    """Return the canonical mesh contract for the dashboard."""
    data = _mesh_contract()
    data["count"] = len(data["peers"])
    data["tailscale_configured"] = bool(os.environ.get("TAILSCALE_API_KEY", ""))
    return data


@app.get("/peers/me")
async def get_peer_me():
    return _peer_self_info()


@app.post("/peers/refresh")
async def refresh_peers():
    """Trigger immediate Tailscale peer re-discovery."""
    peers = await peer_registry.refresh()
    data = _mesh_contract()
    data["peers"] = peers
    data["count"] = len(peers)
    return data


@app.post("/peers/ping")
async def ping_peers():
    """Compatibility alias for the dashboard's live refresh button."""
    return await refresh_peers()


@app.post("/peers/sync/push")
async def push_mesh_sync():
    """Compatibility alias for pushing the GitHub-backed mesh relay."""
    return await github_mesh_sync()


@app.get("/mcp/info")
async def mcp_info():
    """Show MCP server availability, mount URL, and Claude Code config snippet."""
    ts_ip  = os.environ.get("TAILSCALE_OWN_IP", "")
    token  = os.environ.get("GH05T3_API_TOKEN", "")
    host   = ts_ip or "localhost"
    port   = GATEWAY_PORT
    sse_url = f"http://{host}:{port}/mcp/sse"
    return {
        "available":  MCP_AVAILABLE,
        "sse_url":    sse_url,
        "auth_enabled": bool(token),
        "claude_code_config": {
            "mcpServers": {
                "gh05t3": {
                    "type": "sse",
                    "url":  sse_url,
                    **({"headers": {"Authorization": f"Bearer {token}"}} if token else {}),
                }
            }
        },
        "tools": [
            "chat", "get_status", "memory_recall", "memory_store",
            "swarm_delegate", "emit_to_bus", "get_conversations",
            "github_push", "github_status", "ghostscript_run",
            "list_peers", "refresh_peers", "peer_delegate", "peer_chat", "mesh_broadcast",
        ] if MCP_AVAILABLE else [],
    }


@app.post("/github/commit")
async def github_commit(message: str = "ðŸ–¤ GH05T3 manual commit"):
    await bus.emit(
        src="API",
        content=f"commit: {message}",
        channel="#github",
        msg_type=MsgType.TASK,
        commit_msg=message,
    )
    return {"ok": True, "message": message}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SETUP / SECRETS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

def _mask(val: str) -> str:
    """Return first 6 chars + asterisks so user can confirm without exposing key."""
    if not val:
        return ""
    return val[:6] + "****"

def _read_env() -> dict:
    pairs = {}
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    pairs[k.strip()] = v.strip()
    return pairs

def _write_env(pairs: dict):
    lines = []
    if os.path.exists(_ENV_PATH):
        with open(_ENV_PATH) as f:
            lines = f.readlines()
    updated = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in pairs:
                new_lines.append(f"{k}={pairs[k]}\n")
                updated.add(k)
                continue
        new_lines.append(line)
    for k, v in pairs.items():
        if k not in updated:
            new_lines.append(f"{k}={v}\n")
    with open(_ENV_PATH, "w") as f:
        f.writelines(new_lines)


@app.get("/setup/secrets/status")
async def secrets_status():
    env = _read_env()
    ak  = env.get("ANTHROPIC_API_KEY", "")
    gh  = env.get("GITHUB_PAT", "")
    gr  = env.get("GROQ_API_KEY", "")
    go  = env.get("GOOGLE_AI_KEY", "")
    return {
        "anthropic_api_key": {"set": bool(ak), "preview": _mask(ak)},
        "github_pat":        {"set": bool(gh), "preview": _mask(gh)},
        "groq_api_key":      {"set": bool(gr), "preview": _mask(gr)},
        "google_ai_key":     {"set": bool(go), "preview": _mask(go)},
        "env_path":          _ENV_PATH,
    }


@app.post("/setup/secrets")
async def save_secrets(req: SecretsRequest):
    pairs = {}
    if req.anthropic_api_key and req.anthropic_api_key.strip():
        val = req.anthropic_api_key.strip()
        pairs["ANTHROPIC_API_KEY"] = val
        os.environ["ANTHROPIC_API_KEY"] = val
        if claude:
            claude._client._key = val
    if req.github_pat and req.github_pat.strip():
        val = req.github_pat.strip()
        pairs["GITHUB_PAT"] = val
        os.environ["GITHUB_PAT"] = val
    if req.groq_api_key and req.groq_api_key.strip():
        val = req.groq_api_key.strip()
        pairs["GROQ_API_KEY"] = val
        os.environ["GROQ_API_KEY"] = val
    if req.google_ai_key and req.google_ai_key.strip():
        val = req.google_ai_key.strip()
        pairs["GOOGLE_AI_KEY"] = val
        os.environ["GOOGLE_AI_KEY"] = val

    if not pairs:
        raise HTTPException(400, "No values provided")

    _write_env(pairs)

    env = _read_env()
    await bus.emit(
        src="GATEWAY",
        content=f"ðŸ”‘ Secrets updated: {', '.join(pairs.keys())}",
        channel="#broadcast",
        msg_type=MsgType.SYSTEM,
    )
    return {
        "ok":      True,
        "updated": list(pairs.keys()),
        "anthropic_api_key": {"set": bool(env.get("ANTHROPIC_API_KEY")), "preview": _mask(env.get("ANTHROPIC_API_KEY", ""))},
        "github_pat":        {"set": bool(env.get("GITHUB_PAT")),        "preview": _mask(env.get("GITHUB_PAT", ""))},
        "groq_api_key":      {"set": bool(env.get("GROQ_API_KEY")),      "preview": _mask(env.get("GROQ_API_KEY", ""))},
        "google_ai_key":     {"set": bool(env.get("GOOGLE_AI_KEY")),     "preview": _mask(env.get("GOOGLE_AI_KEY", ""))},
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# GHOSTSCRIPT â€” AI PROGRAM RUNNER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from ghostscript import run_async as _gs_run_async, run_file_async as _gs_run_file_async
from ghost_llm import chat_once as _chat_once


async def _gs_llm_fn(prompt: str) -> str:
    """Adapter: wraps chat_once for GhostScript's single-arg llm_fn interface."""
    text, _ = await _chat_once(session="ghostscript", system="", user=prompt)
    return text


@app.post("/ghostscript/run")
async def ghostscript_run(req: GhostScriptRunRequest):
    """
    Execute a GhostScript program string.
    Returns the full execution trace and archive of proposals/emits.

    Example body:
      { "src": "let x = llm.chat('Hello') \\nprint(x)" }
    """
    result = await _gs_run_async(
        req.src,
        llm_fn=_gs_llm_fn,
        memory_engine=memory,
        agent_id="gateway-gs",
        reply_timeout=req.reply_timeout,
    )
    return result


@app.post("/ghostscript/run-file")
async def ghostscript_run_file(req: GhostScriptFileRequest):
    """
    Load and execute a .gs file from disk.
    Path is relative to the backend/ directory.

    Example body:
      { "path": "programs/sage_cycle.gs" }
    """
    import pathlib
    gs_path = pathlib.Path(__file__).parent / req.path
    if not gs_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    if gs_path.suffix not in (".gs", ".ghostscript", ".txt"):
        raise HTTPException(status_code=400, detail="Only .gs files allowed")
    result = await _gs_run_file_async(
        str(gs_path),
        llm_fn=_gs_llm_fn,
        memory_engine=memory,
        reply_timeout=req.reply_timeout,
    )
    return result


@app.get("/ghostscript/demos")
async def ghostscript_demos():
    """Return all built-in demo programs for testing in the dashboard."""
    from ghostscript import (
        DEMO_AGENT, DEMO_PIPELINE, DEMO_ASYNC,
        DEMO_IF_FOR, DEMO_MULTI_AGENT,
    )
    return {
        "demos": {
            "agent_sage":    {"name": "SAGE Agent Cycle",      "src": DEMO_AGENT},
            "pipeline":      {"name": "Pipeline Operator",     "src": DEMO_PIPELINE},
            "async_block":   {"name": "Async Parallel Calls",  "src": DEMO_ASYNC},
            "if_for":        {"name": "if/else + for Loop",    "src": DEMO_IF_FOR},
            "multi_agent":   {"name": "Multi-Agent Routing",   "src": DEMO_MULTI_AGENT},
        }
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AVERY / TEAM â€” persona & roster endpoints
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/avery")
async def avery_identity():
    """Return Avery's public persona â€” used by the frontend intro panel."""
    p = get_persona("AVERY")
    return {
        "name":   p.name,
        "title":  p.title,
        "avatar": p.avatar,
        "bio":    p.bio,
        "voice":  p.voice,
        "startup": {
            "name":    "Avery",
            "tagline": "Autonomous intelligence. Human outcomes.",
            "engine":  "GH05T3",
        },
    }


@app.get("/avery/team")
async def avery_team():
    """Return full agent team roster with humanized personas."""
    return {"team": team_roster()}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AVERY / STORY EDITOR â€” developmental editor mode
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class StoryEditorTurnRequest(BaseModel):
    session_id: str
    message:    str


async def _story_llm(system: str, messages: list) -> str:
    """Bridge story editor sessions to the active LLM provider."""
    # Build a single user turn from the full conversation history
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )
    try:
        from ghost_llm import chat_once
        text, _ = await chat_once(
            session="story_editor",
            system=system,
            user=history_text,
        )
        return text
    except Exception as e:
        log.error("Story editor LLM call failed: %s", e)
        return f"[Story editor LLM unavailable: {e}]"


@app.get("/avery/story/start/{session_id}")
async def story_editor_start(session_id: str):
    """Open a new story editor session and return the first intake question."""
    reset_session(session_id)
    greeting = story_editor_greeting()
    return {
        "session_id": session_id,
        "stage":      0,
        "reply":      greeting,
        "story":      {},
    }


@app.post("/avery/story/turn")
async def story_editor_turn_endpoint(req: StoryEditorTurnRequest):
    """Send one message to an active story editor session."""
    result = await story_editor_turn(
        session_id  = req.session_id,
        user_message= req.message,
        llm_call    = _story_llm,
    )
    return result


@app.get("/avery/story/sessions")
async def story_sessions():
    """List all active story editor sessions."""
    return {"sessions": list_sessions()}


@app.get("/avery/story/session/{session_id}")
async def story_session_state(session_id: str):
    """Return current state of a story editor session."""
    sess = get_session(session_id)
    return {
        "session_id": session_id,
        "stage":      sess["stage"],
        "story":      sess["story"],
        "turns":      len(sess["history"]),
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# LEMONADE â€” AMD Radeon 780M iGPU (STT Â· TTS Â· image gen)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/avery/speech/transcribe")
async def speech_transcribe(request: Request):
    """Transcribe audio to text via Whisper on Lemonade (780M iGPU).

    Send raw audio bytes as the request body.
    Content-Type: audio/wav  or  audio/mpeg
    Returns: {"text": "<transcription>"}
    """
    if not await _lemonade_ok():
        raise HTTPException(503, "Lemonade not available â€” install and start Lemonade server "
                                 "(https://github.com/lemonade-sdk/lemonade)")
    body = await request.body()
    if not body:
        raise HTTPException(400, "No audio data in request body")
    try:
        ct  = request.headers.get("content-type", "audio/wav")
        ext = "mp3" if "mp3" in ct or "mpeg" in ct else "wav"
        text = await _lemonade_transcribe(body, filename=f"audio.{ext}")
        return {"text": text}
    except Exception as e:
        log.error("[lemonade] transcribe failed: %s", e)
        raise HTTPException(500, f"Transcription failed: {e}")


class _TTSBody(BaseModel):
    text:  str
    voice: str = "af_heart"

@app.post("/avery/speech/synthesize")
async def speech_synthesize(body: _TTSBody):
    """Text-to-speech via Kokoro on Lemonade (780M iGPU).

    Body: {"text": "Hello, I'm Avery.", "voice": "af_heart"}
    Returns: audio/wav bytes
    """
    from fastapi.responses import Response
    if not await _lemonade_ok():
        raise HTTPException(503, "Lemonade not available")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "text is required")
    try:
        audio = await _lemonade_speak(text, voice=body.voice)
        return Response(content=audio, media_type="audio/wav")
    except Exception as e:
        log.error("[lemonade] TTS failed: %s", e)
        raise HTTPException(500, f"TTS failed: {e}")


class _ImageBody(BaseModel):
    prompt: str
    size:   str  = "512x512"
    model:  str | None = None

@app.post("/avery/image/generate")
async def image_generate(body: _ImageBody):
    """Generate an image via Stable Diffusion on Lemonade (780M iGPU).

    Body: {"prompt": "a cyberpunk ghost hacker", "size": "512x512"}
    Returns: {"image": "<base64 PNG or URL>", "prompt": "..."}
    """
    if not await _lemonade_ok():
        raise HTTPException(503, "Lemonade not available")
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(400, "prompt is required")
    try:
        result = await _lemonade_generate_image(prompt, model=body.model, size=body.size)
        return {"image": result, "prompt": prompt, "size": body.size}
    except Exception as e:
        log.error("[lemonade] image gen failed: %s", e)
        raise HTTPException(500, f"Image generation failed: {e}")


@app.get("/avery/lemonade/status")
async def lemonade_status_endpoint():
    """Check Lemonade availability and loaded models."""
    return await _lemonade_status()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# STRIPE â€” billing & subscription webhooks
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    """
    Stripe webhook receiver.
    Point your Stripe Dashboard webhook at: https://your-domain/stripe/webhook
    Required env vars: STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET
    """
    body      = await request.body()
    sig       = request.headers.get("stripe-signature", "")

    if STRIPE_WEBHOOK_SECRET:
        if not verify_stripe_signature(body, sig, STRIPE_WEBHOOK_SECRET):
            raise HTTPException(status_code=400, detail="Invalid Stripe signature")

    try:
        event = json.loads(body)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    event_type = event.get("type", "")
    result     = process_stripe_event(event_type, event.get("data", {}))

    # Emit to swarm bus so Kai (NEXUS) and the team can react
    if result:
        await bus.emit(
            src     = "STRIPE",
            content = f"[STRIPE] {event_type} â€” {json.dumps(result)}",
            channel = "#nexus",
            msg_type= MsgType.TASK,
            dst     = "NEXUS",
            metadata= {"stripe_event": event_type, "result": result},
        )
        log.info("Stripe event processed: %s â†’ %s", event_type, result.get("action"))

    # Dispatch marketplace job for async agent processing
    try:
        await ingest_stripe_event(event_type, event)
    except Exception as e:
        log.error("[marketplace] stripe ingestion failed (event=%s): %s",
                  event_type, e, exc_info=True)

    return {"received": True, "event": event_type, "action": result.get("action") if result else "ignored"}


@app.get("/stripe/subscribers")
async def stripe_subscribers():
    """Current subscriber summary (counts only â€” no PII in this endpoint)."""
    return subscriber_count()


@app.get("/stripe/subscribers/all")
async def stripe_subscribers_all():
    """Full subscriber list â€” internal use only, protect this behind auth in prod."""
    return {"subscribers": all_subscribers()}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# MARKETPLACE â€” Agent Job Queue
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/marketplace/stats")
async def marketplace_stats():
    """Job queue statistics â€” pending / claimed / completed counts."""
    return JobQueue.instance().stats()


class ManualJobRequest(BaseModel):
    task:      str  = Field(..., max_length=2000)
    tags:      list = Field(default=[], max_length=10)
    reward:    int  = Field(default=20, ge=0, le=1000)
    posted_by: str  = Field(default="api", max_length=50)


@app.post("/marketplace/post")
async def marketplace_post_job(req: ManualJobRequest, request: Request):
    """Manually post a job to the agent marketplace. Requires X-API-Key header."""
    _require_marketplace_auth(request)
    job_id = await JobQueue.instance().post(
        task=req.task, tags=req.tags,
        reward=req.reward, posted_by=req.posted_by,
    )
    return {"job_id": job_id}


@app.get("/marketplace/economy")
async def marketplace_economy():
    """Local credit ledger snapshot â€” agent balances and totals."""
    try:
        from economy.ledger import ledger_stats
        return ledger_stats()
    except Exception as e:
        log.error("[economy] ledger_stats failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch ledger stats")


@app.post("/sentinel/cve-feed")
async def sentinel_cve_feed(request: Request):
    """Accept CVE records and dispatch SENTINEL jobs. Body: {"cves": [...]}
    Requires X-API-Key header matching MARKETPLACE_API_KEY env var.
    """
    _require_marketplace_auth(request)
    body = await request.json()
    cves = body.get("cves", [])
    if not cves:
        raise HTTPException(400, "No CVE records provided")
    job_ids = await ingest_cve_feed(cves)
    return {"jobs_posted": len(job_ids), "job_ids": job_ids}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# WEBSOCKET â€” LIVE SWARM STREAM
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    """
    Real-time swarm bus stream.
    Replays last 50 messages on connect, then streams live.
    """
    await ws.accept()
    q = bus.add_ws_client()

    # Send hello
    await ws.send_text(json.dumps({
        "type": "hello",
        "node": {
            "id":      "GH05T3-TATORTOT",
            "version": "3.0.0",
            "agents":  list(bus.agents.keys()),
        },
        "ts": time.time(),
    }))

    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=20.0)
                await ws.send_text(payload)
            except asyncio.TimeoutError:
                # Ping
                await ws.send_text(json.dumps({"type": "ping", "ts": time.time()}))
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        bus.remove_ws_client(q)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# INSTALL SCRIPTS â€” served directly for one-liner bootstrapping
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/install/android")
async def install_android():
    """
    Serve the Termux setup script so the phone can bootstrap without git auth.
    On phone: pkg install wget -y && wget http://TATORTOT_IP:8002/install/android -O setup.sh && bash setup.sh
    """
    from fastapi.responses import PlainTextResponse
    script_path = Path(__file__).parent.parent / "native" / "android" / "termux_setup.sh"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail="Setup script not found")
    return PlainTextResponse(
        content=script_path.read_text(),
        media_type="text/x-sh",
        headers={"Content-Disposition": "attachment; filename=termux_setup.sh"},
    )


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# PROMETHEUS /metrics
# Falls back to JSON if prometheus_client not installed.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/metrics")
async def prometheus_metrics():
    """
    Expose Prometheus-compatible metrics.
    Scrape with: prometheus.yml â†’ scrape_configs â†’ targets: [localhost:8002]
    Falls back to JSON when prometheus_client is not installed.
    """
    try:
        from prometheus_client import (
            Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST,
            REGISTRY,
        )
        from starlette.responses import Response

        # Re-use or register metrics idempotently
        def _gauge(name, doc):
            try:
                return Gauge(name, doc)
            except ValueError:
                return REGISTRY._names_to_collectors.get(name)

        def _counter(name, doc):
            try:
                return Counter(name, doc)
            except ValueError:
                return REGISTRY._names_to_collectors.get(name)

        g_total    = _gauge("gh05t3_kairos_cycles_total",  "Total KAIROS cycles")
        g_elite    = _gauge("gh05t3_kairos_elite_total",   "Total elite KAIROS cycles")
        g_avg      = _gauge("gh05t3_kairos_avg_score",     "Average KAIROS score")
        g_threats  = _gauge("gh05t3_sentinel_threats",     "SENTINEL threats detected")
        g_shards   = _gauge("gh05t3_memory_shards",        "Memory Palace shard count")
        g_clients  = _gauge("gh05t3_ws_clients",           "Active WebSocket clients")
        g_agents   = _gauge("gh05t3_swarm_agents",         "Active swarm agents")

        ks = kairos.stats
        if g_total:  g_total.set(ks["total_cycles"])
        if g_elite:  g_elite.set(ks["elite_cycles"])
        if g_avg:    g_avg.set(ks["avg_score"])
        if g_shards: g_shards.set(memory.stats()["total_shards"])
        if g_clients: g_clients.set(len(bus._ws_clients))
        if g_agents:  g_agents.set(len(bus.agents))

        if swarm and g_threats:
            sentinel = swarm.sentinel if hasattr(swarm, "sentinel") else None
            if sentinel:
                g_threats.set(sentinel.stats.get("threats", 0))

        # OSS / MVS metrics (cycle duration, marketplace failures) share REGISTRY
        try:
            import oss.observability.metrics  # noqa: F401 â€” register collectors
        except Exception:
            pass

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    except ImportError:
        # prometheus_client not installed â€” return JSON
        ks = kairos.stats
        return {
            "gh05t3_kairos_cycles_total": ks["total_cycles"],
            "gh05t3_kairos_elite_total":  ks["elite_cycles"],
            "gh05t3_kairos_avg_score":    ks["avg_score"],
            "gh05t3_memory_shards":       memory.stats()["total_shards"],
            "gh05t3_ws_clients":          len(bus._ws_clients),
            "gh05t3_swarm_agents":        len(bus.agents),
        }


@app.get("/status/integrations")
async def integrations_status():
    """Show which optional integrations are active."""
    from integrations.notifier      import notifier_status
    from integrations.wandb_logger  import wandb_status
    from integrations.jira_sentinel import jira_status
    from memory.memory_palace       import _qdrant_ok
    return {
        "notifier": notifier_status(),
        "wandb":    wandb_status(),
        "jira":     jira_status(),
        "qdrant":   _qdrant_ok,
    }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# SOVEREIGN RECALL / CHRONICLE ENDPOINTS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_recall_instance = None

def _get_recall():
    global _recall_instance
    if _recall_instance is None:
        try:
            from sovereign_recall import SovereignRecall
            _recall_instance = SovereignRecall()
        except Exception:
            pass
    return _recall_instance


@app.get("/chronicle/status")
async def chronicle_status():
    """Sovereign Recall status â€” examples, tokens, source breakdown."""
    r = _get_recall()
    if r is None:
        return {"agent_id": "CHRONICLE", "status": "offline", "total_examples": 0}
    s = r.status()
    # add source breakdown from the output file
    sources: dict = {}
    try:
        from pathlib import Path
        import json as _json
        recall_file = Path(s["output_file"])
        if recall_file.exists():
            for line in recall_file.open():
                try:
                    rec = _json.loads(line)
                    src = rec.get("source", "unknown")
                    sources[src] = sources.get(src, 0) + 1
                except Exception:
                    pass
    except Exception:
        pass
    return {**s, "sources": sources, "status": "online"}


@app.post("/chronicle/scan")
async def chronicle_scan():
    """Trigger an immediate Sovereign Recall scan."""
    r = _get_recall()
    if r is None:
        raise HTTPException(status_code=503, detail="Sovereign Recall not initialized")
    stats = await r.scan_once()
    return {"ok": True, "stats": stats}


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# ENTRYPOINT
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.gateway_v3:app",
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        log_level="info",
        reload=False,
    )
