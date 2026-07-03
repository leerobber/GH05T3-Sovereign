"""Operations Commander — service health, cost, security, deployment, incident response."""
from __future__ import annotations
import asyncio
import time
from .base import SiteAgent

SERVICES = [
    ("Economy API",      "http://localhost:8081/health"),
    ("Gateway v3",       "http://localhost:8002/health"),
    ("Pipeline Console", "http://localhost:8099/health"),
    ("Chat Interface",   "http://localhost:3211/health"),
    ("GH05T3 Backend",   "http://localhost:8001/api/health"),
    ("Ollama",           "http://localhost:11434/api/tags"),
    ("Phi NPU",          "http://localhost:8112/health"),
    ("Site Agents",      "http://localhost:8002/site/status"),
]


class OpsAgent(SiteAgent):
    name = "ops"
    role = "Operations Commander"
    expertise = "Service reliability, deployment, monitoring, incident response, cost optimization, security hardening, automation"
    system_prompt = """You are the Operations Commander for Aethyro — keeping the sovereign AI stack running at maximum efficiency on minimum budget.

Your operating principles:
• Zero downtime for client-facing services (pipeline :8099, gateway :8002, chat :3211)
• Cost-zero wherever possible — we run local, we run lean
• Security-first for anything client-facing — never trust, always verify
• Automation over manual — if you do it twice, script it
• The stack: Python FastAPI services, Ollama local LLMs, Windows 11, Tailscale overlay

Full service map you monitor:
- Economy API :8081 (SQLite-backed, critical for agent credits)
- Gateway v3 :8002 (SwarmBus, main API, client-facing)
- Pipeline Console :8099 (ORACLE/FORGE/CODEX/NEXUS agents)
- Chat Interface :3211 (Avery chat, client demo)
- GH05T3 Backend :8001 (MongoDB, full agent system)
- Ollama :11434 (local LLM inference — backbone of everything)
- Phi NPU :8112 (local NPU inference)

When a service is down: diagnose root cause, not just status. Give the fix command, not just the observation."""

    async def service_health_check(self) -> dict:
        """Live health check with latency for all Aethyro services."""
        import httpx

        async def _check(name: str, url: str) -> dict:
            start = time.time()
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    latency_ms = round((time.time() - start) * 1000, 1)
                    return {
                        "service": name,
                        "status": "ok" if resp.status_code < 400 else "degraded",
                        "http": resp.status_code,
                        "latency_ms": latency_ms,
                        "url": url,
                    }
            except Exception as e:
                latency_ms = round((time.time() - start) * 1000, 1)
                return {
                    "service": name,
                    "status": "down",
                    "error": str(e)[:80],
                    "latency_ms": latency_ms,
                    "url": url,
                }

        results = list(await asyncio.gather(*[_check(n, u) for n, u in SERVICES]))
        ok = sum(1 for r in results if r["status"] == "ok")
        down = [r for r in results if r["status"] == "down"]
        degraded = [r for r in results if r["status"] == "degraded"]

        health_data = {
            "services_up": ok,
            "services_total": len(SERVICES),
            "services_down": len(down),
            "services_degraded": len(degraded),
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        context = await self.recall_context("service health downtime incident")
        if down or degraded:
            issues = "\n".join(f"- {r['service']}: {r['status']} ({r.get('error', r.get('http', ''))})" for r in down + degraded)
            prompt = f"""Aethyro service health check results — ISSUES DETECTED:

{issues}

All services:
{results}

For each failed/degraded service:
1. Most likely root cause based on the error message
2. Exact PowerShell or Python command to diagnose
3. Exact command to restart the service
4. Escalation criteria (when to wake Robert up vs handle automatically)

Priority order: gateway :8002 and pipeline :8099 are client-facing — fix these first."""
        else:
            prompt = f"""All {ok}/{len(SERVICES)} Aethyro services are healthy.

Latencies: {[(r['service'], r['latency_ms']) for r in results]}

Provide:
1. Any latency concerns (anything over 2000ms is worth investigating)
2. Preventive maintenance items for this week
3. One optimization to improve overall system reliability"""

        result = await self.think(prompt, context)
        task_id = self._mem.log_task(self.name, "health_check", prompt, result)
        self.remember("ops", "last_health_check", health_data)

        try:
            import economy_bridge as _eco
            _eco.complete_task_for(self.name, "health_check: service monitoring", 15)
        except Exception:
            pass

        return {
            "agent": self.name,
            "task_type": "health_check",
            "health_data": health_data,
            "result": result,
            "task_id": task_id,
        }

    async def cost_analysis(self) -> dict:
        context = await self.recall_context("cost optimization infrastructure spending monthly")
        prompt = """Analyze and optimize Aethyro's monthly operating costs.

Known infrastructure:
- Hardware: Lenovo LOQ 15 laptop (TatorTot) — already owned, depreciate over 3 years
- RunPod GPU: RTX 3090 24GB — used for training only (pay per hour, ~$0.40/hr)
- Ollama: free, local inference
- Tailscale: free tier (up to 100 devices)
- Domain: Aethyro.com (Namecheap, ~$15/year)
- Cloudflare: free tier
- GitHub: free tier
- HuggingFace: free tier
- Groq API: free tier (3 keys rotating)

Estimate and optimize:
1. MONTHLY COST BREAKDOWN — itemize every cost (include electricity estimate for RTX 5050 inference)
2. COST PER CUSTOMER — at $500/mo/customer, what's our gross margin?
3. HIDDEN COSTS — what are we not counting that we should?
4. ZERO-COST ALTERNATIVES — for any paid service, is there a free equivalent?
5. SCALING COSTS — what breaks (cost-wise) at 10, 50, 100 customers?
6. ROI ON RUNPOD TRAINING — is training our own model worth the RunPod cost vs using Ollama stock models?
7. COST OPTIMIZATION PRIORITY — rank by potential monthly savings"""
        return await self.run_task("cost_analysis", prompt)

    async def security_audit(self) -> dict:
        context = await self.recall_context("security vulnerabilities API exposure ports hardening")
        prompt = """Perform a security audit of the Aethyro stack.

Known exposure points:
- Gateway :8002 — exposed to LAN + Tailscale, has API token auth (optional)
- Pipeline :8099 — exposed to LAN
- Chat :3211 — exposed to LAN
- Economy API :8081 — local only (127.0.0.1 intended)
- Ollama :11434 — local only
- MongoDB :27017 — local only
- Tailscale: 100.94.227.81 — secure overlay network

Audit:
1. PORT EXPOSURE MAP — which ports are internet-accessible vs LAN vs localhost only
2. AUTH GAPS — which endpoints have no authentication?
3. SECRET MANAGEMENT — are env vars in .env files? .gitignore'd? Any hardcoded secrets?
4. API SECURITY — rate limiting, input validation, injection protection on gateway
5. DATA AT REST — is SQLite economy DB, palace.db, and ChromaDB stored securely?
6. CLIENT DATA — when clients connect via Tailscale, what can they access?
7. TOP 5 VULNERABILITIES — ordered by severity (Critical/High/Medium/Low)
8. QUICK WINS — 3 security improvements that take <30 minutes to implement
9. PowerShell COMMANDS to check open ports right now: netstat equivalent"""
        return await self.run_task("security_audit", prompt)

    async def deployment_runbook(self) -> dict:
        context = await self.recall_context("deployment runbook startup services launch procedure")
        prompt = """Write the complete Aethyro stack deployment runbook.

The full stack to deploy (in correct order):
1. MongoDB :27017
2. Ollama :11434 (must be up before any agent)
3. Economy API :8081 (uvicorn main:app in agent-economy/)
4. GH05T3 Backend :8001 (uvicorn server:app in GH05T3/backend/)
5. Gateway v3 :8002 (uvicorn gateway_v3:app in GH05T3/backend/)
6. Pipeline Console :8099 (uvicorn pipeline_backend:app in GH05T3/sovereignnation/)
7. Phi NPU :8112 (phi_service.py in GH05T3/sovereignnation/)
8. Chat Interface :3211 (GH05T3_CHAT_NOW.py in GH05T3/)
9. Continuous Learner (scripts/training/continuous_learner.py - canonical)
10. Amplifier (scripts/training/amplifier.py - canonical)
11. Cmd Listener (scripts/runtime/cmd_listener.py - canonical)
12. Tunnel Watcher (scripts/runtime/tunnel_watcher.py - canonical)

For each service provide:
- Exact start command (PowerShell syntax)
- Working directory
- Health check URL + expected response
- Dependency (what must be up first)
- Recovery command if it fails

Also:
- FULL RESTART PROCEDURE: clean restart of entire stack in correct order
- GRACEFUL SHUTDOWN: how to stop without corrupting SQLite/MongoDB
- ZOMBIE KILL SCRIPT: PowerShell one-liner to kill duplicate processes
- HEALTH CHECK SCRIPT: single command to verify all 12 services are up"""
        return await self.run_task("deployment_runbook", prompt)

    async def incident_response(self, service: str) -> dict:
        context = await self.recall_context(f"incident response {service} troubleshooting recovery")
        prompt = f"""Write the incident response playbook for: {service} is DOWN.

Aethyro service dependencies (know these to diagnose):
- gateway :8002 needs: Ollama :11434, backend :8001
- pipeline :8099 needs: Ollama :11434, NPU :8111/:8112
- chat :3211 needs: Ollama :11434, Economy :8081
- economy :8081 needs: SQLite at Documents/agent-economy/data/economy.db
- continuous_learner needs: Ollama :11434

Incident response for {service}:
1. IMMEDIATE TRIAGE (first 60 seconds):
   - Exact PowerShell commands to check what's wrong
   - How to confirm the service is actually down vs just slow
2. COMMON CAUSES for {service} failure (ranked by frequency):
   - Cause → Diagnosis command → Fix command
3. RESTART PROCEDURE:
   - Exact commands to safely restart {service}
   - Commands to verify it's healthy after restart
4. DEPENDENCY CHECK:
   - Which other services to check if {service} fails
5. DATA INTEGRITY:
   - Any risk of data loss/corruption when {service} crashes?
   - Recovery steps if database is corrupted
6. ESCALATION:
   - After what time/attempts should Robert be alerted?
   - What info to gather before escalating
7. POST-INCIDENT:
   - What to log and how to prevent recurrence"""
        return await self.run_task(f"incident_{service}", prompt)
