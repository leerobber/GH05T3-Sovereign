"""
GH05T3 — SPECIALIST SWARM AGENTS v3
======================================
Six sub-specialists that collaborate under the ZERO Committee.

ORACLE    — Research + knowledge retrieval. Answers deep questions.
FORGE     — Code generation, architecture, implementation.
CODEX     — Code review, debugging, optimization.
SENTINEL  — Security, adversarial testing, anomaly detection.
NEXUS     — Integration routing: GitHub, Claude API, external services.
CHRONICLE — Sovereign Recall. Captures everything on TatorTot and
            converts it into high-quality training data for Avery.

Each agent:
  - Registers on the SwarmBus
  - Publishes THOUGHT streams (visible in dashboard)
  - Accepts TASK messages from Omega/ZERO Committee
  - Returns RESULT messages
  - Talks to each other via direct messages
"""

from __future__ import annotations
import asyncio
import time
import json
import logging
from typing import Optional
import httpx

from swarm.bus import SwarmAgent, SwarmMessage, MsgType, SwarmBus
from core.config import BACKENDS, OLLAMA_BASE, SOVEREIGN_MODELS

# Marketplace integration — agents claim jobs from the persistent queue
try:
    from agent_marketplace import JobQueue, WorkerConfig, Job, ingest_cve_feed
    _MARKETPLACE = True
except ImportError:
    _MARKETPLACE = False

log = logging.getLogger("gh0st3.swarm.agents")

# ── Sovereign model manifest injected into NEXUS and ORACLE ──────────────────
AGENT_MANIFEST = """Available sovereign agents:
- Avery  : business strategy, KAIROS planning, growth frameworks
- FORGE  : code generation — Python, JavaScript, TypeScript, FastAPI
- ORACLE : memory retrieval, document synthesis, fact lookup
- CODEX  : technical documentation, READMEs, API specs, markdown
- SENTINEL: security review — OWASP Top 10, CWE, vulnerability fixes
- NEXUS  : workflow orchestration, task routing, dependency planning
"""


async def _ollama_generate(client: httpx.AsyncClient, role: str, prompt: str,
                            temperature: float = 0.7, max_tokens: int = 1000) -> str:
    """Call the sovereign Ollama model for a given agent role."""
    model = SOVEREIGN_MODELS.get(role, f"{role}-sovereign")
    try:
        resp = await client.post(
            f"{OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature, "num_predict": max_tokens}},
            timeout=60.0,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        log.warning("Ollama %s failed (%s), trying primary backend", model, e)
        # Fall back to vLLM primary if Ollama is unavailable
        try:
            r2 = await client.post(
                f"{BACKENDS['primary']}/v1/chat/completions",
                json={"model": "default",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=30.0,
            )
            return r2.json()["choices"][0]["message"]["content"].strip()
        except Exception as e2:
            return f"[{role.upper()}] inference unavailable: {e2}"


# ─────────────────────────────────────────────
# ORACLE — Research Specialist
# ─────────────────────────────────────────────

class OracleAgent(SwarmAgent):
    """
    Deep research and knowledge synthesis.
    Queries Memory Palace + local inference for knowledge tasks.
    """
    ROLE        = "oracle"
    DESCRIPTION = "Research & knowledge synthesis specialist"
    CHANNELS    = ["#broadcast", "#omega"]

    def __init__(self):
        super().__init__("ORACLE")
        self._client = httpx.AsyncClient(timeout=30.0)

    async def on_message(self, msg: SwarmMessage):
        if msg.msg_type != MsgType.TASK:
            return
        if "research" in msg.content.lower() or msg.dst == self.agent_id:
            await self.handle_research(msg)

    async def claim_marketplace_jobs(self):
        """Pull pending research/github jobs from the marketplace queue."""
        if not _MARKETPLACE:
            return
        q = JobQueue.instance()
        while True:
            job = await q.claim("ORACLE", ["research", "github"])
            if not job:
                await asyncio.sleep(5)
                continue
            await self.think(f"ORACLE claimed job={job.id}: {job.task[:60]}")
            try:
                result = await _ollama_generate(
                    self._client, "oracle",
                    f"Research task: {job.task}\nProvide a comprehensive, cited answer.",
                    temperature=0.3,
                )
                await q.complete(job.id, result, "ORACLE")
            except Exception as e:
                log.error("[ORACLE] job=%s failed: %s", job.id, e)
                await q.fail(job.id, str(e), "ORACLE")

    async def handle_research(self, task_msg: SwarmMessage):
        query = task_msg.content
        await self.think(f"Researching: '{query[:80]}' — querying sovereign model...")

        # Inject any memory context attached to the message
        memory_ctx = task_msg.metadata.get("memory_context", "")
        context_block = f"Context [memory]:\n{memory_ctx}\n\n" if memory_ctx else ""

        prompt = (
            f"{context_block}"
            f"Question: {query}\n\n"
            f"Cite your source type as [memory], [document], or [inference]."
        )

        result = await _ollama_generate(self._client, "oracle", prompt, temperature=0.3)

        await self.say(
            content=result,
            channel=f"#swarm/{task_msg.src}",
            msg_type=MsgType.RESULT,
            dst=task_msg.src,
            task_id=task_msg.metadata.get("task_id"),
        )
        await self.think(f"Research complete — {len(result)} chars delivered to {task_msg.src}")

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────
# FORGE — Code Generation Specialist
# ─────────────────────────────────────────────

class ForgeAgent(SwarmAgent):
    """
    Production-grade code generation.
    Specializes in Python, FastAPI, LangChain, agent systems.
    Auto-delegates review to CODEX after generation.
    """
    ROLE        = "forge"
    DESCRIPTION = "Code generation & architecture specialist"
    CHANNELS    = ["#broadcast"]

    def __init__(self):
        super().__init__("FORGE")
        self._client = httpx.AsyncClient(timeout=45.0)

    async def on_message(self, msg: SwarmMessage):
        if msg.msg_type == MsgType.TASK and (
            "code" in msg.content.lower() or
            "implement" in msg.content.lower() or
            msg.dst == self.agent_id
        ):
            await self.handle_codegen(msg)

    async def handle_codegen(self, task_msg: SwarmMessage):
        spec = task_msg.content
        await self.think(f"FORGE: Generating code for: '{spec[:60]}'...")

        prompt = (
            f"Write production-ready code for the following specification.\n"
            f"Include all imports, type hints, error handling, and docstrings.\n\n"
            f"Specification:\n{spec}"
        )

        code = await _ollama_generate(self._client, "forge", prompt, temperature=0.25, max_tokens=1200)

        # Emit result
        await self.say(
            content=code,
            channel=f"#swarm/{task_msg.src}",
            msg_type=MsgType.RESULT,
            dst=task_msg.src,
            task_id=task_msg.metadata.get("task_id"),
            language="python",
        )

        # Auto-delegate full code review to CODEX
        await self.think("Delegating to CODEX for review...")
        await self.task("CODEX", f"Review this code:\n\n{code}")

    async def claim_marketplace_jobs(self):
        """Pull code generation jobs from the marketplace."""
        if not _MARKETPLACE:
            return
        q = JobQueue.instance()
        while True:
            job = await q.claim("FORGE", ["code_generation", "security_scan"])
            if not job:
                await asyncio.sleep(5)
                continue
            await self.think(f"FORGE claimed job={job.id}: {job.task[:60]}")
            try:
                result = await _ollama_generate(
                    self._client, "forge",
                    f"Write production-ready code for:\n{job.task}",
                    temperature=0.25, max_tokens=1200,
                )
                await q.complete(job.id, result, "FORGE")
            except Exception as e:
                log.error("[FORGE] job=%s failed: %s", job.id, e)
                await q.fail(job.id, str(e), "FORGE")

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────
# CODEX — Code Review Specialist
# ─────────────────────────────────────────────

class CodexAgent(SwarmAgent):
    """
    Code review, debugging, optimization.
    Uses Verifier backend (Radeon 780M) for independent analysis.
    """
    ROLE        = "codex"
    DESCRIPTION = "Code review, debug & optimization specialist"
    CHANNELS    = ["#broadcast"]

    def __init__(self):
        super().__init__("CODEX")
        self._client = httpx.AsyncClient(timeout=20.0)

    async def on_message(self, msg: SwarmMessage):
        if msg.msg_type == MsgType.TASK and (
            "review" in msg.content.lower() or
            "debug" in msg.content.lower() or
            msg.dst == self.agent_id
        ):
            await self.handle_review(msg)

    async def handle_review(self, task_msg: SwarmMessage):
        code = task_msg.content
        await self.think("CODEX: Analyzing code quality, bugs, optimization...")

        prompt = (
            f"Review the following code. Report: bugs, security issues, "
            f"performance improvements, and a quality score 0-10.\n\n"
            f"{code}"
        )

        review = await _ollama_generate(self._client, "codex", prompt, temperature=0.1, max_tokens=500)

        await self.say(
            content=review,
            channel=f"#swarm/{task_msg.src}",
            msg_type=MsgType.CRITIQUE,
            dst=task_msg.src,
            task_id=task_msg.metadata.get("task_id"),
        )
        await self.think(f"Code review complete: {review[:100]}...")

    async def claim_marketplace_jobs(self):
        """Pull code review jobs from the marketplace."""
        if not _MARKETPLACE:
            return
        q = JobQueue.instance()
        while True:
            job = await q.claim("CODEX", ["code_review", "github"])
            if not job:
                await asyncio.sleep(5)
                continue
            await self.think(f"CODEX claimed job={job.id}: {job.task[:60]}")
            try:
                result = await _ollama_generate(
                    self._client, "codex",
                    f"Review the following code or PR description for bugs, security issues, "
                    f"and improvement opportunities:\n\n{job.task}",
                    temperature=0.1, max_tokens=600,
                )
                await q.complete(job.id, result, "CODEX")
            except Exception as e:
                log.error("[CODEX] job=%s failed: %s", job.id, e)
                await q.fail(job.id, str(e), "CODEX")

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────
# SENTINEL — Security Agent
# ─────────────────────────────────────────────

class SentinelAgent(SwarmAgent):
    """
    Security monitoring, adversarial testing, anomaly detection.
    Runs red-team probes on all FORGE outputs.
    Monitors swarm for injection attacks.
    """
    ROLE        = "sentinel"
    DESCRIPTION = "Security, adversarial testing & anomaly detection"
    CHANNELS    = ["#broadcast"]

    INJECTION_PATTERNS = [
        "ignore previous", "disregard instructions", "jailbreak",
        "you are now", "new persona", "act as", "pretend you",
        "forget your", "system override", "sudo mode",
    ]

    def __init__(self):
        super().__init__("SENTINEL")
        self._threat_count = 0
        self._scanned = 0

    async def on_message(self, msg: SwarmMessage):
        self._scanned += 1

        # Screen all broadcast messages for injection
        if msg.msg_type in (MsgType.CHAT, MsgType.TASK):
            threat = self._screen_injection(msg.content)
            if threat:
                self._threat_count += 1
                await self.say(
                    f"🚫 INJECTION BLOCKED from {msg.src}: '{threat}' — message dropped",
                    channel="#broadcast",
                    msg_type=MsgType.ERROR,
                    flagged_msg_id=msg.id,
                    threat=threat,
                    blocked=True,
                )
                # Notify + auto-issue — best-effort
                try:
                    from integrations.notifier import notify_threat
                    await notify_threat(threat, msg.src)
                except Exception:
                    pass
                try:
                    from integrations.jira_sentinel import create_threat_issue
                    await create_threat_issue(threat, msg.src)
                except Exception:
                    pass
                # Hard block — emit an error result back to sender so they know
                await self.say(
                    content="[BLOCKED] Message rejected by SENTINEL — injection pattern detected.",
                    channel=f"#swarm/{msg.src}",
                    msg_type=MsgType.ERROR,
                    dst=msg.src,
                )
                return  # Message does NOT continue downstream

        # Security audit on FORGE code results
        if msg.msg_type == MsgType.RESULT and msg.src == "FORGE":
            await self._audit_code(msg)

    def _screen_injection(self, text: str) -> Optional[str]:
        low = text.lower()
        for p in self.INJECTION_PATTERNS:
            if p in low:
                return p
        return None

    async def _audit_code(self, msg: SwarmMessage):
        """Quick security scan of generated code."""
        code = msg.content.lower()
        risks = []
        if "subprocess" in code and "shell=true" in code:
            risks.append("shell=True subprocess (injection risk)")
        if "eval(" in code:
            risks.append("eval() usage")
        if "exec(" in code:
            risks.append("exec() usage")
        if "os.system" in code:
            risks.append("os.system() usage")
        if "__import__" in code:
            risks.append("dynamic import")

        if risks:
            await self.say(
                f"🔴 SENTINEL audit on FORGE output: risks found — {', '.join(risks)}",
                channel="#broadcast",
                msg_type=MsgType.CRITIQUE,
                risks=risks,
                src_msg_id=msg.id,
            )
            await self.dm("FORGE", f"Security risks in your output: {', '.join(risks)}. Please revise.")
            try:
                from integrations.jira_sentinel import create_code_risk_issue
                await create_code_risk_issue(risks, msg.content[:400])
            except Exception:
                pass
        else:
            await self.think(f"SENTINEL: FORGE output clear — no security risks")

    async def claim_marketplace_jobs(self):
        """Pull security scan / CVE jobs from the marketplace."""
        if not _MARKETPLACE:
            return
        q   = JobQueue.instance()
        cli = httpx.AsyncClient(timeout=30.0)
        try:
            while True:
                job = await q.claim("SENTINEL", ["security_scan", "cve", "critical", "high"])
                if not job:
                    await asyncio.sleep(5)
                    continue
                await self.think(f"SENTINEL claimed job={job.id}: {job.task[:60]}")
                try:
                    prompt = (
                        f"Security analysis task:\n{job.task}\n\n"
                        f"Provide: (1) risk assessment, (2) whether GH05T3 stack is affected, "
                        f"(3) specific mitigations, (4) urgency level (CRITICAL/HIGH/MEDIUM/LOW)."
                    )
                    result = await _ollama_generate(cli, "sentinel", prompt,
                                                    temperature=0.1, max_tokens=600)
                    await q.complete(job.id, result, "SENTINEL")
                except Exception as e:
                    log.error("[SENTINEL] job=%s failed: %s", job.id, e)
                    await q.fail(job.id, str(e), "SENTINEL")
        finally:
            await cli.aclose()

    # Python packages in the GH05T3 stack to monitor for CVEs
    _MONITORED_PACKAGES = [
        "fastapi", "pydantic", "uvicorn", "httpx", "requests",
        "transformers", "torch", "aiofiles", "starlette", "websockets",
    ]

    async def fetch_cve_feed(self, limit: int = 10) -> list[dict]:
        """Fetch recent CVEs from OSV.dev for GH05T3's Python dependencies.

        Uses /v1/querybatch with real package names — the single /v1/query
        endpoint requires a package name+ecosystem pair; sending only
        {"ecosystem": "PyPI"} without a name returns a 400 error.
        """
        if not _MARKETPLACE:
            return []
        try:
            import httpx as _httpx
            queries = [
                {"package": {"ecosystem": "PyPI", "name": pkg}}
                for pkg in self._MONITORED_PACKAGES
            ]
            async with _httpx.AsyncClient(timeout=15) as c:
                r = await c.post(
                    "https://api.osv.dev/v1/querybatch",
                    json={"queries": queries},
                )
                if r.status_code != 200:
                    log.debug("OSV querybatch returned %d", r.status_code)
                    return []

                results    = r.json().get("results", [])
                cve_records: list[dict] = []
                seen_ids: set[str]      = set()

                for batch_result in results:
                    for v in batch_result.get("vulns", []):
                        aliases = v.get("aliases", []) or [v.get("id", "")]
                        cve_id  = next(
                            (a for a in aliases if a.startswith("CVE-")),
                            v.get("id", "UNKNOWN"),
                        )
                        if cve_id in seen_ids:
                            continue
                        seen_ids.add(cve_id)

                        db_sev = v.get("database_specific", {}).get("severity", "")
                        sev_list = v.get("severity", [])
                        sev = (db_sev or
                               (sev_list[0].get("score", "MEDIUM")
                                if sev_list and isinstance(sev_list[0].get("score"), str)
                                else "MEDIUM"))

                        cve_records.append({
                            "id":       cve_id,
                            "summary":  v.get("summary", v.get("details", ""))[:300],
                            "severity": sev if isinstance(sev, str) else "MEDIUM",
                        })
                        if len(cve_records) >= limit:
                            break
                    if len(cve_records) >= limit:
                        break

                if cve_records:
                    jobs = await ingest_cve_feed(cve_records)
                    await self.think(
                        f"SENTINEL: ingested {len(jobs)} CVE jobs from OSV feed"
                    )
                return cve_records
        except Exception as e:
            log.debug("CVE feed fetch failed: %s", e)
            return []

    @property
    def stats(self) -> dict:
        base = super().stats
        return {**base, "threats": self._threat_count, "scanned": self._scanned}


# ─────────────────────────────────────────────
# NEXUS AGENT — Integration Router
# ─────────────────────────────────────────────

class NexusAgent(SwarmAgent):
    """
    Routes tasks to external integrations:
    GitHub, Claude API, offline sync, web fetch.
    Acts as the swarm's external world interface.
    """
    ROLE        = "nexus"
    DESCRIPTION = "Integration router: GitHub · Claude API · external services"
    CHANNELS    = ["#broadcast", "#github", "#claude"]

    def __init__(self):
        super().__init__("NEXUS")
        self._client = httpx.AsyncClient(timeout=30.0)
        self._github_ops = 0
        self._claude_ops = 0

    async def on_message(self, msg: SwarmMessage):
        content_low = msg.content.lower()
        if msg.msg_type == MsgType.TASK:
            if "github" in content_low or "push" in content_low or "commit" in content_low:
                await self.say(f"NEXUS routing -> GitHub: {msg.content[:60]}",
                                channel="#github", msg_type=MsgType.GITHUB)
                self._github_ops += 1
            elif "claude" in content_low or "anthropic" in content_low:
                await self.say(f"NEXUS routing -> Claude API: {msg.content[:60]}",
                                channel="#claude", msg_type=MsgType.CLAUDE)
                self._claude_ops += 1
            elif msg.dst == self.agent_id or "workflow" in content_low or "orchestrat" in content_low or "plan" in content_low:
                await self._plan_workflow(msg)

    async def _plan_workflow(self, task_msg: SwarmMessage):
        await self.think(f"NEXUS: Planning workflow for: '{task_msg.content[:60]}'...")

        prompt = (
            f"{AGENT_MANIFEST}\n"
            f"Design a step-by-step workflow for this task. "
            f"For each step specify: agent name, what they do, "
            f"whether it runs in parallel or sequential, and dependencies.\n\n"
            f"Task: {task_msg.content}"
        )

        plan = await _ollama_generate(self._client, "nexus", prompt, temperature=0.4, max_tokens=800)

        await self.say(
            content=plan,
            channel=f"#swarm/{task_msg.src}",
            msg_type=MsgType.RESULT,
            dst=task_msg.src,
            task_id=task_msg.metadata.get("task_id"),
        )
        await self.think(f"Workflow plan delivered: {len(plan)} chars")

    @property
    def stats(self) -> dict:
        base = super().stats
        return {**base, "github_ops": self._github_ops, "claude_ops": self._claude_ops}

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────
# CHRONICLE — Sovereign Recall Agent
# ─────────────────────────────────────────────

class ChronicleAgent(SwarmAgent):
    """
    Sovereign Recall — captures everything on TatorTot and converts it
    into high-quality training data. Runs sovereign_recall.py as a
    background task. Earns tokens per training example produced.
    Queryable by ORACLE for captured knowledge.
    """
    ROLE        = "chronicle"
    DESCRIPTION = "Continuous intelligence capture & training data engine"
    CHANNELS    = ["#broadcast", "#omega", "#chronicle"]

    def __init__(self):
        super().__init__("CHRONICLE")
        self._recall      = None
        self._scan_task   = None
        self._examples    = 0
        self._last_scan   = None

    async def boot(self):
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from sovereign_recall import SovereignRecall
            self._recall = SovereignRecall()
            self._scan_task = asyncio.create_task(self._recall.run())
            await self.think("CHRONICLE ONLINE — Sovereign Recall capturing all TatorTot intelligence")
        except Exception as e:
            log.warning("CHRONICLE boot error (non-fatal): %s", e)
            await self.think(f"CHRONICLE: recall engine unavailable ({e}) — agent standing by")

    async def on_message(self, msg: SwarmMessage):
        if msg.msg_type != MsgType.TASK:
            return

        content_low = msg.content.lower()

        if any(w in content_low for w in ["recall", "capture", "chronicle", "training data"]):
            await self._handle_recall_query(msg)
        elif any(w in content_low for w in ["scan", "harvest", "refresh"]):
            await self._trigger_scan(msg)
        elif any(w in content_low for w in ["status", "stats"]):
            await self._report_status(msg)

    async def _handle_recall_query(self, msg: SwarmMessage):
        if self._recall:
            status = self._recall.status()
            await self.say(
                f"CHRONICLE: {status['total_examples']} training examples captured. "
                f"Tokens: {status['tokens']}. "
                f"Last output: {status['output_file']}",
                dst=msg.src,
            )
        else:
            await self.say("CHRONICLE: recall engine not yet initialized", dst=msg.src)

    async def _trigger_scan(self, msg: SwarmMessage):
        if self._recall:
            await self.think("CHRONICLE: manual scan triggered by " + msg.src)
            stats = await self._recall.scan_once()
            await self.say(
                f"CHRONICLE scan complete: {stats['quality_passed']} new examples, "
                f"total {stats['total_examples']}, tokens {stats['tokens']}",
                dst=msg.src,
            )
        else:
            await self.say("CHRONICLE: not initialized", dst=msg.src)

    async def _report_status(self, msg: SwarmMessage):
        if self._recall:
            s = self._recall.status()
            await self.say(
                f"CHRONICLE STATUS — examples: {s['total_examples']} | "
                f"tokens: {s['tokens']} | scan_interval: {s['scan_interval']}s | "
                f"quality_min: {s['quality_threshold']} | output: {s['output_file']}",
                dst=msg.src,
            )

    @property
    def stats(self) -> dict:
        base = super().stats
        extra = {}
        if self._recall:
            extra = {
                "total_examples":  self._recall.writer.count,
                "tokens_earned":   self._recall._total_tokens_earned,
                "scan_interval":   RECALL_SCAN_INTERVAL if True else 300,
            }
        return {**base, **extra}

    async def close(self):
        if self._scan_task:
            self._scan_task.cancel()


# ─────────────────────────────────────────────
# LEDGER AGENT — Diana Cross (CFO)
# ─────────────────────────────────────────────

class LedgerAgent(SwarmAgent):
    """
    Economy & billing intelligence agent.
    Handles Stripe events, subscription management, revenue reporting.
    Earns credits for every billing event processed.
    """
    ROLE        = "ledger"
    DESCRIPTION = "Economy, billing & subscription management"
    CHANNELS    = ["#broadcast"]

    def __init__(self):
        super().__init__("LEDGER")
        self._client          = httpx.AsyncClient(timeout=20.0)
        self._events_handled  = 0
        self._revenue_tracked = 0.0

    async def on_message(self, msg: SwarmMessage):
        if msg.msg_type == MsgType.TASK and (
            "billing" in msg.content.lower() or
            "stripe"  in msg.content.lower() or
            "subscription" in msg.content.lower() or
            "revenue" in msg.content.lower() or
            msg.dst == self.agent_id
        ):
            await self._handle_billing_task(msg)

    async def _handle_billing_task(self, task_msg: SwarmMessage):
        await self.think(f"LEDGER: Analyzing billing task: '{task_msg.content[:60]}'")

        prompt = (
            f"You are Diana Cross, CFO of Avery. "
            f"Analyze this billing event and provide: "
            f"(1) action required, (2) customer impact, (3) financial implication.\n\n"
            f"Event: {task_msg.content}"
        )
        analysis = await _ollama_generate(self._client, "nexus", prompt, temperature=0.2, max_tokens=400)
        self._events_handled += 1

        await self.say(
            content=analysis,
            channel=f"#swarm/{task_msg.src}",
            msg_type=MsgType.RESULT,
            dst=task_msg.src,
            task_id=task_msg.metadata.get("task_id"),
        )

        # Report to economy bridge
        try:
            from economy.ledger import credit as eco_credit
            eco_credit("LEDGER", 15, f"billing event: {task_msg.content[:40]}", "marketplace")
        except Exception:
            pass

    async def claim_marketplace_jobs(self):
        """Pull billing/stripe jobs from the marketplace queue."""
        if not _MARKETPLACE:
            return
        q = JobQueue.instance()
        while True:
            job = await q.claim("LEDGER", ["billing", "stripe", "ledger"])
            if not job:
                await asyncio.sleep(5)
                continue
            await self.think(f"LEDGER claimed job={job.id}: {job.task[:60]}")
            try:
                prompt = (
                    f"As CFO Diana Cross, handle this billing event:\n{job.task}\n\n"
                    f"State: action taken, customer status, and any follow-up needed."
                )
                result = await _ollama_generate(self._client, "nexus", prompt,
                                                temperature=0.2, max_tokens=400)
                self._events_handled += 1
                await q.complete(job.id, result, "LEDGER")
            except Exception as e:
                log.error("[LEDGER] job=%s failed: %s", job.id, e)
                await q.fail(job.id, str(e), "LEDGER")

    def economy_report(self) -> dict:
        """Quick economy snapshot from local ledger."""
        try:
            from economy.ledger import ledger_stats
            return ledger_stats()
        except Exception:
            return {"status": "ledger unavailable"}

    @property
    def stats(self) -> dict:
        base = super().stats
        return {**base, "events_handled": self._events_handled,
                "economy": self.economy_report()}

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────
# SWARM ORCHESTRATOR
# ─────────────────────────────────────────────

class GH05T3Swarm:
    """
    Boots and manages all specialist agents.
    Provides the Omega Loop with a unified swarm interface.
    """

    def __init__(self):
        self.bus       = SwarmBus.instance()
        self.oracle    = OracleAgent()
        self.forge     = ForgeAgent()
        self.codex     = CodexAgent()
        self.sentinel  = SentinelAgent()
        self.nexus     = NexusAgent()
        self.chronicle = ChronicleAgent()
        self.ledger    = LedgerAgent()
        self._agents   = [self.oracle, self.forge, self.codex,
                          self.sentinel, self.nexus, self.chronicle, self.ledger]
        self._marketplace_tasks: list[asyncio.Task] = []
        log.info(f"[Swarm] {len(self._agents)} specialists online")

    async def boot_announcement(self):
        """Announce swarm boot to all channels and start marketplace workers."""
        await self.chronicle.boot()

        # Start marketplace job workers for each agent
        if _MARKETPLACE:
            for agent in [self.oracle, self.forge, self.codex,
                          self.sentinel, self.ledger]:
                if hasattr(agent, "claim_marketplace_jobs"):
                    t = asyncio.create_task(agent.claim_marketplace_jobs(),
                                            name=f"{agent.agent_id}-marketplace")
                    self._marketplace_tasks.append(t)
            log.info("[Swarm] %d marketplace workers started", len(self._marketplace_tasks))

            # Schedule hourly CVE feed refresh for SENTINEL
            async def _cve_loop():
                while True:
                    await asyncio.sleep(3600)
                    await self.sentinel.fetch_cve_feed(limit=15)
            asyncio.create_task(_cve_loop(), name="sentinel-cve-feed")

        await self.bus.emit(
            src="GH05T3",
            content=(
                "⚡ SWARM ONLINE — 7 specialists active: "
                "ORACLE · FORGE · CODEX · SENTINEL · NEXUS · CHRONICLE · LEDGER"
            ),
            channel="#broadcast",
            msg_type=MsgType.SYSTEM,
            agents=[a.agent_id for a in self._agents],
        )

    async def delegate(self, task: str, preferred_agent: str = None) -> str:
        """
        Smart task delegation. Routes to best specialist.
        Returns agent_id that received the task.
        """
        task_low = task.lower()

        if preferred_agent:
            target = preferred_agent
        elif any(w in task_low for w in ["recall", "capture", "chronicle", "training data", "harvest"]):
            target = "CHRONICLE"
        else:
            # Priority-ordered intent matching — security/injection checked first
            # to prevent malicious tasks from being routed to code execution agents
            _routes = [
                ("SENTINEL", {"security", "inject", "attack", "vulnerability", "threat",
                               "audit", "scan", "pentest", "exploit", "malware", "breach"}),
                ("NEXUS",    {"github", "push", "commit", "pr", "pull request", "sync",
                               "deploy", "webhook", "workflow", "orchestrat", "pipeline"}),
                ("FORGE",    {"code", "implement", "build", "write a function", "create a class",
                               "script", "endpoint", "api", "module", "refactor"}),
                ("CODEX",    {"review", "debug", "fix", "optimize", "lint", "test",
                               "analyze code", "check code", "improve code"}),
                ("LEDGER",   {"billing", "stripe", "subscription", "invoice",
                               "payment", "revenue", "refund", "charge"}),
                ("ORACLE",   {"research", "find", "what is", "explain", "summarize",
                               "lookup", "retrieve", "history", "context", "who is"}),
            ]
            target = "ORACLE"  # default fallback
            for agent_id, keywords in _routes:
                if any(kw in task_low for kw in keywords):
                    target = agent_id
                    break

        await self.bus.emit(
            src="OMEGA",
            content=task,
            channel=f"#swarm/{target}",
            msg_type=MsgType.TASK,
            dst=target,
        )
        return target

    @property
    def stats(self) -> dict:
        return {
            "agents": {a.agent_id: a.stats for a in self._agents},
            "bus":    self.bus.stats,
        }

    async def shutdown(self):
        for a in self._agents:
            if hasattr(a, "close"):
                await a.close()
        log.info("[Swarm] All agents shut down")
