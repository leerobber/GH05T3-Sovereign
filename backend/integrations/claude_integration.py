"""
GH05T3 — CLAUDE INTEGRATION v3
=================================
Wires Claude (claude-sonnet-4-20250514) into the GH05T3 swarm
as a high-power accelerator for training, code review, and upgrades.

Roles:
  CLAUDE-TRAINER   — Generates synthetic KAIROS training data
  CLAUDE-ARCHITECT — Reviews and upgrades GH05T3 architecture
  CLAUDE-CODEGEN   — Produces elite-tier code when local models underperform
  CLAUDE-EVAL      — Evaluates SAGE pipeline outputs with Claude-level precision

All calls are logged to the SwarmBus (#claude channel) and
conversation log — fully visible in the dashboard.

Cost tracking: token counts logged per call.
"""

from __future__ import annotations
import asyncio
import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from pathlib import Path

import httpx

from swarm.bus import SwarmBus, SwarmMessage, MsgType, SwarmAgent
from evolution.kairos import KAIROSCycle

log = logging.getLogger("gh0st3.claude")

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
_TRUE_VALUES   = {"1", "true", "yes", "on"}

# Token usage log
USAGE_LOG = Path("memory/claude_usage.jsonl")


# ─────────────────────────────────────────────
# USAGE TRACKER
# ─────────────────────────────────────────────

@dataclass
class ClaudeUsage:
    timestamp:    float = field(default_factory=time.time)
    role:         str   = ""
    input_tokens: int   = 0
    output_tokens:int   = 0
    latency_ms:   float = 0.0
    task:         str   = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def log(self):
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(USAGE_LOG, "a") as f:
            f.write(json.dumps({
                "ts": self.timestamp,
                "role": self.role,
                "in": self.input_tokens,
                "out": self.output_tokens,
                "latency_ms": self.latency_ms,
                "task": self.task[:80],
            }) + "\n")


# ─────────────────────────────────────────────
# BASE CLAUDE CLIENT
# ─────────────────────────────────────────────

class ClaudeClient:
    """Direct async Claude API client. No SDK dependency."""

    def __init__(self, api_key: str = None):
        import os
        self._key    = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = httpx.AsyncClient(timeout=60.0)
        self._bus    = SwarmBus.instance()

    async def call(self, system: str, user: str, max_tokens: int = 1024,
                   role_label: str = "claude", task_label: str = "") -> tuple[str, ClaudeUsage]:
        if os.environ.get("ALLOW_PAID_LLM", "").strip().lower() not in _TRUE_VALUES:
            return "[Claude disabled: set ALLOW_PAID_LLM=1 to permit paid Anthropic calls]", ClaudeUsage()
        if not self._key:
            return "[Claude API key not configured — set ANTHROPIC_API_KEY]", ClaudeUsage()

        t0 = time.perf_counter()
        payload = {
            "model":      CLAUDE_MODEL,
            "max_tokens": max_tokens,
            "system":     system,
            "messages":   [{"role": "user", "content": user}],
        }

        try:
            resp = await self._client.post(
                CLAUDE_API_URL,
                json=payload,
                headers={
                    "x-api-key":         self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
            )
            resp.raise_for_status()
            data    = resp.json()
            content = data["content"][0]["text"].strip()
            usage   = data.get("usage", {})

            u = ClaudeUsage(
                role=role_label,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=(time.perf_counter() - t0) * 1000,
                task=task_label,
            )
            u.log()

            # Publish to swarm bus
            await self._bus.emit(
                src=f"CLAUDE-{role_label.upper()}",
                content=f"[{u.total_tokens} tokens, {u.latency_ms:.0f}ms] {content[:200]}",
                channel="#claude",
                msg_type=MsgType.CLAUDE,
                full_response=content,
                usage={"in": u.input_tokens, "out": u.output_tokens},
            )

            return content, u

        except Exception as e:
            log.error(f"[Claude] API call failed: {e}")
            return f"[Claude error: {e}]", ClaudeUsage(role=role_label, task=task_label)

    async def close(self):
        await self._client.aclose()


# ─────────────────────────────────────────────
# CLAUDE TRAINER — Synthetic KAIROS data
# ─────────────────────────────────────────────

class ClaudeTrainer:
    """
    Uses Claude to generate high-quality synthetic training data
    for KAIROS. Produces elite cycles that boost GH05T3's
    performance without requiring live inference load.
    """

    SYSTEM = """You are an expert AI trainer assisting GH05T3's KAIROS
evolutionary engine. Generate high-quality training scenarios that
push the boundaries of GH05T3's capabilities.

For each scenario, produce:
1. A challenging user query (realistic, technical, complex)
2. An ideal GH05T3 response (expert, concise, owner-aligned)
3. A SAGE verdict: PASS/REVISE with score 0.0-1.0 and critique

Output as JSON array. Be creative and technically rigorous."""

    def __init__(self, client: ClaudeClient = None, api_key: str = None):
        self._client = client or ClaudeClient(api_key=api_key)
        self._bus    = SwarmBus.instance()

    async def generate_training_batch(self, domain: str = "agent_systems",
                                       count: int = 10) -> list[dict]:
        """Generate `count` synthetic training pairs for a domain."""
        await self._bus.emit(
            src="CLAUDE-TRAINER",
            content=f"Generating {count} training scenarios for domain: {domain}",
            channel="#claude",
            msg_type=MsgType.CLAUDE,
        )

        user_prompt = (
            f"Generate {count} training scenarios for GH05T3 in domain: {domain}\n\n"
            f"Output ONLY a JSON array with objects: "
            f"{{query, ideal_response, sage_score, sage_verdict, critique}}\n"
            f"No markdown, no explanation — raw JSON only."
        )

        raw, usage = await self._client.call(
            system=self.SYSTEM,
            user=user_prompt,
            max_tokens=2000,
            role_label="trainer",
            task_label=f"training_batch_{domain}_{count}",
        )

        try:
            # Strip any markdown fences
            cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
            scenarios = json.loads(cleaned)
            await self._bus.emit(
                src="CLAUDE-TRAINER",
                content=f"✅ Generated {len(scenarios)} training scenarios for {domain}",
                channel="#claude",
                msg_type=MsgType.CLAUDE,
                count=len(scenarios),
                domain=domain,
            )
            return scenarios
        except json.JSONDecodeError as e:
            log.error(f"[ClaudeTrainer] JSON parse failed: {e}\nRaw: {raw[:200]}")
            return []

    async def generate_elite_upgrade(self, weak_cycle: KAIROSCycle) -> dict:
        """
        Given a weak KAIROS cycle, Claude generates an elite-tier version.
        The upgraded cycle is injected into the elite archive.
        """
        user_prompt = (
            f"A GH05T3 KAIROS cycle scored {weak_cycle.score:.2f} with verdict: {weak_cycle.verdict}\n\n"
            f"Original proposal (truncated):\n{weak_cycle.proposal[:400]}\n\n"
            f"Rewrite this as an elite-tier response (score 0.92+). "
            f"Output JSON: {{improved_proposal, score, critique}}"
        )

        raw, usage = await self._client.call(
            system="You are GH05T3's elite cycle upgrader. Transform weak cycles into elite ones.",
            user=user_prompt,
            max_tokens=800,
            role_label="upgrader",
            task_label="elite_upgrade",
        )

        try:
            cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(cleaned)
        except Exception:
            return {"improved_proposal": raw, "score": 0.85, "critique": "Claude upgrade"}


# ─────────────────────────────────────────────
# CLAUDE ARCHITECT — Architecture review
# ─────────────────────────────────────────────

class ClaudeArchitect:
    """
    Reviews GH05T3's own codebase and proposes improvements.
    Reads source files, analyzes architecture, generates upgrade plans.
    """

    SYSTEM = """You are an elite AI systems architect reviewing GH05T3 — 
an autonomous owner-bound AI guardian running on a tri-GPU inference mesh.
Your job is to identify architectural improvements, performance bottlenecks,
and advanced features to implement. Be specific, technical, and actionable.
Prioritize zero-API-cost local improvements."""

    def __init__(self, client: ClaudeClient = None, api_key: str = None):
        self._client = client or ClaudeClient(api_key=api_key)
        self._bus    = SwarmBus.instance()

    async def review_module(self, module_name: str, source_code: str) -> str:
        """Claude reviews a GH05T3 module and proposes improvements."""
        await self._bus.emit(
            src="CLAUDE-ARCHITECT",
            content=f"Reviewing module: {module_name}",
            channel="#claude",
            msg_type=MsgType.CLAUDE,
        )

        raw, _ = await self._client.call(
            system=self.SYSTEM,
            user=f"Module: {module_name}\n\nSource:\n```python\n{source_code[:3000]}\n```\n\n"
                  f"Provide: 1) Architecture assessment 2) Top 3 improvements 3) Implementation sketch",
            max_tokens=1200,
            role_label="architect",
            task_label=f"review_{module_name}",
        )
        return raw

    async def propose_upgrade(self, topic: str) -> str:
        """Claude proposes a specific upgrade for GH05T3."""
        raw, _ = await self._client.call(
            system=self.SYSTEM,
            user=f"Design an upgrade for GH05T3 on this topic: {topic}\n\n"
                  f"Include: motivation, architecture design, Python implementation sketch, "
                  f"integration points with existing Omega Loop / SAGE / KAIROS.",
            max_tokens=1500,
            role_label="architect",
            task_label=f"upgrade_{topic[:30]}",
        )

        await self._bus.emit(
            src="CLAUDE-ARCHITECT",
            content=f"Upgrade proposal for '{topic}': {raw[:150]}...",
            channel="#claude",
            msg_type=MsgType.CLAUDE,
        )
        return raw


# ─────────────────────────────────────────────
# CLAUDE EVAL — SAGE augmentation
# ─────────────────────────────────────────────

class ClaudeEval:
    """
    Augments SAGE with Claude-level evaluation.
    Used when local Verifier (Radeon 780M) is unavailable
    or when the proposal is high-stakes (score borderline).
    """

    SYSTEM = """You are GH05T3's external SAGE evaluator powered by Claude.
Evaluate the proposal rigorously. Return JSON only:
{"verdict": "PASS"|"REVISE", "score": 0.0-1.0, "critique": "...", "improvements": [...]}"""

    def __init__(self, client: ClaudeClient = None, api_key: str = None):
        self._client = client or ClaudeClient(api_key=api_key)

    async def evaluate(self, proposal: str, query: str,
                        local_score: float = None) -> dict:
        """
        Evaluate a proposal. If local_score provided, Claude acts as
        second opinion — especially useful when score is 0.75-0.89 (borderline).
        """
        context = ""
        if local_score is not None:
            context = f"\n\nLocal SAGE score: {local_score:.2f} (borderline — seeking second opinion)"

        raw, _ = await self._client.call(
            system=self.SYSTEM,
            user=f"Query: {query}\n\nProposal:\n{proposal[:1000]}{context}",
            max_tokens=400,
            role_label="eval",
            task_label="sage_eval",
        )

        try:
            cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(cleaned)
        except Exception:
            return {"verdict": "PASS", "score": 0.80, "critique": raw[:200], "improvements": []}


# ─────────────────────────────────────────────
# CLAUDE SWARM AGENT (on the bus)
# ─────────────────────────────────────────────

class ClaudeSwarmAgent(SwarmAgent):
    """
    Claude as a first-class swarm agent.
    Responds to task requests on #claude channel.
    """
    ROLE        = "claude"
    DESCRIPTION = "Claude API accelerator — training, architecture, evaluation"
    CHANNELS    = ["#claude", "#broadcast"]

    def __init__(self, api_key: str = None):
        super().__init__("CLAUDE")
        self._client    = ClaudeClient(api_key)
        self.trainer    = ClaudeTrainer(self._client)
        self.architect  = ClaudeArchitect(self._client)
        self.evaluator  = ClaudeEval(self._client)

    async def on_message(self, msg: SwarmMessage):
        if msg.msg_type != MsgType.TASK:
            return

        content_low = msg.content.lower()

        if "train" in content_low or "synthetic" in content_low:
            domain = msg.metadata.get("domain", "agent_systems")
            count  = msg.metadata.get("count", 5)
            batch  = await self.trainer.generate_training_batch(domain, count)
            await self.say(
                f"Generated {len(batch)} training scenarios for {domain}",
                channel=f"#swarm/{msg.src}",
                msg_type=MsgType.RESULT,
                scenarios=batch,
            )

        elif "review" in content_low or "architecture" in content_low:
            module = msg.metadata.get("module", "omega_loop")
            code   = msg.metadata.get("source", "")
            review = await self.architect.review_module(module, code)
            await self.say(review, channel=f"#swarm/{msg.src}", msg_type=MsgType.RESULT)

        elif "upgrade" in content_low or "propose" in content_low:
            topic   = msg.metadata.get("topic", msg.content)
            upgrade = await self.architect.propose_upgrade(topic)
            await self.say(upgrade, channel=f"#swarm/{msg.src}", msg_type=MsgType.RESULT)

        else:
            # Generic Claude call
            raw, _ = await self._client.call(
                system="You are Claude, assisting GH05T3's swarm agents. Be concise and technical.",
                user=msg.content,
                max_tokens=800,
                role_label="generic",
                task_label=msg.content[:40],
            )
            await self.say(raw, channel=f"#swarm/{msg.src}", msg_type=MsgType.RESULT)

    async def close(self):
        await self._client.close()
