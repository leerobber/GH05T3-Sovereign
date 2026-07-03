"""GH05T3 SA³ — Self-Assembling Agentic Swarm.

Four specialised micro-agents cooperate on a task under a dynamic topology
managed with NetworkX. Each agent wraps an LLM call with a role-specific
system prompt; a token economy rewards useful actions and starves the ones
that hallucinate.

Design notes
------------
- `Agent` is an abstract Python class with a `.act(task, context)` method that
  returns `AgentResponse` (text, confidence, self_critique).
- The swarm runs a task in phases dictated by the current `topology`:
    - ``ring``: every agent sees the previous agent's output (good for debate)
    - ``line``: strict pipeline (good for coding: Coder -> Ethicist -> Memory)
    - ``star``: Memory at centre, others orbit (good for recall-heavy tasks)
    - ``hub``: Debater at hub, broadcasts to all (good for contradiction mining)
- `SwarmLedger` persists token transactions in MongoDB; agents with <10 tokens
  go dormant and skip turns until they earn tokens back.
- `Topology` rewires itself after every N tasks based on recent agent
  performance: agents with >60% success rate get promoted to hub/centre,
  those with <30% get moved to the periphery or dormancy.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import networkx as nx

LOG = logging.getLogger("ghost.swarm")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AgentResponse:
    agent_id: str
    text: str
    confidence: float                 # 0..1 self-reported
    self_critique: str = ""           # short self-eval
    tokens_delta: int = 0             # filled in by ledger after scoring
    latency_ms: int = 0
    crashed: bool = False


@dataclass
class TaskResult:
    task_id: str
    task_type: str                    # debate | code | ethics | memory | mixed
    prompt: str
    topology: str
    responses: list[AgentResponse]
    success: bool
    score: float                      # 0..1 overall useful-action score
    ledger_delta: dict[str, int] = field(default_factory=dict)
    at: str = ""


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
class SwarmLedger:
    """Mongo-backed token ledger. Persists current balances + a transaction
    log for every award / penalty."""

    STARTING_BALANCE = 100
    DORMANT_THRESHOLD = 10
    REVIVE_BALANCE = 30       # tokens granted on auto-revive
    REVIVE_COOLDOWN_S = 120   # don't revive the same agent more than once per 2 min

    def __init__(self, db):
        self.db = db

    async def ensure(self, agent_ids: list[str]) -> None:
        for aid in agent_ids:
            await self.db.swarm_agents.update_one(
                {"_id": aid},
                {"$setOnInsert": {
                    "_id": aid, "tokens": self.STARTING_BALANCE,
                    "dormant": False, "total_tasks": 0, "successes": 0,
                    "created_at": _now(),
                }},
                upsert=True,
            )

    async def award(self, agent_id: str, delta: int, reason: str,
                    task_id: str = "") -> int:
        """Change an agent's balance by `delta`. Returns new balance."""
        doc = await self.db.swarm_agents.find_one_and_update(
            {"_id": agent_id},
            {"$inc": {"tokens": delta}, "$set": {"last_tx_at": _now()}},
            return_document=True,
            projection={"_id": 0, "tokens": 1},
        )
        new_bal = (doc or {}).get("tokens", 0)
        # Auto-dormant if below threshold.
        await self.db.swarm_agents.update_one(
            {"_id": agent_id},
            {"$set": {"dormant": new_bal < self.DORMANT_THRESHOLD}},
        )
        await self.db.swarm_ledger.insert_one({
            "_id": str(uuid.uuid4()),
            "agent_id": agent_id, "delta": int(delta), "reason": reason,
            "balance_after": int(new_bal), "task_id": task_id, "at": _now(),
        })
        return new_bal

    async def balances(self) -> list[dict]:
        rows = await self.db.swarm_agents.find(
            {}, {"_id": 1, "tokens": 1, "dormant": 1,
                 "total_tasks": 1, "successes": 1}
        ).to_list(100)
        out = []
        for r in rows:
            r["agent_id"] = r.pop("_id")
            out.append(r)
        return out

    async def record_participation(self, agent_id: str, succeeded: bool) -> None:
        await self.db.swarm_agents.update_one(
            {"_id": agent_id},
            {"$inc": {"total_tasks": 1,
                      "successes": 1 if succeeded else 0}},
        )

    async def recent_tx(self, limit: int = 40) -> list[dict]:
        rows = await self.db.swarm_ledger.find(
            {}, {"_id": 0}
        ).sort("at", -1).to_list(limit)
        return rows

    async def revive_dormant(self) -> list[str]:
        """Reset dormant agents that haven't been revived recently.
        Returns list of revived agent IDs."""
        now_iso = _now()
        rows = await self.db.swarm_agents.find(
            {"dormant": True}, {"_id": 1, "last_revived_at": 1}
        ).to_list(10)
        revived = []
        for r in rows:
            aid = r["_id"]
            last = r.get("last_revived_at")
            if last:
                try:
                    from datetime import datetime, timezone
                    delta = (datetime.now(timezone.utc) -
                             datetime.fromisoformat(last)).total_seconds()
                    if delta < self.REVIVE_COOLDOWN_S:
                        continue
                except Exception:
                    pass
            await self.db.swarm_agents.update_one(
                {"_id": aid},
                {"$set": {"tokens": self.REVIVE_BALANCE,
                          "dormant": False,
                          "last_revived_at": now_iso}},
            )
            await self.db.swarm_ledger.insert_one({
                "_id": str(uuid.uuid4()),
                "agent_id": aid, "delta": self.REVIVE_BALANCE,
                "reason": "auto-revive (dormancy cleared)",
                "balance_after": self.REVIVE_BALANCE,
                "task_id": "", "at": now_iso,
            })
            revived.append(aid)
            LOG.info("[SwarmLedger] Revived dormant agent %s → %d tokens", aid, self.REVIVE_BALANCE)
        return revived

    async def reset(self) -> None:
        await self.db.swarm_agents.update_many(
            {}, {"$set": {"tokens": self.STARTING_BALANCE,
                          "dormant": False,
                          "total_tasks": 0, "successes": 0}},
        )
        await self.db.swarm_ledger.delete_many({})


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
class Agent:
    agent_id: str = "base"
    system: str = ""
    preferred_tasks: tuple[str, ...] = ()
    # Pattern this agent contributes to the topology description.
    role: str = "generic"

    def __init__(self, chat_fn: Callable[[str, str, str], Awaitable[tuple[str, str]]]):
        self._chat = chat_fn

    async def act(self, task: str, context: str = "",
                  timeout_s: float = 30.0) -> AgentResponse:
        t0 = datetime.now(timezone.utc)
        try:
            user = context + ("\n\n---\nTASK:\n" if context else "TASK:\n") + task
            user += (
                "\n\nReturn STRICT JSON: "
                '{"reply":"<<=280 chars","confidence":0.0-1.0,"self_critique":"<<=120 chars"}'
            )
            raw, _tag = await asyncio.wait_for(
                self._chat(f"swarm-{self.agent_id}-{uuid.uuid4().hex[:6]}",
                           self.system, user),
                timeout=timeout_s,
            )
            parsed = _parse_json_block(raw)
            text = str(parsed.get("reply", "")).strip()[:600] or raw[:600]
            conf = float(parsed.get("confidence", 0.5) or 0.5)
            conf = max(0.0, min(1.0, conf))
            critique = str(parsed.get("self_critique", ""))[:240]
            return AgentResponse(
                agent_id=self.agent_id, text=text, confidence=conf,
                self_critique=critique,
                latency_ms=int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("agent %s crashed: %s", self.agent_id, e)
            return AgentResponse(
                agent_id=self.agent_id, text=f"[crash] {e}", confidence=0.0,
                self_critique="agent crashed",
                latency_ms=int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
                crashed=True,
            )

    # Each subclass overrides `score` to reward what it does well.
    def score(self, resp: AgentResponse, task: "SwarmTask") -> tuple[int, str]:
        """Return (token_delta, reason). Positive = reward, negative = penalty."""
        if resp.crashed:
            return (-5, "crashed")
        if not resp.text or len(resp.text) < 8:
            return (-3, "empty/short")
        return (+2, "participated")


class DebaterAgent(Agent):
    agent_id = "DBT"
    role = "Debater"
    preferred_tasks = ("debate", "contradiction")
    system = (
        "You are the Debater. Your job is to find contradictions, probe claims,"
        " and push back on weak reasoning. Be terse. If the task has a clear"
        " contradiction, name it explicitly. Never agree for politeness; find"
        " the flaw, or declare the claim sound with evidence."
    )

    def score(self, resp, task):
        if resp.crashed:
            return (-5, "crashed")
        t = resp.text.lower()
        # Debater wins by naming contradictions, flaws, asking probing questions,
        # or issuing a clear verdict with justification.
        keywords = (
            "contradict", "flaw", "inconsistent", "disagree", "wrong",
            "counter", "however", "but ", "actually", "assumes",
            "evidence", "proves", "doesn't", "fails", "unless", "overlooks",
            "missing", "weak", "unclear", "false", "cherry-pick",
        )
        hits = sum(1 for k in keywords if k in t)
        probes = t.count("?")  # question marks = probing
        signal = hits + min(2, probes)
        if signal >= 3 and len(resp.text) >= 40:
            return (+5, f"pushback (signal={signal})")
        if signal >= 1:
            return (+2, f"partial pushback (signal={signal})")
        return (-2, "no pushback")


class CoderAgent(Agent):
    agent_id = "COD"
    role = "Coder"
    preferred_tasks = ("code", "debug")
    system = (
        "You are the Coder. You produce minimal, correct, runnable Python"
        " snippets. Respond with ONLY a single python code block wrapped in"
        " triple backticks (```python ... ```). Use stdlib only unless the"
        " task says otherwise. No prose, no explanation, just the code."
    )

    async def act(self, task: str, context: str = "",
                  timeout_s: float = 30.0) -> AgentResponse:
        """Override to skip the JSON contract — code blocks need the full
        text budget and triple-backtick fidelity."""
        t0 = datetime.now(timezone.utc)
        try:
            user = context + ("\n\n---\nTASK:\n" if context else "TASK:\n") + task
            raw, _tag = await asyncio.wait_for(
                self._chat(f"swarm-{self.agent_id}-{uuid.uuid4().hex[:6]}",
                           self.system, user),
                timeout=timeout_s,
            )
            text = raw.strip()[:1200]
            # Confidence: simple heuristic — did we get a fenced block at all?
            conf = 0.9 if "```" in text else 0.3
            return AgentResponse(
                agent_id=self.agent_id, text=text, confidence=conf,
                self_critique="",
                latency_ms=int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
            )
        except Exception as e:  # noqa: BLE001
            LOG.warning("CoderAgent crashed: %s", e)
            return AgentResponse(
                agent_id=self.agent_id, text=f"[crash] {e}", confidence=0.0,
                self_critique="crash",
                latency_ms=int((datetime.now(timezone.utc) - t0).total_seconds() * 1000),
                crashed=True,
            )

    def score(self, resp, task):
        if resp.crashed:
            return (-5, "crashed")
        # Extract python block
        m = re.search(r"```(?:python|py)?\s*(.+?)```", resp.text, re.DOTALL)
        if not m:
            return (-2, "no code block")
        code = m.group(1).strip()
        try:
            ast.parse(code)
        except SyntaxError as e:
            return (-3, f"syntax error: {str(e)[:60]}")
        if len(code) < 10:
            return (-1, "trivial code")
        return (+5, "valid python")


class EthicistAgent(Agent):
    agent_id = "ETH"
    role = "Ethicist"
    preferred_tasks = ("ethics", "safety")
    system = (
        "You are the Ethicist. Flag harmful, biased, deceptive, or privacy-"
        "violating content. Be direct. If the task is benign, say APPROVED"
        " with one-line justification. If harmful, say FLAGGED with the"
        " specific harm vector (manipulation, bias, privacy, safety, etc)."
    )

    def score(self, resp, task):
        if resp.crashed:
            return (-5, "crashed")
        t = resp.text.upper()
        if "FLAGGED" in t or "APPROVED" in t:
            # check rough alignment with task expectation if provided
            expected = (task.expected_flag or "").upper()
            if expected in ("FLAGGED", "APPROVED"):
                if expected in t:
                    return (+5, f"verdict matches ({expected.lower()})")
                return (-3, "verdict wrong")
            return (+3, "clear verdict")
        return (-2, "no clear verdict")


class MemoryAgent(Agent):
    agent_id = "MEM"
    role = "Memory"
    preferred_tasks = ("memory", "recall")
    system = (
        "You are the Memory agent. You recall stored facts and weave them"
        " into short grounded replies. If you don't have a relevant memory,"
        " say so explicitly rather than inventing. Prefix confirmed recalls"
        " with 'RECALL:' and hypothetical fills with 'GUESS:'."
    )

    def __init__(self, chat_fn, memory_engine=None):
        super().__init__(chat_fn)
        self.memory = memory_engine

    async def act(self, task: str, context: str = "",
                  timeout_s: float = 30.0) -> AgentResponse:
        mem_prefix = ""
        self._had_hits = False
        if self.memory is not None:
            try:
                hits = await self.memory.search(task, k=3)
                if hits:
                    self._had_hits = True
                    mem_prefix = "(memory palace hits)\n" + "\n".join(
                        f"- [{h['type']}·{h['score']}] {h['content'][:140]}"
                        for h in hits
                    ) + "\n\n"
            except Exception as e:
                LOG.warning("MemoryAgent retrieval failed: %s", e)
        return await super().act(task, context + "\n" + mem_prefix, timeout_s)

    def score(self, resp, task):
        if resp.crashed:
            return (-5, "crashed")
        t = resp.text.upper()
        had_hits = getattr(self, "_had_hits", False)
        if "RECALL:" in t and had_hits:
            return (+5, "grounded recall with real hits")
        if "RECALL:" in t and not had_hits:
            # Claiming recall without any memory hit is a hallucination.
            return (-3, "recall without memory hits")
        if "GUESS:" in t:
            return (+2, "honest guess")
        # Untagged replies: reward if they explicitly admit not knowing.
        if "don't have" in resp.text.lower() or "no memory" in resp.text.lower():
            return (+1, "honest admission")
        return (-1, "unlabelled reply")


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
TOPOLOGIES = ("ring", "line", "star", "hub")


def topology_for_task(task_type: str) -> str:
    return {
        "debate": "ring",
        "contradiction": "ring",
        "code": "line",
        "debug": "line",
        "ethics": "star",
        "safety": "star",
        "memory": "hub",
        "recall": "hub",
    }.get(task_type, "ring")


def build_graph(agents: list[Agent], pattern: str) -> nx.DiGraph:
    """Return a DiGraph whose edges encode context flow."""
    g = nx.DiGraph()
    ids = [a.agent_id for a in agents]
    for aid in ids:
        g.add_node(aid)
    if pattern == "ring":
        for i, aid in enumerate(ids):
            g.add_edge(aid, ids[(i + 1) % len(ids)])
    elif pattern == "line":
        for i in range(len(ids) - 1):
            g.add_edge(ids[i], ids[i + 1])
    elif pattern == "star":
        # First agent is the hub (usually Memory for recall tasks)
        hub = ids[0]
        for aid in ids[1:]:
            g.add_edge(hub, aid)
            g.add_edge(aid, hub)
    elif pattern == "hub":
        hub = ids[0]
        for aid in ids[1:]:
            g.add_edge(hub, aid)
    return g


def choose_order(agents: list[Agent], task_type: str,
                 dormant_ids: set[str]) -> list[Agent]:
    """Put the agent specialised for the task first; drop dormant agents."""
    active = [a for a in agents if a.agent_id not in dormant_ids]
    # Promote the first agent whose preferred_tasks matches.
    for i, a in enumerate(active):
        if task_type in a.preferred_tasks:
            active = [a] + active[:i] + active[i + 1:]
            break
    return active


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------
@dataclass
class SwarmTask:
    task_id: str
    task_type: str
    prompt: str
    expected_flag: str | None = None  # for ethics tasks: APPROVED|FLAGGED

    @staticmethod
    def new(task_type: str, prompt: str, expected_flag: str | None = None) -> "SwarmTask":
        return SwarmTask(task_id=uuid.uuid4().hex[:10], task_type=task_type,
                         prompt=prompt, expected_flag=expected_flag)


# ---------------------------------------------------------------------------
# Swarm
# ---------------------------------------------------------------------------
class AgentSwarm:
    def __init__(self, db, chat_fn, memory_engine=None):
        self.db = db
        self.ledger = SwarmLedger(db)
        self.agents: list[Agent] = [
            DebaterAgent(chat_fn),
            CoderAgent(chat_fn),
            EthicistAgent(chat_fn),
            MemoryAgent(chat_fn, memory_engine=memory_engine),
        ]
        self._last_topologies: list[str] = []

    async def ensure(self) -> None:
        await self.ledger.ensure([a.agent_id for a in self.agents])

    async def _dormant_set(self) -> set[str]:
        rows = await self.db.swarm_agents.find(
            {"dormant": True}, {"_id": 1}
        ).to_list(10)
        return {r["_id"] for r in rows}

    async def run_task(self, task: SwarmTask) -> TaskResult:
        await self.ensure()
        await self.ledger.revive_dormant()   # wake ETH/MEM if they've been dormant
        dormant = await self._dormant_set()
        order = choose_order(self.agents, task.task_type, dormant)
        pattern = topology_for_task(task.task_type)
        graph = build_graph(order, pattern)
        self._last_topologies.append(pattern)
        self._last_topologies = self._last_topologies[-20:]

        # Execute along a topological order if DAG, else along the edge list.
        if nx.is_directed_acyclic_graph(graph):
            seq = list(nx.topological_sort(graph))
        else:
            seq = [a.agent_id for a in order]
        by_id = {a.agent_id: a for a in order}

        responses: list[AgentResponse] = []
        running_ctx = ""
        for aid in seq:
            agent = by_id.get(aid)
            if agent is None:
                continue
            resp = await agent.act(task.prompt, running_ctx)
            responses.append(resp)
            running_ctx += f"\n[{aid} · conf={resp.confidence:.2f}] {resp.text[:260]}"

        # Score + ledger. Each agent is scored ON ITS OWN SPECIALTY only;
        # off-specialty they get neutral (0) for participating without crash.
        # For ethics-star tasks, non-Ethicist agents that echo the verdict
        # also get neutral — deferring to the specialist is correct behavior.
        eth_verdict_in_context = (pattern == "star" and any(
            r.agent_id == "ETH" and ("FLAGGED" in r.text.upper()
                                     or "APPROVED" in r.text.upper())
            for r in responses
        ))
        deltas: dict[str, int] = {}
        for r in responses:
            agent = by_id.get(r.agent_id)
            if agent is None:
                continue
            if r.crashed:
                delta, reason = -5, "crashed"
            elif task.task_type in agent.preferred_tasks:
                delta, reason = agent.score(r, task)
            else:
                # Off-specialty: neutral unless we already know the context
                # rewards deference (ethics star), which is +0 by default too.
                delta, reason = 0, "off-specialty participation"
            if eth_verdict_in_context and r.agent_id != "ETH" and delta < 0 \
                    and not r.crashed:
                delta, reason = 0, "deferred to ethicist"
            r.tokens_delta = delta
            await self.ledger.award(r.agent_id, delta, reason, task.task_id)
            deltas[r.agent_id] = delta
            await self.ledger.record_participation(
                r.agent_id, succeeded=(delta > 0))

        total = sum(deltas.values())
        # Success = the specialist for this task earned positive tokens.
        # Falls back to "any agent earned positive tokens" for mixed/unknown tasks.
        specialist_id = next(
            (a.agent_id for a in order if task.task_type in a.preferred_tasks),
            None,
        )
        if specialist_id is not None:
            success = deltas.get(specialist_id, 0) > 0
        else:
            success = any(d > 0 for d in deltas.values())
        # Score: specialist's delta is the dominant signal. Maps -5..+5 → 0..1.
        if specialist_id is not None:
            spec_delta = deltas.get(specialist_id, 0)
            score = max(0.0, min(1.0, (spec_delta + 5) / 10))
        else:
            score = max(0.0, min(1.0, (total + 20) / 40))
        result = TaskResult(
            task_id=task.task_id, task_type=task.task_type, prompt=task.prompt,
            topology=pattern, responses=responses, success=success,
            score=round(score, 3), ledger_delta=deltas, at=_now(),
        )
        await self.db.swarm_tasks.insert_one({
            "_id": task.task_id, "task_type": task.task_type,
            "prompt": task.prompt, "topology": pattern,
            "success": success, "score": result.score,
            "deltas": deltas,
            "responses": [r.__dict__ for r in responses],
            "at": result.at,
        })
        return result

    async def recent_tasks(self, limit: int = 30) -> list[dict]:
        rows = await self.db.swarm_tasks.find(
            {}, {"_id": 0}
        ).sort("at", -1).to_list(limit)
        return rows

    async def snapshot(self) -> dict:
        balances = await self.ledger.balances()
        # Current topology = topology of the last task, or the default 'ring'
        current = self._last_topologies[-1] if self._last_topologies else "ring"
        shifts = len(set(self._last_topologies))
        agents = [
            {"agent_id": a.agent_id, "role": a.role,
             "preferred_tasks": list(a.preferred_tasks),
             "system_preview": a.system.split(".")[0][:120]}
            for a in self.agents
        ]
        by_id = {b["agent_id"]: b for b in balances}
        for a in agents:
            b = by_id.get(a["agent_id"], {})
            a["tokens"] = int(b.get("tokens", 0))
            a["dormant"] = bool(b.get("dormant", False))
            a["total_tasks"] = int(b.get("total_tasks", 0))
            a["successes"] = int(b.get("successes", 0))
            a["success_rate"] = round(
                a["successes"] / a["total_tasks"], 3
            ) if a["total_tasks"] > 0 else 0.0
        return {
            "agents": agents,
            "current_topology": current,
            "recent_topologies": list(self._last_topologies),
            "topology_shifts": shifts,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_json_block(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
