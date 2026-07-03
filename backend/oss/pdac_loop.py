"""
PDAC Loop — Plan -> Delegate -> Act -> Codify.

Maps to the SovereignNation agent roles:
  Plan    = Oracle    (highest M_REASONING_DEPTH)
  Delegate= Nexus     (highest M_PATTERN_RECOGNITION)
  Act     = Forge(s)  (highest M_ADAPTABILITY — can be multiple)
  Codify  = canonical memory write via SharedWorkingMemory

Wired as a first-class primitive for complex multi-step tasks.
The loop returns a PDACResult with intermediate phases, metrics,
and a context maturity score for the oracle agent.

Usage:
    loop = PDACLoop(swarm)
    result = loop.run(task)
    print(result.metrics["pdac_score"])
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.oss.context_maturity import get_context_maturity_scorer
from backend.oss.mind.shared_working_memory import get_shared_memory

LOG = logging.getLogger("ghost.pdac_loop")
_UTC = lambda: datetime.now(timezone.utc).isoformat()


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class PDACResult:
    task_id:            str
    plan:               Dict[str, Any]
    delegation:         Dict[str, Any]
    execution_results:  List[Dict[str, Any]]
    assessment:         Dict[str, Any]
    codified:           Dict[str, Any]
    metrics:            Dict[str, float]
    context_maturity:   int
    timestamp:          str = field(default_factory=_UTC)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":           self.task_id,
            "plan":              self.plan,
            "delegation":        self.delegation,
            "execution_results": self.execution_results,
            "assessment":        self.assessment,
            "codified":          self.codified,
            "metrics":           self.metrics,
            "context_maturity":  self.context_maturity,
            "timestamp":         self.timestamp,
        }


# ── PDACLoop ──────────────────────────────────────────────────────────────────

class PDACLoop:
    """
    Runs the Plan → Delegate → Act → Codify loop over an AgentSwarm.

    Agent selection:
      - Oracle  = agent with highest M_REASONING_DEPTH in cognitive locus
      - Nexus   = agent with highest M_PATTERN_RECOGNITION in cognitive locus
      - Forge   = top-3 agents by M_ADAPTABILITY in cognitive locus
    """

    def __init__(self, swarm: Any):
        self._swarm      = swarm
        self._scorer     = get_context_maturity_scorer()
        self._memory     = get_shared_memory()

    # ── Agent selection ───────────────────────────────────────────────────────

    def _pick_oracle(self) -> Optional[Tuple[str, Any]]:
        return self._pick_by_molecule("cognitive", "M_REASONING_DEPTH")

    def _pick_nexus(self) -> Optional[Tuple[str, Any]]:
        return self._pick_by_molecule("cognitive", "M_PATTERN_RECOGNITION")

    def _pick_forge(self, n: int = 3) -> List[Tuple[str, Any]]:
        scored = [
            (aid, agent, agent.genome.get_value("cognitive", "M_ADAPTABILITY"))
            for aid, agent in self._swarm.agents.items()
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(a[0], a[1]) for a in scored[:n]]

    def _pick_by_molecule(self, locus: str, mol: str) -> Optional[Tuple[str, Any]]:
        best_id, best_agent, best_val = None, None, -1.0
        for aid, agent in self._swarm.agents.items():
            v = agent.genome.get_value(locus, mol)
            if v > best_val:
                best_val, best_id, best_agent = v, aid, agent
        return (best_id, best_agent) if best_id else None

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self, task: Dict[str, Any]) -> PDACResult:
        """Execute the full PDAC loop for a task."""
        if not self._swarm.agents:
            self._swarm.add_agent()

        task_id = task.get("task_id", f"pdac_{uuid.uuid4().hex[:8]}")

        # ── Phase 0: context maturity gate ───────────────────────────────────
        oracle_pair = self._pick_oracle()
        if oracle_pair is None:
            oracle_pair = next(iter(self._swarm.agents.items()))

        oracle_id, oracle = oracle_pair
        maturity_score = self._scorer.score(oracle.genome, task)
        context_maturity = maturity_score.score

        LOG.info(
            "PDAC start: task_id=%s oracle=%s ctx_maturity=%d",
            task_id, oracle_id, context_maturity,
        )

        # ── Phase 1: PLAN (Oracle) ─────────────────────────────────────────
        plan_task = {
            **task,
            "task_id":    f"{task_id}_plan",
            "type":       "analysis",
            "phase":      "PDAC_Plan",
            "prompt": (
                f"[PLAN] {task.get('prompt', '')} | "
                "Create a structured plan with sub-tasks, success criteria, and risk flags."
            ),
        }
        plan_result = oracle.act(plan_task)
        self._memory.write(
            task_id, oracle_id,
            content={"phase": "plan", "output": plan_result["output"]},
            entry_type="plan", ttl_seconds=600,
        )

        # ── Contextual Short-Circuit ───────────────────────────────────────
        # When Oracle has deep context maturity (>6) AND the task is routine,
        # skip Nexus delegation entirely. Forge agents self-organize from
        # SharedWorkingMemory, freeing Nexus cycles for high-order strategy.
        _ROUTINE_TYPES = {"analysis", "persuasion", "trade", "market", "default"}
        _short_circuit = (
            context_maturity > 6
            and task.get("type", "default") in _ROUTINE_TYPES
        )

        delegation_result: Dict[str, Any]
        nexus_id: str

        if _short_circuit:
            # Self-organize: Forge reads prior canonical hints from SWM
            prior_entries = self._memory.read(f"enrichment_{task.get('type', 'default')}")
            prior_hints   = [e.content for e in prior_entries]
            delegation_result = {
                "agent_id": oracle_id,
                "output":   {
                    "type":    "short_circuit",
                    "content": (
                        f"[SHORT-CIRCUIT] ctx_maturity={context_maturity} >6 on routine task "
                        f"'{task.get('type', 'default')}'. Delegate skipped. "
                        f"{len(prior_hints)} SWM hints loaded for Forge self-organization."
                    ),
                    "hints": prior_hints,
                },
                "fitness":  plan_result["fitness"],
                "metrics":  {},
            }
            nexus_id = oracle_id
            LOG.info(
                "PDAC short-circuit: task_id=%s maturity=%d skipping Delegate",
                task_id, context_maturity,
            )
        else:
            # ── Phase 2: DELEGATE (Nexus) ─────────────────────────────────
            nexus_pair = self._pick_nexus()
            if nexus_pair is None:
                nexus_pair = oracle_pair
            nexus_id, nexus = nexus_pair

            delegate_task = {
                **task,
                "task_id":  f"{task_id}_delegate",
                "type":     "analysis",
                "phase":    "PDAC_Delegate",
                "prompt": (
                    f"[DELEGATE] {task.get('prompt', '')} | "
                    "Assign sub-tasks to specialist agents. Specify agent type, expected output."
                ),
                "plan_output": plan_result["output"],
            }
            delegation_result = nexus.act(delegate_task)

        self._memory.write(
            task_id, nexus_id,
            content={"phase": "delegate", "output": delegation_result["output"],
                     "short_circuit": _short_circuit},
            entry_type="delegation", ttl_seconds=600,
        )

        # ── Phase 3: ACT (Forge agents) ───────────────────────────────────
        forge_pairs = self._pick_forge(n=min(3, len(self._swarm.agents)))
        if not forge_pairs:
            forge_pairs = [oracle_pair]

        execution_results: List[Dict[str, Any]] = []
        for i, (fid, forge) in enumerate(forge_pairs):
            exec_task = {
                **task,
                "task_id":  f"{task_id}_exec_{i}",
                "type":     task.get("type", "default"),
                "phase":    "PDAC_Act",
                "prompt": (
                    f"[EXECUTE sub-task {i+1}] {task.get('prompt', '')} | "
                    f"Deliver concrete output for your assigned segment."
                ),
                "delegation_output": delegation_result["output"],
                "short_circuit":     _short_circuit,
            }
            exec_result = forge.act(exec_task)
            execution_results.append(exec_result)
            self._memory.write(
                task_id, fid,
                content={"phase": "execute", "subtask": i, "output": exec_result["output"]},
                entry_type="execution", ttl_seconds=600,
            )

        # ── Phase 4: ASSESS (Oracle again) ────────────────────────────────
        assess_task = {
            **task,
            "task_id":  f"{task_id}_assess",
            "type":     "analysis",
            "phase":    "PDAC_Assess",
            "prompt": (
                f"[ASSESS] Evaluate the execution results for task: {task.get('prompt', '')} | "
                "Score quality, completeness, alignment with plan."
            ),
            "exec_count": len(execution_results),
        }
        assessment = oracle.act(assess_task)
        self._memory.write(
            task_id, oracle_id,
            content={"phase": "assess", "output": assessment["output"]},
            entry_type="assessment", ttl_seconds=600,
        )

        # ── Phase 5: CODIFY (persist to SharedWorkingMemory for future recall)
        insights_text = "\n".join(
            str(r["output"].get("content", "")) for r in execution_results
        )
        codified = {
            "task_id":      task_id,
            "insights":     insights_text[:1000],
            "plan_fitness": plan_result["fitness"],
            "exec_fitness": sum(r["fitness"] for r in execution_results) / max(1, len(execution_results)),
            "assess_fitness": assessment["fitness"],
        }
        self._memory.write(
            task_id, "pdac_loop",
            content=codified, entry_type="codified", ttl_seconds=3600,
        )

        # ── Metrics ───────────────────────────────────────────────────────
        exec_fitness = codified["exec_fitness"]
        metrics = {
            "plan_quality":       round(plan_result["fitness"], 4),
            "delegation_quality": round(delegation_result["fitness"], 4),
            "execution_quality":  round(exec_fitness, 4),
            "assessment_quality": round(assessment["fitness"], 4),
            "context_maturity":   float(context_maturity),
            "short_circuit":      _short_circuit,
            "pdac_score":         round(
                (plan_result["fitness"] + delegation_result["fitness"]
                 + exec_fitness + assessment["fitness"]) / 4.0, 4
            ),
        }

        LOG.info(
            "PDAC complete: task_id=%s pdac_score=%.3f",
            task_id, metrics["pdac_score"],
        )

        return PDACResult(
            task_id=task_id,
            plan=plan_result,
            delegation=delegation_result,
            execution_results=execution_results,
            assessment=assessment,
            codified=codified,
            metrics=metrics,
            context_maturity=context_maturity,
        )

    def get_task_memory(self, task_id: str) -> Dict[str, Any]:
        return self._memory.summary(task_id)

    def clear_task(self, task_id: str) -> None:
        self._memory.clear(task_id)
