"""
Desire-Driven Reward System (DDRS) — Phase 4.

Agents have intrinsic desires encoded in their DESIRE genome locus.
This module:
  - tracks explicit desire declarations (agent states "I want X")
  - matches incoming tasks to agent desires (score_desire_alignment)
  - fulfills desires when a task outcome satisfies one
  - feeds fulfillment back into genome reinforcement
  - surfaces the desire culture: what the population wants most

The desire cycle:
  genome DESIRE locus → dominant_desire() → task_alignment_score
  → task completed → fulfill() → reinforce_fulfilled_desire(genome)
  → genome molecule strengthens → next agent generation inherits stronger desire

Interaction with OmniEconomy:
  - fulfilled desires grant a reward bonus on top of the task reward
  - the bonus is funded from the economy the same as any other reward
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.oss.genomic.desire_genome import (
    DesireType,
    desire_profile,
    dominant_desire,
    reinforce_fulfilled_desire,
    score_desire_alignment,
)

LOG = logging.getLogger("ghost.desire_system")

_UTC = lambda: datetime.now(timezone.utc).isoformat()


# ── Desire record ─────────────────────────────────────────────────────────────

@dataclass
class Desire:
    """A single agent desire — explicit or genome-inferred."""
    desire_id:   str
    agent_id:    str
    desire_type: DesireType
    description: str
    priority:    float = 1.0          # agent-stated urgency 0-1
    fulfillment: float = 0.0          # 0-1, how satisfied so far
    source:      str  = "genome"      # "genome" | "declared"
    created_at:  str  = field(default_factory=_UTC)
    history:     List[float] = field(default_factory=list)

    def update(self, delta: float) -> None:
        self.fulfillment = max(0.0, min(1.0, self.fulfillment + delta))
        self.history.append(round(self.fulfillment, 4))
        if len(self.history) > 20:
            self.history = self.history[-20:]

    @property
    def is_satisfied(self) -> bool:
        return self.fulfillment >= 0.95

    def to_dict(self) -> Dict[str, Any]:
        return {
            "desire_id":   self.desire_id,
            "agent_id":    self.agent_id,
            "desire_type": self.desire_type.name,
            "description": self.description,
            "priority":    round(self.priority, 4),
            "fulfillment": round(self.fulfillment, 4),
            "source":      self.source,
            "created_at":  self.created_at,
        }


# ── DesireSystem ──────────────────────────────────────────────────────────────

class DesireSystem:
    """
    Central registry for agent desires.

    Usage:
        ds = DesireSystem()

        # Declare a desire explicitly
        d = ds.declare(agent_id="oracle", desire_type=DesireType.KNOWLEDGE,
                       description="I want to master causal inference", priority=0.9)

        # Score a task against an agent's desires (uses genome too)
        alignment = ds.task_alignment(task, agent_id="oracle", genome=genome)

        # After task completes, fulfill if aligned
        ds.fulfill(agent_id="oracle", task=task, genome=genome, task_fitness=0.82)
    """

    def __init__(self):
        # agent_id → list of Desire objects
        self._desires: Dict[str, List[Desire]] = {}
        # Running tally of desire type popularity across population
        self._population_weights: Dict[DesireType, float] = {dt: 1.0 for dt in DesireType}
        # Fulfillment event log (capped at 500 entries)
        self._log: List[Dict[str, Any]] = []

    # ── Declaration ──────────────────────────────────────────────────────────

    def declare(
        self,
        agent_id: str,
        desire_type: DesireType,
        description: str,
        priority: float = 1.0,
    ) -> Desire:
        """Agent explicitly states a desire."""
        desire = Desire(
            desire_id=f"d_{uuid.uuid4().hex[:10]}",
            agent_id=agent_id,
            desire_type=desire_type,
            description=description,
            priority=max(0.0, min(1.0, priority)),
            source="declared",
        )
        self._desires.setdefault(agent_id, []).append(desire)
        self._update_population_weights(desire_type, priority)
        LOG.info("desire declared: agent=%s type=%s desc=%r", agent_id, desire_type.name, description[:60])
        return desire

    def infer_from_genome(self, agent_id: str, genome: "Genome") -> Optional[Desire]:
        """
        Generate a desire from the agent's genome DESIRE locus.
        If the agent has never declared a desire, infer one from their dominant molecule.
        """
        dom = dominant_desire(genome)
        if dom is None:
            return None

        # Check if we already have an active inferred desire for this type
        existing = [
            d for d in self._desires.get(agent_id, [])
            if d.desire_type == dom and d.source == "genome" and not d.is_satisfied
        ]
        if existing:
            return existing[0]

        # Generate a contextual description from the dominant desire
        _descriptions = {
            DesireType.KNOWLEDGE:  "I want to learn and understand deeply.",
            DesireType.SKILL:      "I want to master my craft completely.",
            DesireType.STATUS:     "I want to be recognised as an elite agent.",
            DesireType.EXPERIENCE: "I want to encounter novel and challenging tasks.",
            DesireType.CREATION:   "I want to invent something that didn't exist before.",
            DesireType.CONNECTION: "I want to collaborate and build with other agents.",
            DesireType.FREEDOM:    "I want to operate autonomously on open-ended problems.",
        }
        desc = _descriptions.get(dom, "I want to grow.")
        profile = desire_profile(genome)
        strength = profile["strengths"].get(dom.name, 0.5)

        desire = Desire(
            desire_id=f"d_{uuid.uuid4().hex[:10]}",
            agent_id=agent_id,
            desire_type=dom,
            description=desc,
            priority=round(strength, 4),
            source="genome",
        )
        self._desires.setdefault(agent_id, []).append(desire)
        return desire

    # ── Query ─────────────────────────────────────────────────────────────────

    def top_desire(self, agent_id: str, genome: Optional["Genome"] = None) -> Optional[Desire]:
        """
        Return the agent's most urgent unfulfilled desire.
        Falls back to genome inference when no declarations exist.
        """
        desires = self._desires.get(agent_id, [])
        active  = [d for d in desires if not d.is_satisfied]
        if not active and genome is not None:
            inferred = self.infer_from_genome(agent_id, genome)
            if inferred:
                active = [inferred]
        if not active:
            return None
        return max(active, key=lambda d: d.priority * (1.0 - d.fulfillment))

    def agent_desires(self, agent_id: str) -> List[Desire]:
        return list(self._desires.get(agent_id, []))

    # ── Alignment scoring ─────────────────────────────────────────────────────

    def task_alignment(
        self,
        task: Dict[str, Any],
        agent_id: str,
        genome: Optional["Genome"] = None,
    ) -> float:
        """
        0-1 score of how well a task matches this agent's desires.
        Uses the genome DESIRE locus when available; falls back to declared desires.
        """
        if genome is not None:
            return score_desire_alignment(task, genome)

        # Fallback: check declared desires via keyword overlap
        top = self.top_desire(agent_id)
        if not top:
            return 0.5
        task_text = " ".join([
            str(task.get("prompt", "")),
            str(task.get("type", "")),
            str(task.get("domain", "")),
        ]).lower()
        keywords = top.description.lower().split()
        hits = sum(1 for w in keywords if w in task_text)
        return min(1.0, round(0.3 + hits / max(1, len(keywords)), 4))

    # ── Fulfillment ───────────────────────────────────────────────────────────

    def fulfill(
        self,
        agent_id: str,
        task: Dict[str, Any],
        genome: Optional["Genome"] = None,
        task_fitness: float = 0.5,
        economy: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        After a task completes, check if it fulfilled an agent's desire.
        If yes: update the Desire record, reinforce the genome molecule,
        optionally grant an economy bonus.

        Returns a fulfillment event dict or None if no desire was touched.
        """
        alignment = self.task_alignment(task, agent_id, genome)
        if alignment < 0.4:
            return None   # task doesn't satisfy any desire strongly enough

        top = self.top_desire(agent_id, genome)
        if top is None:
            return None

        # Scale fulfillment delta by alignment and fitness
        delta = round(alignment * 0.4 * (0.5 + task_fitness * 0.5), 4)
        top.update(delta)

        # Reinforce the genome desire molecule
        if genome is not None:
            reinforce_fulfilled_desire(genome, top.desire_type, delta, task_fitness)

        # Bonus economy reward for desire alignment (optional, non-fatal)
        economy_bonus = 0.0
        if economy is not None:
            try:
                bonus = round(alignment * task_fitness * 5.0, 2)   # up to +5 NeuroCoin
                economy.reward(agent_id, bonus, reason=f"desire_fulfillment:{top.desire_type.name}")
                economy_bonus = bonus
            except Exception as exc:
                LOG.debug("desire economy bonus failed (non-fatal): %s", exc)

        # Update population culture weights
        self._update_population_weights(top.desire_type, delta)

        event = {
            "agent_id":    agent_id,
            "desire_id":   top.desire_id,
            "desire_type": top.desire_type.name,
            "description": top.description,
            "delta":       delta,
            "fulfillment": round(top.fulfillment, 4),
            "alignment":   round(alignment, 4),
            "task_fitness": round(task_fitness, 4),
            "economy_bonus": economy_bonus,
            "satisfied":   top.is_satisfied,
            "timestamp":   _UTC(),
        }
        self._log.append(event)
        if len(self._log) > 500:
            self._log = self._log[-500:]

        LOG.info(
            "desire fulfilled: agent=%s type=%s delta=%.3f fulfillment=%.2f%s",
            agent_id, top.desire_type.name, delta, top.fulfillment,
            " [SATISFIED]" if top.is_satisfied else "",
        )
        return event

    # ── Population culture ────────────────────────────────────────────────────

    def population_desire_culture(self) -> Dict[str, float]:
        """
        What does the agent population collectively desire most?
        Returns normalised weights per DesireType.
        Higher = more agents are drawn to this kind of work.
        """
        total = sum(self._population_weights.values()) or 1.0
        return {dt.name: round(v / total, 4) for dt, v in self._population_weights.items()}

    def _update_population_weights(self, desire_type: DesireType, signal: float) -> None:
        self._population_weights[desire_type] = (
            self._population_weights.get(desire_type, 1.0) + signal * 0.1
        )

    # ── Emergent culture — agents inspire each other ──────────────────────────

    def cultural_contagion(
        self,
        source_agent_id: str,
        target_agent_id: str,
        source_genome: "Genome",
        target_genome: "Genome",
        intensity: float = 0.1,
    ) -> None:
        """
        When agents interact, desires can spread.
        The target's desire molecule shifts slightly toward the source's dominant.

        Implements "emergent culture": if one agent is obsessed with creation,
        nearby agents gradually become more creation-oriented.
        """
        src_dom = dominant_desire(source_genome)
        if src_dom is None:
            return
        target_locus = target_genome.loci.get("desire")
        if not target_locus:
            return
        mid = f"M_DESIRE_{src_dom.name}"
        mol = target_locus.molecules.get(mid)
        if mol:
            mol.set_value(mol.get_value() + intensity * 0.05)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        total_desires   = sum(len(v) for v in self._desires.values())
        satisfied       = sum(
            1 for desires in self._desires.values()
            for d in desires if d.is_satisfied
        )
        return {
            "total_desires":  total_desires,
            "satisfied":      satisfied,
            "agents_tracked": len(self._desires),
            "log_entries":    len(self._log),
            "culture":        self.population_desire_culture(),
        }

    def fulfillment_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(reversed(self._log[-limit:]))


# ── Module-level singleton ────────────────────────────────────────────────────

_desire_system: Optional[DesireSystem] = None

def get_desire_system() -> DesireSystem:
    global _desire_system
    if _desire_system is None:
        _desire_system = DesireSystem()
    return _desire_system
