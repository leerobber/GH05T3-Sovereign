"""
BMEBridge — cross-universe bridge for the Binary Multiverse Engine.

This is the integration layer that connects:
  ChronosLedger  (agent binary state)
  GenomePlane    (skill gene arrays)
  SkillRegistry  (compiled skills.bin)
  UniverseEngine (desire amplification + fitness)
  PatentOffice   (breakthrough proposals)

All operations read/write binary mmap — no JSON, no DB, no LLM calls.
The BMEBridge is what GenesisThread calls on each tick to run the
multi-universe evolution pass.

Key flows:
  1. Tick pass         → BMEBridge.universe_pass() called by GenesisThread
  2. Migration         → agent's universe bits updated in ChronosLedger scratchpad
  3. Speciation        → elite agent proposes new skill → SkillRegistry.propose_skill()
  4. Role promotion    → scratchpad bits updated
  5. SovereignCore :9000 integration → /v1/universes endpoint (optional HTTP push)

Scratchpad bit layout (in ChronosLedger slot, 8 bytes from _OFF_SCRATCHPAD=24):
  bit 0:     LOCKED
  bit 1:     PATENTED
  bit 2:     NEEDS_REVIEW
  bit 3-5:   UNIVERSE_ID     (0-7, 3 bits)
  bit 6-8:   ROLE_TIER       (0-7, 3 bits)
  bit 9:     MIGRANT         (migrated from another universe this gen)
  bit 10:    SPECIATION      (triggered a speciation event)
  bit 11:    ELITE_PROPOSAL  (has proposed a new skill)
  bit 12:    BREAKTHROUGH_GENE (has a discovery gene)

These constants are also defined in chronos_ledger.py (added by this session).
This file references them by name, not by re-defining them.
"""

from __future__ import annotations

import json
import logging
import struct
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Any, Dict
import numpy as np

class SovereignCorePushSchema(BaseModel):
    """
    Bridge security layer for SovereignCore communication.
    Enforces zero-trust validation matrices on incoming telemetry and execution commands.
    """
    tick: int
    mode: Literal["live", "observability", "soft"] = Field(default="observability")
    shadow_manifest: Optional[Dict[str, Any]] = Field(default=None)
    execution_params: Optional[Dict[str, Any]] = Field(default=None)

    @property
    def is_observability_only(self) -> bool:
        return self.mode == "observability"

    @model_validator(mode="after")
    def verify_soft_mode_constraints(self) -> "SovereignCorePushSchema":
        """
        Hard gatekeeper for soft-mode micro caps.
        Guarantees that no agent can bypass size caps at the serialization layer.
        """
        if self.mode == "soft":
            if not self.execution_params:
                raise ValueError("Soft-mode activation requires valid execution_params.")
            
            # Hard boundary: Maximum allowed allocation per micro-cycle
            MAX_ALLOWED_USD = 500.00
            size_usd = self.execution_params.get("size_usd", 0.0)
            
            if size_usd > MAX_ALLOWED_USD:
                raise ValueError(
                    "CRITICAL_VIOLATION: Soft-mode cap exceeded. "
                    "Requested: ${0:.2f}, Max Allowed: ${1:.2f}".format(size_usd, MAX_ALLOWED_USD)
                )
        return self

    def to_validated_dict(self) -> dict:
        return self.model_dump()

    def to_validated_json(self) -> str:
        return self.model_dump_json()

from .chronos_ledger import (
    ChronosLedger,
    DESIRE_NAMES,
    SCRATCH_LOCKED,
    SCRATCH_PATENTED,
    SCRATCH_NEEDS_REVIEW,
    SCRATCH_UNIVERSE_MASK,
    SCRATCH_ROLE_TIER_MASK,
    SCRATCH_MIGRANT,
    SCRATCH_SPECIATION,
    SCRATCH_ELITE_PROPOSAL,
    SCRATCH_BREAKTHROUGH_GENE,
    UNIVERSE_SHIFT,
    ROLE_TIER_SHIFT,
)
from .genome_plane import GenomePlane, GENE_FLAG_ACTIVE, GENE_FLAG_INHERITED, get_genome_plane
from .skill_registry import (
    SkillRegistry,
    SKILL_FLAG_ELITE_ONLY,
    UNIVERSE_NAMES,
    get_skill_registry,
)
from .universe_engine import (
    UniverseEngine,
    UNIVERSE_ADJACENCY,
    get_universe_engine,
)

LOG = logging.getLogger("ghost.bme_bridge")

_DESIRE_STRUCT = "7e"          # 7 × float16
_DESIRE_SIZE   = struct.calcsize("<" + _DESIRE_STRUCT)

# Thresholds for autonomous events
MIGRATION_FITNESS_DELTA  = 0.15   # minimum improvement to trigger universe migration
SPECIATION_TIER_GATE     = 4      # minimum role tier to propose a new skill
PROMOTION_DISCOVERY_GATE = 3      # discoveries needed per tier promotion above 3
ELITE_GENOME_THRESHOLD   = 0.80   # genome_fitness_contribution to get BREAKTHROUGH_GENE


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    return value


def shape_flagship_profile(profile: Any, species_name: str | None = None) -> Dict[str, Any]:
    if not profile:
        return {}

    if isinstance(profile, dict):
        species = species_name or str(
            profile.get("species_name")
            or profile.get("species")
            or profile.get("name")
            or ""
        )
        universe = profile.get("preferred_universe", profile.get("universe", 7))
        traits = profile.get("traits") or profile.get("mutation_bias") or {}
        rewards = profile.get("rewards") or {}
        shaped = {
            "species": species,
            "universe": int(universe),
            "traits": {k: _to_json_safe(v) for k, v in traits.items()},
            "rewards": {k: _to_json_safe(v) for k, v in rewards.items()},
        }
        if "preferred_universe" in profile:
            shaped["preferred_universe"] = int(profile["preferred_universe"])
        if "desire_bias" in profile:
            shaped["desire_bias"] = _to_json_safe(profile["desire_bias"])
        if "role_bias" in profile:
            shaped["role_bias"] = profile["role_bias"]
        return shaped

    species = species_name or str(getattr(profile, "species_name", ""))
    universe = int(getattr(profile, "universe_id", getattr(profile, "preferred_universe", 7)))
    traits = getattr(profile, "traits", {})
    rewards = getattr(profile, "rewards", {})
    return {
        "species": species,
        "universe": universe,
        "traits": {k: _to_json_safe(v) for k, v in traits.items()},
        "rewards": {k: _to_json_safe(v) for k, v in rewards.items()},
    }


class BMEStats(dict):
    """Dict-backed summary so callers can treat stats as mapping or JSON payload."""

    def __init__(
        self,
        *,
        active_slots: int,
        universe_distribution: Dict[str, int],
        role_counts: Dict[int, int],
        trait_summary: Dict[str, Any],
        reward_summary: Dict[str, Any],
    ) -> None:
        super().__init__(
            active_slots=active_slots,
            universe_distribution=universe_distribution,
            role_counts=role_counts,
            trait_summary=trait_summary,
            reward_summary=reward_summary,
        )

    def as_dict(self) -> Dict[str, Any]:
        return dict(self)


class BMEBridge:
    """
    Integration hub for the Binary Multiverse Engine.

    Instantiate once per process; the singletons it pulls (ledger, genome, etc.)
    are global. Pass explicit instances for testing.
    """

    def __init__(
        self,
        ledger:   Optional[ChronosLedger]  = None,
        genome:   Optional[GenomePlane]    = None,
        registry: Optional[SkillRegistry]  = None,
        engine:   Optional[UniverseEngine] = None,
        patent_office: Optional[Any]       = None,
    ):
        self._ledger   = ledger   or ChronosLedger()
        self._genome   = genome   or get_genome_plane()
        self._registry = registry or get_skill_registry()
        self._engine   = engine   or get_universe_engine()
        # PatentOffice is optional — import lazily to avoid circular deps
        if patent_office is not None:
            self._patent = patent_office
        else:
            try:
                from .patent_office import PatentOffice
                self._patent = PatentOffice(self._ledger)
            except ImportError:
                self._patent = None

    # ─────────────────────────────────────────────────────────────────────────
    # Scratchpad helpers (universe + role tier bits)
    # ─────────────────────────────────────────────────────────────────────────

    def get_agent_universe(self, slot: int) -> int:
        scratch = self._ledger.get_scratchpad(slot)
        return (scratch & SCRATCH_UNIVERSE_MASK) >> UNIVERSE_SHIFT

    def set_agent_universe(self, slot: int, universe_id: int) -> None:
        scratch = self._ledger.get_scratchpad(slot)
        scratch &= ~SCRATCH_UNIVERSE_MASK
        scratch |= (universe_id & 0b111) << UNIVERSE_SHIFT
        self._ledger.set_scratchpad(slot, scratch)

    def get_role_tier(self, slot: int) -> int:
        scratch = self._ledger.get_scratchpad(slot)
        return (scratch & SCRATCH_ROLE_TIER_MASK) >> ROLE_TIER_SHIFT

    def set_role_tier(self, slot: int, tier: int) -> None:
        scratch = self._ledger.get_scratchpad(slot)
        scratch &= ~SCRATCH_ROLE_TIER_MASK
        scratch |= (tier & 0b111) << ROLE_TIER_SHIFT
        self._ledger.set_scratchpad(slot, scratch)

    def get_discovery_count(self, slot: int) -> int:
        """Proxy: count non-zero skill_ids in genome as a discovery score."""
        genes = self._genome.active_genes(slot)
        return len(genes)

    # ─────────────────────────────────────────────────────────────────────────
    # Fitness scoring
    # ─────────────────────────────────────────────────────────────────────────

    def agent_universe_fitness(self, slot: int) -> Dict[int, float]:
        """Score this agent under every universe. Used for migration decisions."""
        desires     = self._ledger.get_desires(slot)   # float16 (7,) or (13,)
        desires_f32 = desires.astype(np.float32)
        base_fit    = float(self._ledger.get_fitness(slot))
        return self._engine.compare_universe_fitness(desires_f32, base_fit)

    def compute_genome_fitness(self, slot: int) -> float:
        """Weighted genome expression contribution to raw fitness."""
        return self._genome.genome_fitness_contribution(slot)

    # ─────────────────────────────────────────────────────────────────────────
    # Universe migration
    # ─────────────────────────────────────────────────────────────────────────

    def check_migration(self, slot: int) -> Optional[int]:
        """
        Return the universe to migrate to if a better adjacent one exists.
        Returns None if no migration is warranted.
        """
        current = self.get_agent_universe(slot)
        tier = self.get_role_tier(slot)
        desires = self._ledger.get_desires(slot).astype(np.float32)
        base_fit = float(self._ledger.get_fitness(slot))
        return self._engine.migration_candidate(
            current, desires, base_fit, MIGRATION_FITNESS_DELTA, agent_tier=tier
        )

    def migrate_agent(self, slot: int, target_universe: int) -> bool:
        """
        Execute universe migration: update scratchpad universe bits + MIGRANT flag,
        inject a seed gene from target universe into agent's genome.
        Returns True if migration occurred.
        """
        current = self.get_agent_universe(slot)
        if current == target_universe:
            return False
        if target_universe not in UNIVERSE_ADJACENCY.get(current, []):
            LOG.debug(
                "BMEBridge: migration blocked slot=%d %d→%d not adjacent",
                slot, current, target_universe,
            )
            return False

        # Update scratchpad
        self.set_agent_universe(slot, target_universe)
        scratch = self._ledger.get_scratchpad(slot)
        self._ledger.set_scratchpad(slot, scratch | SCRATCH_MIGRANT)

        # Inject one accessible skill from target universe into genome
        tier = self.get_role_tier(slot)
        accessible = sorted(
            self._registry.get_accessible_skills(target_universe, tier),
            key=lambda skill: (
                float(skill.get("reward_weight", 0.0)),
                int(skill.get("role_tier", 0)),
            ),
            reverse=True,
        )
        if accessible:
            skill = accessible[0]   # highest reward_weight (registry returns sorted)
            idx = self._genome.inject_gene(
                slot,
                target_universe,
                skill["role_tier"],
                skill["skill_id"],
                expression=0.4,
            )
            if idx >= 0:
                LOG.info(
                    "BMEBridge: slot=%d migrated %s→%s + gene '%s'",
                    slot, UNIVERSE_NAMES[current], UNIVERSE_NAMES[target_universe],
                    skill["name"],
                )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Role tier promotion
    # ─────────────────────────────────────────────────────────────────────────

    def check_promotion(self, slot: int) -> bool:
        """
        Promote agent to next role tier if criteria met.
        Returns True if promotion occurred.
        """
        tier      = self.get_role_tier(slot)
        fitness   = float(self._ledger.get_fitness(slot))
        gen       = self._ledger.get_generation(slot)
        discovery = self.get_discovery_count(slot)

        if not self._engine.promotion_criteria(tier, fitness, gen, discovery):
            return False

        new_tier = tier + 1
        self.set_role_tier(slot, new_tier)

        scratch = self._ledger.get_scratchpad(slot)
        if new_tier >= SPECIATION_TIER_GATE:
            scratch |= SCRATCH_SPECIATION
        self._ledger.set_scratchpad(slot, scratch)

        LOG.info(
            "BMEBridge: slot=%d promoted to tier %d (fit=%.3f gen=%d disc=%d)",
            slot, new_tier, fitness, gen, discovery,
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Genome augmentation: BREAKTHROUGH_GENE flag
    # ─────────────────────────────────────────────────────────────────────────

    def check_breakthrough_gene(self, slot: int) -> bool:
        contrib = self.compute_genome_fitness(slot)
        if contrib >= ELITE_GENOME_THRESHOLD:
            scratch = self._ledger.get_scratchpad(slot)
            self._ledger.set_scratchpad(slot, scratch | SCRATCH_BREAKTHROUGH_GENE)
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Speciation events (elite agent proposes new skill)
    # ─────────────────────────────────────────────────────────────────────────

    def speciation_event(
        self,
        slot: int,
        skill_name: str,
        input_sig: int,
        output_sig: int,
        dominance: int = 1,
        mutation_rate: float = 0.05,
        reward_weight: float = 0.70,
    ) -> Optional[int]:
        """
        Elite agent proposes a new skill to the registry.
        Returns new skill_id or None if tier gate not met.
        """
        tier = self.get_role_tier(slot)
        if tier < SPECIATION_TIER_GATE:
            LOG.debug(
                "BMEBridge: speciation denied slot=%d tier=%d (need %d)",
                slot, tier, SPECIATION_TIER_GATE,
            )
            return None

        universe_id = self.get_agent_universe(slot)
        universe_bits = (1 << universe_id)

        sid = self._registry.propose_skill(
            universe_id=universe_id,
            name=skill_name,
            input_sig=input_sig,
            output_sig=output_sig,
            dominance=dominance,
            mutation_rate=mutation_rate,
            reward_weight=reward_weight,
            universe_bits=universe_bits,
            flags=0b010,   # experimental
            role_tier=tier,
        )

        # Mark agent as having made a proposal
        scratch = self._ledger.get_scratchpad(slot)
        self._ledger.set_scratchpad(slot, scratch | SCRATCH_ELITE_PROPOSAL)

        # Inject the new skill into the proposing agent's genome at high expression
        self._genome.inject_gene(slot, universe_id, tier, sid, expression=0.8)

        # Optionally route to PatentOffice for governance review
        if self._patent is not None:
            try:
                self._patent.handle_breakthrough_signal(
                    slot, self._ledger.get_desires(slot),
                    {"skill_id": sid, "name": skill_name},
                )
            except Exception as exc:
                LOG.warning("BMEBridge: patent_office signal failed: %s", exc)

        LOG.info(
            "BMEBridge: speciation slot=%d tier=%d → skill '%s' (id=%d)",
            slot, tier, skill_name, sid,
        )
        return sid

    # ─────────────────────────────────────────────────────────────────────────
    # Offspring genome inheritance
    # ─────────────────────────────────────────────────────────────────────────

    def inherit_genome(
        self,
        parent_slot: int,
        child_slot: int,
        mutation_rate: Optional[float] = None,
    ) -> None:
        """
        Copy parent genome to child, mutate, and inherit universe/role bits.
        Called by GenesisThread._spawn_offspring() AFTER writing the child slot.
        """
        parent_universe = self.get_agent_universe(parent_slot)
        parent_tier     = self.get_role_tier(parent_slot)

        # Inherit genome: copy then mutate
        univ_params = __import__(
            "backend.oss.core.universe_engine",
            fromlist=["UNIVERSE_MUTATION_PARAMS"],
        ).UNIVERSE_MUTATION_PARAMS
        params = univ_params.get(parent_universe, {"mutation_rate": 0.05, "drift_sigma": 0.1})
        rate   = mutation_rate or params["mutation_rate"]
        sigma  = params["drift_sigma"]

        # Start child genome as copy of parent
        self._genome.clear_genome(child_slot)
        parent_genes = self._genome.read_genome(parent_slot)
        for i, gene in enumerate(parent_genes):
            if gene is None:
                continue
            self._genome.write_gene(
                child_slot, i,
                gene["universe_id"], gene["role_tier"],
                gene["skill_id"], gene["expression"],
                GENE_FLAG_ACTIVE | GENE_FLAG_INHERITED,
            )

        # Mutate child
        self._genome.mutate_genome(
            child_slot,
            mutation_rate=rate,
            expression_drift=sigma,
        )

        # Inherit universe from parent; tier starts one below parent (capped at 0)
        child_tier = max(0, parent_tier - 1)
        self.set_agent_universe(child_slot, parent_universe)
        self.set_role_tier(child_slot, child_tier)

    # ─────────────────────────────────────────────────────────────────────────
    # Universe-level pass (called from GenesisThread)
    # ─────────────────────────────────────────────────────────────────────────

    def universe_pass(
        self,
        active_slots: Optional[List[int]] = None,
        target_universe: Optional[int] = None,
        allow_migration: bool = True,
        allow_promotion: bool = True,
        allow_breakthrough: bool = True,
    ) -> Dict:
        """
        Run one universe evolution tick across all active agent slots.

        1. Compute per-universe average fitness
        2. Check migrations (adjacent universe pressure)
        3. Check role promotions
        4. Flag breakthrough genes

        Returns summary dict for logging.
        """
        if active_slots is None:
            active_slots = list(range(min(self._ledger.active_slots, self._ledger.capacity)))
        if target_universe is not None:
            active_slots = [
                slot for slot in active_slots
                if self.get_agent_universe(slot) == target_universe
            ]

        migrations     = 0
        promotions     = 0
        breakthroughs  = 0
        universe_counts: Dict[int, int] = {}

        for slot in active_slots:
            uid = self.get_agent_universe(slot)
            universe_counts[uid] = universe_counts.get(uid, 0) + 1

            if allow_migration:
                target = self.check_migration(slot)
                if target is not None:
                    if self.migrate_agent(slot, target):
                        migrations += 1

            if allow_promotion:
                if self.check_promotion(slot):
                    promotions += 1

            if allow_breakthrough:
                if self.check_breakthrough_gene(slot):
                    breakthroughs += 1

        return {
            "agents_processed": len(active_slots),
            "migrations":        migrations,
            "promotions":        promotions,
            "breakthroughs":     breakthroughs,
            "universe_counts":   universe_counts,
            "target_universe":   target_universe,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SovereignCore :9000 push (optional)
    # ─────────────────────────────────────────────────────────────────────────

    async def push_to_sovereign_core(self, session: Any, schema: SovereignCorePushSchema) -> Dict[str, Any]:
        """
        Bifurcated network router. 
        Guarantees absolute separation between telemetry and production endpoints.
        """
        payload = schema.model_dump()
        
        if schema.mode == "observability":
            endpoint = "http://localhost:9000/v1/ingest/shadow"
        elif schema.mode == "soft":
            endpoint = "http://localhost:9000/v1/universes/soft"
        elif schema.mode == "live":
            # Production live endpoint remains hard-locked until verification completion
            raise PermissionError("HARD_LOCK: Production live engine is globally deactivated.")
        else:
            raise ValueError("UNSUPPORTED_BRIDGE_MODE: Rejecting packet packet injection.")

        # Implementation uses explicit formatting to maintain zero-f-string discipline
        print("[BRIDGE_ROUTING] Routing payload to: {0} | Mode: {1}".format(endpoint, schema.mode))
        
        async with session.post(endpoint, json=payload) as response:
            return await response.json()

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def population_universe_distribution(self, active_slots: List[int]) -> Dict:
        dist: Dict[int, int] = {}
        for slot in active_slots:
            uid = self.get_agent_universe(slot)
            dist[uid] = dist.get(uid, 0) + 1
        return {UNIVERSE_NAMES.get(k, str(k)): v for k, v in dist.items()}

    def population_tier_distribution(self, active_slots: List[int]) -> Dict:
        dist: Dict[int, int] = {}
        for slot in active_slots:
            tier = self.get_role_tier(slot)
            dist[tier] = dist.get(tier, 0) + 1
        return dist

    def collect_stats(self, active_slots: Optional[List[int]] = None) -> BMEStats:
        """
        Summarize the current BME population.

        Designed for the stress runner and /oss/bme/status surface.
        """
        if active_slots is None:
            active_slots = list(range(min(self._ledger.active_slots, self._ledger.capacity)))
        active_slots = [
            slot for slot in active_slots
            if 0 <= slot < self._ledger.capacity
        ]
        if not active_slots:
            return BMEStats(
                active_slots=0,
                universe_distribution={},
                role_counts={},
                trait_summary={},
                reward_summary={},
            )

        universe_distribution = self.population_universe_distribution(active_slots)
        role_counts = self.population_tier_distribution(active_slots)
        fitness_values = np.array(
            [self._ledger.get_fitness(slot) for slot in active_slots],
            dtype=np.float32,
        )
        gene_counts = np.array(
            [len(self._genome.active_genes(slot)) for slot in active_slots],
            dtype=np.float32,
        )
        genome_expression = np.array(
            [self._genome.genome_fitness_contribution(slot) for slot in active_slots],
            dtype=np.float32,
        )
        desire_matrix = np.array(
            [self._ledger.get_desires(slot).astype(np.float32) for slot in active_slots],
            dtype=np.float32,
        )

        trait_summary = {
            "mean_active_genes": round(float(gene_counts.mean()), 2),
            "max_active_genes": int(gene_counts.max()) if gene_counts.size else 0,
            "mean_genome_expression": round(float(genome_expression.mean()), 4),
            "mean_desires": {
                name: round(float(val), 4)
                for name, val in zip(DESIRE_NAMES, desire_matrix.mean(axis=0))
            },
        }
        reward_summary = {
            "mean_fitness": round(float(fitness_values.mean()), 4),
            "max_fitness": round(float(fitness_values.max()), 4),
            "min_fitness": round(float(fitness_values.min()), 4),
            "std_fitness": round(float(fitness_values.std()), 4),
        }

        return BMEStats(
            active_slots=len(active_slots),
            universe_distribution=universe_distribution,
            role_counts=role_counts,
            trait_summary=trait_summary,
            reward_summary=reward_summary,
        )

    def flagship_profiles(self) -> Dict[str, Any]:
        species = ("DataMycologist", "QuantumArchitect", "CosmicEngineer")
        return {
            "profiles": {
                name: shape_flagship_profile(self._engine.flagship_species_profile(name), name)
                for name in species
            }
        }

    def flagship_profile(self, species_name: str) -> Dict[str, Any]:
        profile = self._engine.flagship_species_profile(species_name)
        return {
            "species": species_name,
            "found": bool(profile),
            "preferred_universe": self._engine.preferred_universe_for_species(species_name),
            "profile": shape_flagship_profile(profile, species_name),
        }


# ── Module-level singleton ─────────────────────────────────────────────────────
_bridge: Optional[BMEBridge] = None


def get_bme_bridge(ledger: Optional[ChronosLedger] = None) -> BMEBridge:
    global _bridge
    if _bridge is None:
        _bridge = BMEBridge(ledger=ledger)
    return _bridge
