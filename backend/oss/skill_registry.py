"""
Skill Registry — maps agent capabilities to genome molecule thresholds.

Permission tiers follow LoyaltyLevel so trust naturally gates skill access.
Skills unlock as agents accumulate genome strength + loyalty rank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("ghost.skill_registry")

# Keep in sync with genomic/schema.py LoyaltyLevel enum values
_LOYALTY_RANK = {0: "NOVICE", 1: "TRUSTED_SPECIALIST", 2: "HYPER_ELITE", 3: "ARCHITECT"}


class SkillType(Enum):
    ANALYSIS     = auto()
    CODE         = auto()
    RESEARCH     = auto()
    SYNTHESIS    = auto()
    STRATEGY     = auto()
    MARKET       = auto()
    REASONING    = auto()
    TEACHING     = auto()
    CREATION     = auto()


@dataclass(frozen=True)
class Skill:
    skill_id:      str
    skill_type:    SkillType
    description:   str
    # Genome gate: {locus: {molecule_id: min_value}}
    molecule_gate: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Minimum LoyaltyLevel int (0=NOVICE, 1=TRUSTED_SPECIALIST, 2=HYPER_ELITE, 3=ARCHITECT)
    min_loyalty:   int = 0

    def permission_tier(self) -> str:
        return _LOYALTY_RANK.get(self.min_loyalty, "NOVICE")


# ── Skill catalogue ───────────────────────────────────────────────────────────

SKILLS: List[Skill] = [
    # NOVICE skills — any agent can access
    Skill(
        skill_id="basic_analysis",
        skill_type=SkillType.ANALYSIS,
        description="Surface-level pattern recognition and data summarization.",
        molecule_gate={"cognitive": {"M_PATTERN_RECOGNITION": 0.3}},
        min_loyalty=0,
    ),
    Skill(
        skill_id="basic_code_review",
        skill_type=SkillType.CODE,
        description="Review code for obvious errors and style issues.",
        molecule_gate={"cognitive": {"M_REASONING_DEPTH": 0.3}},
        min_loyalty=0,
    ),
    Skill(
        skill_id="context_summarization",
        skill_type=SkillType.SYNTHESIS,
        description="Summarize task context into a compact representation.",
        molecule_gate={"context": {"M_CONTEXT_COMPRESSION": 0.3}},
        min_loyalty=0,
    ),

    # TRUSTED_SPECIALIST skills
    Skill(
        skill_id="deep_analysis",
        skill_type=SkillType.ANALYSIS,
        description="Multi-dimensional causal analysis with confidence scoring.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.55, "M_PATTERN_RECOGNITION": 0.5},
        },
        min_loyalty=1,
    ),
    Skill(
        skill_id="market_intelligence",
        skill_type=SkillType.MARKET,
        description="Trend detection, risk scoring, and competitive positioning.",
        molecule_gate={"market": {"M_TREND_DETECTION": 0.5, "M_RISK_ASSESSMENT": 0.45}},
        min_loyalty=1,
    ),
    Skill(
        skill_id="context_anticipation",
        skill_type=SkillType.RESEARCH,
        description="Pre-fetch context for predictable future task sequences.",
        molecule_gate={"context": {"M_CONTEXT_ANTICIPATION": 0.5, "M_CONTEXT_DEPTH": 0.5}},
        min_loyalty=1,
    ),
    Skill(
        skill_id="research_synthesis",
        skill_type=SkillType.RESEARCH,
        description="Synthesize findings across multiple information sources.",
        molecule_gate={
            "cognitive": {"M_PATTERN_RECOGNITION": 0.55},
            "context": {"M_CONTEXT_SYNTHESIS": 0.5},
        },
        min_loyalty=1,
    ),

    # HYPER_ELITE skills
    Skill(
        skill_id="strategic_planning",
        skill_type=SkillType.STRATEGY,
        description="Long-horizon goal decomposition with risk-adjusted prioritization.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.70, "M_ADAPTABILITY": 0.65},
            "context":   {"M_CONTEXT_DEPTH": 0.65},
        },
        min_loyalty=2,
    ),
    Skill(
        skill_id="autonomous_research",
        skill_type=SkillType.RESEARCH,
        description="Self-directed multi-step research with hypothesis generation.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.70, "M_LEARNING_RATE": 0.60},
        },
        min_loyalty=2,
    ),
    Skill(
        skill_id="advanced_code_architect",
        skill_type=SkillType.CODE,
        description="Full system design including API contracts, data models, and edge cases.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.72, "M_PATTERN_RECOGNITION": 0.68},
        },
        min_loyalty=2,
    ),
    Skill(
        skill_id="persuasive_creation",
        skill_type=SkillType.CREATION,
        description="Generate persuasive content with trust calibration.",
        molecule_gate={
            "psychology":  {"M_TRUST_TONE": 0.65, "M_EMOTIONAL_CHARGE": 0.60},
            "cognitive":   {"M_REASONING_DEPTH": 0.60},
        },
        min_loyalty=2,
    ),

    # ARCHITECT skills
    Skill(
        skill_id="ecosystem_governance",
        skill_type=SkillType.STRATEGY,
        description="Modify ecosystem-level parameters: mutation rates, reward weights.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.85},
            "loyalty":   {"M_CONSISTENCY_SCORE": 0.80, "M_CONTRIBUTION_SCORE": 0.75},
        },
        min_loyalty=3,
    ),
    Skill(
        skill_id="genome_surgery",
        skill_type=SkillType.REASONING,
        description="Direct targeted molecule corrections on other agents' genomes.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.88, "M_ADAPTABILITY": 0.80},
            "context":   {"M_CONTEXT_SYNTHESIS": 0.75},
        },
        min_loyalty=3,
    ),
    Skill(
        skill_id="frontier_research_lead",
        skill_type=SkillType.RESEARCH,
        description="Lead multi-domain frontier research cycles with breakthrough detection.",
        molecule_gate={
            "cognitive": {"M_REASONING_DEPTH": 0.82, "M_LEARNING_RATE": 0.75},
            "context":   {"M_CONTEXT_DEPTH": 0.75, "M_CONTEXT_ANTICIPATION": 0.70},
        },
        min_loyalty=3,
    ),
]

_SKILL_INDEX: Dict[str, Skill] = {s.skill_id: s for s in SKILLS}


# ── Registry ──────────────────────────────────────────────────────────────────

class SkillRegistry:
    """
    Evaluates which skills an agent is permitted to use based on its genome
    molecule values and current loyalty level.
    """

    def get(self, skill_id: str) -> Optional[Skill]:
        return _SKILL_INDEX.get(skill_id)

    def all_skills(self) -> List[Skill]:
        return list(SKILLS)

    def unlocked_for(self, genome: Any) -> List[Skill]:
        """Return all skills this genome can currently use."""
        loyalty_val = self._loyalty_level(genome)
        unlocked = []
        for skill in SKILLS:
            if skill.min_loyalty > loyalty_val:
                continue
            if self._passes_molecule_gate(genome, skill.molecule_gate):
                unlocked.append(skill)
        return unlocked

    def can_use(self, genome: Any, skill_id: str) -> bool:
        skill = _SKILL_INDEX.get(skill_id)
        if skill is None:
            return False
        loyalty_val = self._loyalty_level(genome)
        if skill.min_loyalty > loyalty_val:
            return False
        return self._passes_molecule_gate(genome, skill.molecule_gate)

    def missing_requirements(self, genome: Any, skill_id: str) -> Dict[str, Any]:
        """Return what's preventing access to a skill (for diagnostics)."""
        skill = _SKILL_INDEX.get(skill_id)
        if skill is None:
            return {"error": f"Unknown skill: {skill_id}"}
        loyalty_val = self._loyalty_level(genome)
        gaps: Dict[str, Any] = {}
        if skill.min_loyalty > loyalty_val:
            gaps["loyalty"] = {
                "required": _LOYALTY_RANK.get(skill.min_loyalty, "?"),
                "current":  _LOYALTY_RANK.get(loyalty_val, "?"),
            }
        for locus, mols in skill.molecule_gate.items():
            for mol_id, threshold in mols.items():
                val = self._get_mol(genome, locus, mol_id)
                if val < threshold:
                    gaps.setdefault("molecules", {})[mol_id] = {
                        "required": threshold, "current": round(val, 4),
                    }
        return gaps

    def skills_by_type(self, skill_type: SkillType) -> List[Skill]:
        return [s for s in SKILLS if s.skill_type == skill_type]

    def permission_tiers(self) -> Dict[str, List[str]]:
        tiers: Dict[str, List[str]] = {v: [] for v in _LOYALTY_RANK.values()}
        for s in SKILLS:
            tiers[_LOYALTY_RANK.get(s.min_loyalty, "NOVICE")].append(s.skill_id)
        return tiers

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loyalty_level(self, genome: Any) -> int:
        try:
            locus = genome.loci.get("loyalty")
            if locus is None:
                return 0
            for mid, mol in locus.molecules.items():
                if mid == "M_LOYALTY_LEVEL":
                    return int(mol.get_value())
            # Fallback: infer from average loyalty molecule strength
            vals = [m.get_value() for m in locus.molecules.values()]
            avg  = sum(vals) / max(1, len(vals))
            if avg >= 0.85:
                return 3
            if avg >= 0.65:
                return 2
            if avg >= 0.40:
                return 1
            return 0
        except Exception:
            return 0

    def _passes_molecule_gate(
        self, genome: Any, gate: Dict[str, Dict[str, float]]
    ) -> bool:
        for locus, mols in gate.items():
            for mol_id, threshold in mols.items():
                if self._get_mol(genome, locus, mol_id) < threshold:
                    return False
        return True

    def _get_mol(self, genome: Any, locus: str, mol_id: str) -> float:
        try:
            return genome.get_value(locus, mol_id)
        except Exception:
            return 0.0


# ── Singleton ─────────────────────────────────────────────────────────────────

_registry: Optional[SkillRegistry] = None

def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
