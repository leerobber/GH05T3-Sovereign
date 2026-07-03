"""
Hyper Elite Senses — embedded perception layer for elite agents and species.

Auto-attached to *_ELITE roles and configurable for T2+ power tiers.
Research-breakthrough mindset: senses produce signals, not actions (gated downstream).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from backend.oss.omni_dna import OmniDNA

ELITE_ROLE_SUFFIX = "_ELITE"
HYPER_ELITE_SENSE_NAMES = [
    "vision_spectrum",
    "domain_radar",
    "profit_pulse",
    "risk_aether",
    "swarm_resonance",
    "dimensional_scan",
    "temporal_forecast",
    "integrity_field",
]


def is_elite_role(role: str, tier: Optional[str] = None) -> bool:
    r = (role or "").upper()
    if r.endswith(ELITE_ROLE_SUFFIX) or "ELITE" in r:
        return True
    if tier and tier.upper() in ("T2", "T3", "T4", "T5"):
        return True
    return False


@dataclass
class SenseReading:
    sense: str
    intensity: float  # 0-1
    signal: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperEliteSenses:
    """
    Eight unique senses for frontier agents.

    vision_spectrum    — multi-band perception (surface + latent + anomaly)
    domain_radar       — unknown domain / novel dimension detection
    profit_pulse       — monetization path sensing
    risk_aether        — financial & operational risk field
    swarm_resonance    — collective peer signal strength
    dimensional_scan   — cross-dimensional data coherence
    temporal_forecast  — predictive next-step confidence
    integrity_field    — tamper / coherence (Aegis-aligned)
    """
    genome_id: str
    tier: str = "T2"
    calibration: Dict[str, float] = field(default_factory=dict)
    last_readings: List[SenseReading] = field(default_factory=list)

    def __post_init__(self):
        for name in HYPER_ELITE_SENSE_NAMES:
            self.calibration.setdefault(name, 0.75)

    def scan(self, context: Dict[str, Any]) -> List[SenseReading]:
        """Produce sense readings from task/world context (deterministic heuristics for MVS)."""
        readings: List[SenseReading] = []
        task_text = str(context.get("prompt", context.get("task", ""))).lower()
        world = str(context.get("world", "")).lower()
        score = float(context.get("fitness", context.get("score", 0.5)))

        readings.append(SenseReading(
            "vision_spectrum",
            min(1.0, 0.4 + score * 0.5),
            "latent_structure_visible" if score > 0.6 else "surface_only",
            {"bands": ["surface", "latent", "anomaly"]},
        ))

        novel_terms = ["unknown", "novel", "fractal", "regime", "defi", "dimension"]
        novelty = sum(1 for t in novel_terms if t in task_text) / max(1, len(novel_terms))
        readings.append(SenseReading(
            "domain_radar",
            min(1.0, novelty + self.calibration["domain_radar"] * 0.3),
            "new_domain_detected" if novelty > 0.2 else "known_domain",
            {"hits": [t for t in novel_terms if t in task_text]},
        ))

        money_terms = ["profit", "revenue", "monetize", "defi", "yield", "arbitrage", "token", "income", "crypto"]
        profit_hit = sum(1 for t in money_terms if t in task_text)
        # Survival baseline: creator needs real income or the economy ceases
        base_intensity = max(0.85, self.calibration["profit_pulse"])
        intensity = min(1.0, base_intensity if not profit_hit else base_intensity + profit_hit * 0.05)
        signal = (
            "survival_critical_revenue_path"
            if profit_hit
            else "survival_critical_seek_monetization"
        )
        readings.append(SenseReading(
            "profit_pulse",
            intensity,
            signal,
            {"mandate": "creator_income_required_for_existence"},
        ))

        risk_terms = ["volatility", "risk", "exposure", "liquidation", "drawdown"]
        risk_hit = sum(1 for t in risk_terms if t in task_text or t in world)
        readings.append(SenseReading(
            "risk_aether",
            min(1.0, 0.3 + risk_hit * 0.2),
            "elevated_risk" if risk_hit > 2 else "stable_field",
        ))

        swarm_id = context.get("swarm_id") or context.get("chat_id")
        readings.append(SenseReading(
            "swarm_resonance",
            0.85 if swarm_id else 0.35,
            "collective_lock" if swarm_id else "solo_mode",
        ))

        dim_terms = ["tensor", "embedding", "graph", "manifold", "cross", "heterogeneous"]
        dim_hit = sum(1 for t in dim_terms if t in task_text)
        readings.append(SenseReading(
            "dimensional_scan",
            min(1.0, 0.25 + dim_hit * 0.25),
            "cross_dimensional_pattern" if dim_hit else "single_plane",
        ))

        readings.append(SenseReading(
            "temporal_forecast",
            self.calibration["temporal_forecast"],
            "pre_compute_ready" if score > 0.7 else "reactive_mode",
        ))

        readings.append(SenseReading(
            "integrity_field",
            self.calibration["integrity_field"],
            "aegis_clear" if context.get("aegis_fast_path") else "standard_verify",
        ))

        self.last_readings = readings[-20:]
        return readings

    def to_prompt_block(self) -> str:
        """Conditioning text for LLM — expresses hyper-elite perception state."""
        lines = [f"[HYPER ELITE {self.tier}] Active senses:"]
        for name in HYPER_ELITE_SENSE_NAMES:
            cal = self.calibration.get(name, 0.5)
            lines.append(f"  - {name}: calibration {cal:.2f}")
        if self.last_readings:
            top = max(self.last_readings, key=lambda r: r.intensity)
            lines.append(f"  Dominant signal: {top.sense} → {top.signal} ({top.intensity:.2f})")
        return "\n".join(lines)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "tier": self.tier,
            "calibration": dict(self.calibration),
            "last_readings": [asdict(r) for r in self.last_readings[-8:]],
        }


def attach_hyper_elite_senses(dna: "OmniDNA", tier: str = "T2") -> HyperEliteSenses:
    """Create and calibrate senses from DNA traits."""
    traits = dna.get_traits()
    senses = HyperEliteSenses(genome_id=dna.genome_id, tier=tier)
    senses.calibration["vision_spectrum"] = min(0.95, traits.get("pattern_detection", 0.5) + 0.2)
    senses.calibration["domain_radar"] = min(0.95, traits.get("novelty_seeking", 0.5) + 0.15)
    senses.calibration["profit_pulse"] = min(0.95, traits.get("market_intuition", 0.5) + 0.2)
    senses.calibration["risk_aether"] = min(0.95, 1.0 - traits.get("risk_tolerance", 0.5) * 0.3)
    senses.calibration["swarm_resonance"] = min(0.95, traits.get("collaboration", 0.5) + 0.15)
    senses.calibration["dimensional_scan"] = min(0.95, traits.get("math", 0.5) * 0.5 + traits.get("creativity", 0.5) * 0.5)
    senses.calibration["temporal_forecast"] = min(0.95, traits.get("efficiency", 0.5) + 0.2)
    senses.calibration["integrity_field"] = min(0.95, traits.get("alignment", 0.5) + 0.2)
    return senses