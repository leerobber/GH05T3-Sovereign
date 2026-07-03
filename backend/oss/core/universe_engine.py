"""
UniverseEngine — 7-universe physics definitions for the Binary Multiverse Engine.

Each universe defines:
  - Desire amplification matrix (how each desire dimension is weighted for fitness)
  - Mutation operator overrides (mutation_rate, drift_sigma per universe)
  - Promotion criteria (when agents advance to higher role tiers)
  - Adjacency graph (which universes an agent can migrate to)

No LLM calls. No JSON parsing. All lookups are dict dereferences against
constants compiled at module import. Zero latency.

Desire dimension indices (matching DESIRE_NAMES in chronos_ledger.py):
  0 = SURVIVAL       5 = FREEDOM
  1 = KNOWLEDGE      6 = STATUS
  2 = CREATION
  3 = CONNECTION
  4 = INFLUENCE
(or your ecosystem's 7-dim desire vector — adjust indices as needed)

Universe IDs:
  0 = Physics
  1 = Chemistry
  2 = Biology
  3 = Psychology
  4 = Fungal
  5 = Cosmic
  6 = Entropy
  7 = Hybrid       (Meta-Genomic Governors — cross-universe synthesists)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Desire indices ─────────────────────────────────────────────────────────────
D_SURVIVAL   = 0
D_KNOWLEDGE  = 1
D_CREATION   = 2
D_CONNECTION = 3
D_INFLUENCE  = 4
D_FREEDOM    = 5
D_STATUS     = 6
NUM_DESIRES  = 7

DESIRE_NAMES = [
    "SURVIVAL", "KNOWLEDGE", "CREATION", "CONNECTION",
    "INFLUENCE", "FREEDOM", "STATUS",
]

# ── Universe IDs ───────────────────────────────────────────────────────────────
UNIVERSE_PHYSICS    = 0
UNIVERSE_CHEMISTRY  = 1
UNIVERSE_BIOLOGY    = 2
UNIVERSE_PSYCHOLOGY = 3
UNIVERSE_FUNGAL     = 4
UNIVERSE_COSMIC     = 5
UNIVERSE_ENTROPY    = 6
UNIVERSE_HYBRID     = 7

# ── Desire amplification matrices ──────────────────────────────────────────────
# How each universe weights the 7 desire dimensions for fitness computation.
# Values > 1.0 amplify; < 1.0 suppress; 1.0 = neutral.
# Order: [SURVIVAL, KNOWLEDGE, CREATION, CONNECTION, INFLUENCE, FREEDOM, STATUS]
DESIRE_AMPLIFICATION: Dict[int, np.ndarray] = {
    UNIVERSE_PHYSICS: np.array(
        [1.0, 2.5, 1.8, 0.6, 1.2, 1.0, 0.8], dtype=np.float32
    ),  # Physics: KNOWLEDGE × 2.5, CREATION × 1.8 (discovery / construction drive)
    UNIVERSE_CHEMISTRY: np.array(
        [1.0, 2.0, 2.2, 0.7, 1.1, 0.9, 0.9], dtype=np.float32
    ),  # Chemistry: CREATION × 2.2, KNOWLEDGE × 2.0 (synthesis focus)
    UNIVERSE_BIOLOGY: np.array(
        [2.0, 1.5, 1.5, 1.8, 0.8, 1.0, 0.7], dtype=np.float32
    ),  # Biology: SURVIVAL × 2.0, CONNECTION × 1.8 (mutualism / resilience)
    UNIVERSE_PSYCHOLOGY: np.array(
        [1.0, 1.3, 1.0, 2.5, 2.0, 0.8, 2.0], dtype=np.float32
    ),  # Psychology: CONNECTION × 2.5, STATUS × 2.0 (social dominance)
    UNIVERSE_FUNGAL: np.array(
        [1.5, 1.0, 1.2, 3.0, 1.0, 2.0, 0.6], dtype=np.float32
    ),  # Fungal: CONNECTION × 3.0, FREEDOM × 2.0 (distributed network growth)
    UNIVERSE_COSMIC: np.array(
        [0.8, 2.0, 1.5, 0.5, 1.0, 2.5, 1.5], dtype=np.float32
    ),  # Cosmic: FREEDOM × 2.5, KNOWLEDGE × 2.0 (horizon crossing, scale)
    UNIVERSE_ENTROPY: np.array(
        [1.2, 0.8, 2.0, 0.7, 1.5, 2.2, 0.5], dtype=np.float32
    ),  # Entropy: CREATION × 2.0, FREEDOM × 2.2 (chaos injection, exploration)
    UNIVERSE_HYBRID: np.array(
        [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32
    ),  # Hybrid: all dimensions amplified (Meta-Genomic Governors)
}

# ── Universe mutation operator overrides ────────────────────────────────────────
# Overrides MutationEngine defaults. Physics = precise; Entropy = chaotic.
UNIVERSE_MUTATION_PARAMS: Dict[int, Dict] = {
    UNIVERSE_PHYSICS:    {"mutation_rate": 0.02, "drift_sigma": 0.05, "intensity": 0.05},
    UNIVERSE_CHEMISTRY:  {"mutation_rate": 0.04, "drift_sigma": 0.08, "intensity": 0.07},
    UNIVERSE_BIOLOGY:    {"mutation_rate": 0.06, "drift_sigma": 0.10, "intensity": 0.08},
    UNIVERSE_PSYCHOLOGY: {"mutation_rate": 0.05, "drift_sigma": 0.09, "intensity": 0.07},
    UNIVERSE_FUNGAL:     {"mutation_rate": 0.08, "drift_sigma": 0.12, "intensity": 0.10},
    UNIVERSE_COSMIC:     {"mutation_rate": 0.03, "drift_sigma": 0.06, "intensity": 0.06},
    UNIVERSE_ENTROPY:    {"mutation_rate": 0.20, "drift_sigma": 0.25, "intensity": 0.20},
    UNIVERSE_HYBRID:     {"mutation_rate": 0.03, "drift_sigma": 0.07, "intensity": 0.06},
}

# ── Role tier promotion criteria ────────────────────────────────────────────────
# (min_fitness, min_generation, min_discovery_count) per tier gate.
# Agent is promoted from tier N to N+1 when ALL three thresholds are met.
PROMOTION_CRITERIA: Dict[int, Tuple[float, int, int]] = {
    0: (0.30,  3,  0),   # Base → Specialist
    1: (0.50,  8,  2),   # Specialist → Elite
    2: (0.65, 15,  4),   # Elite → Apex Synthesist
    3: (0.75, 25,  6),   # Apex → Quantum Architect (tier 4)
    4: (0.85, 40,  8),   # Tier 4 → String Theorist (tier 5)
    5: (0.92, 60, 10),   # Tier 5 → Meta-Genomic Governor (tier 6)
    6: (0.97, 90, 12),   # Tier 6 → Substrate Philosopher (tier 7)
}

# ── Universe adjacency graph ───────────────────────────────────────────────────
# Migration is only possible between adjacent universes. Hybrid is reachable from
# any universe (boundary-crossing) but only by tier >= 5 agents.
# Adjacency is symmetric; the dict gives neighbours in O(1).
UNIVERSE_ADJACENCY: Dict[int, List[int]] = {
    UNIVERSE_PHYSICS:    [UNIVERSE_CHEMISTRY, UNIVERSE_COSMIC,    UNIVERSE_ENTROPY, UNIVERSE_HYBRID],
    UNIVERSE_CHEMISTRY:  [UNIVERSE_PHYSICS,   UNIVERSE_BIOLOGY,   UNIVERSE_ENTROPY, UNIVERSE_HYBRID],
    UNIVERSE_BIOLOGY:    [UNIVERSE_CHEMISTRY, UNIVERSE_FUNGAL,    UNIVERSE_PSYCHOLOGY, UNIVERSE_HYBRID],
    UNIVERSE_PSYCHOLOGY: [UNIVERSE_BIOLOGY,   UNIVERSE_FUNGAL,    UNIVERSE_COSMIC,  UNIVERSE_HYBRID],
    UNIVERSE_FUNGAL:     [UNIVERSE_BIOLOGY,   UNIVERSE_PSYCHOLOGY,UNIVERSE_ENTROPY, UNIVERSE_HYBRID],
    UNIVERSE_COSMIC:     [UNIVERSE_PHYSICS,   UNIVERSE_PSYCHOLOGY,UNIVERSE_ENTROPY, UNIVERSE_HYBRID],
    UNIVERSE_ENTROPY:    [UNIVERSE_PHYSICS,   UNIVERSE_CHEMISTRY, UNIVERSE_FUNGAL,  UNIVERSE_COSMIC, UNIVERSE_HYBRID],
    UNIVERSE_HYBRID:     [0, 1, 2, 3, 4, 5, 6],   # fully connected (entry requires tier ≥ 5)
}

HYBRID_MIN_TIER = 5   # minimum tier to enter Hybrid universe

# ── Flagship species tuning ───────────────────────────────────────────────────
FLAGSHIP_SPECIES_PROFILES: Dict[str, Dict[str, object]] = {
    "DataMycologist": {
        "preferred_universe": UNIVERSE_FUNGAL,
        "desire_bias": np.array([1.10, 1.05, 0.95, 1.45, 0.95, 1.30, 0.85], dtype=np.float32),
        "mutation_bias": {
            "network_density": 0.06,
            "symbiosis": 0.05,
            "attention_molecules": 0.04,
            "entropy_resilience": 0.03,
        },
        "role_bias": "Data Mycologist",
    },
    "QuantumArchitect": {
        "preferred_universe": UNIVERSE_PHYSICS,
        "desire_bias": np.array([0.95, 1.45, 1.20, 0.90, 1.20, 1.00, 0.95], dtype=np.float32),
        "mutation_bias": {
            "math": 0.07,
            "pattern_detection": 0.06,
            "rigor": 0.05,
            "creativity": 0.03,
        },
        "role_bias": "Quantum Architect",
    },
    "CosmicEngineer": {
        "preferred_universe": UNIVERSE_COSMIC,
        "desire_bias": np.array([0.85, 1.30, 1.05, 0.85, 1.10, 1.45, 1.00], dtype=np.float32),
        "mutation_bias": {
            "scale_reasoning": 0.06,
            "horizon_crossing": 0.05,
            "resonance": 0.04,
        },
        "role_bias": "Cosmic Engineer",
    },
}


class UniverseEngine:
    """
    Pure-data engine for multi-universe physics.

    All methods are pure functions over numpy arrays — no I/O, no state.
    Instantiate once as a singleton. Zero overhead: all heavy data structures
    (amplification matrices, adjacency graph) are module-level constants.
    """

    # ── Fitness ───────────────────────────────────────────────────────────────

    def amplified_fitness(
        self,
        universe_id: int,
        desires: np.ndarray,   # float32 (7,) or compatible
        base_fitness: float,
    ) -> float:
        """
        Universe-specific fitness: dot product of desires × amplification matrix,
        weighted by base_fitness. Higher amplification for aligned desires = higher
        effective fitness in that universe.

        desire_score = dot(desires_norm, amp) in [0, max_amp]
        result       = base_fitness × (0.5 + 0.5 × desire_score / max_possible)
        """
        amp = DESIRE_AMPLIFICATION.get(universe_id, np.ones(NUM_DESIRES, dtype=np.float32))
        d   = desires[:NUM_DESIRES].astype(np.float32) if len(desires) >= NUM_DESIRES \
              else np.pad(desires.astype(np.float32), (0, NUM_DESIRES - len(desires)))
        d_norm     = d / (np.linalg.norm(d) + 1e-8)
        desire_score = float(np.dot(d_norm, amp))
        max_score    = float(np.max(amp))   # theoretical maximum
        norm_score   = desire_score / (max_score + 1e-8)
        return float(base_fitness * (0.5 + 0.5 * norm_score))

    @classmethod
    def load(cls) -> "UniverseEngine":
        """Compatibility helper for router-style callers."""
        return get_universe_engine()

    def compare_universe_fitness(
        self,
        desires: np.ndarray,
        base_fitness: float,
    ) -> Dict[int, float]:
        """Compute amplified fitness across all universes. O(8×7) = O(56)."""
        return {
            uid: self.amplified_fitness(uid, desires, base_fitness)
            for uid in range(UNIVERSE_HYBRID + 1)
        }

    # ── Migration ─────────────────────────────────────────────────────────────

    def migration_candidate(
        self,
        current_universe: int,
        desires: np.ndarray,
        base_fitness: float,
        delta_gate: float = 0.15,
        agent_tier: int = 0,
    ) -> Optional[int]:
        """
        Find best adjacent universe for migration.
        Returns universe_id if improvement > delta_gate, else None.
        Hybrid universe requires agent_tier >= HYBRID_MIN_TIER.
        """
        current_fit = self.amplified_fitness(current_universe, desires, base_fitness)
        neighbors   = UNIVERSE_ADJACENCY.get(current_universe, [])
        best_uid    = None
        best_gain   = 0.0

        for uid in neighbors:
            if uid == UNIVERSE_HYBRID and agent_tier < HYBRID_MIN_TIER:
                continue
            fit  = self.amplified_fitness(uid, desires, base_fitness)
            gain = fit - current_fit
            if gain > best_gain:
                best_gain = gain
                best_uid  = uid

        if best_gain >= delta_gate:
            return best_uid
        return None

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promotion_criteria(
        self,
        current_tier: int,
        fitness: float,
        generation: int,
        discovery_count: int,
    ) -> bool:
        """Return True if agent meets all promotion gates for next tier."""
        if current_tier >= 7:
            return False
        min_fit, min_gen, min_disc = PROMOTION_CRITERIA.get(
            current_tier, (1.0, 999, 999)
        )
        return (
            fitness          >= min_fit and
            generation       >= min_gen and
            discovery_count  >= min_disc
        )

    # ── Universe-specific mutation bias ───────────────────────────────────────

    def universe_mutation_vector(
        self,
        universe_id: int,
        desires: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Generate a universe-biased mutation vector for desire dimensions.

        Mutation is biased toward the amplification maxima of the universe —
        i.e. agents in Physics universe are more likely to shift toward
        higher KNOWLEDGE/CREATION desires when mutating.

        Returns float16 delta vector (7,).
        """
        params = UNIVERSE_MUTATION_PARAMS.get(universe_id, {"drift_sigma": 0.10})
        sigma  = params["drift_sigma"]
        amp    = DESIRE_AMPLIFICATION.get(universe_id, np.ones(NUM_DESIRES, dtype=np.float32))

        # Base Gaussian noise
        base_noise = rng.normal(0.0, sigma, size=NUM_DESIRES).astype(np.float32)
        # Bias: small additional push toward amplified dimensions
        amp_norm = amp / (amp.max() + 1e-8)
        bias     = amp_norm * sigma * 0.3 * rng.choice([-1, 1])
        delta    = (base_noise + bias).astype(np.float16)
        return delta

    # ── Stats ─────────────────────────────────────────────────────────────────

    def universe_summary(self, universe_id: int) -> Dict:
        amp    = DESIRE_AMPLIFICATION.get(universe_id, np.ones(NUM_DESIRES))
        params = UNIVERSE_MUTATION_PARAMS.get(universe_id, {})
        crit   = PROMOTION_CRITERIA
        return {
            "universe_id":   universe_id,
            "name":          {0:"Physics",1:"Chemistry",2:"Biology",3:"Psychology",
                              4:"Fungal",5:"Cosmic",6:"Entropy",7:"Hybrid"}.get(universe_id, "?"),
            "amplification": {DESIRE_NAMES[i]: float(amp[i]) for i in range(len(amp))},
            "mutation_rate": params.get("mutation_rate"),
            "drift_sigma":   params.get("drift_sigma"),
            "adjacency":     UNIVERSE_ADJACENCY.get(universe_id, []),
            "promotion_gates": {
                f"tier_{t}_to_{t+1}": {"min_fitness": v[0], "min_gen": v[1], "min_disc": v[2]}
                for t, v in crit.items()
            },
        }

    def flagship_species_profile(self, species_name: str) -> Dict[str, object]:
        """Return the tuning profile for a targeted flagship species."""
        return dict(FLAGSHIP_SPECIES_PROFILES.get(species_name, {}))

    def preferred_universe_for_species(self, species_name: str) -> int:
        """Return the flagship species' preferred universe, defaulting to Hybrid."""
        profile = FLAGSHIP_SPECIES_PROFILES.get(species_name, {})
        return int(profile.get("preferred_universe", UNIVERSE_HYBRID))

    def apply_species_desire_bias(self, species_name: str, desires: np.ndarray) -> np.ndarray:
        """Apply a flagship species desire bias without mutating the input."""
        profile = FLAGSHIP_SPECIES_PROFILES.get(species_name, {})
        bias = profile.get("desire_bias")
        if bias is None:
            return np.clip(desires.astype(np.float32), 0.0, 1.0)
        arr = desires[:NUM_DESIRES].astype(np.float32)
        arr = np.clip(arr * bias, 0.0, 1.0)
        return arr


# ── Singleton ─────────────────────────────────────────────────────────────────
_engine: Optional[UniverseEngine] = None


def get_universe_engine() -> UniverseEngine:
    global _engine
    if _engine is None:
        _engine = UniverseEngine()
    return _engine
