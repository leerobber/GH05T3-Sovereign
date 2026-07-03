"""GH05T3 — Sentinel Gate: economic viability guard for the KAIROS accept path.

The Sentinel Equation (V_E):
    V_E = (Y_d × P_e) / (C_c + D_ε) × S_human

    Y_d     — yield_discovery:  utility of the task output     (SAGE score proxy)
    P_e     — prob_of_evidence: agent confidence / trajectory  (SAGE score proxy)
    C_c     — compute_cost:     tokens × cost_per_token        (normalized USD)
    D_ε     — entropy_drift:    deviation from baseline intent (cosine distance)
    S_human — human_sig:        0 = kill switch (hard block), 1 = authorized

Authorization gate:
    human_sig == 1 AND V_E >= SENTINEL_THRESHOLD (default 0.65)

Wire-in: SAGE.evaluate() calls evaluate_cycle() after scoring; KAIROS records
the viability score alongside each cycle for long-term trend analysis.

Env vars:
    SENTINEL_THRESHOLD        minimum V_E to pass     (default 0.65)
    SENTINEL_COST_PER_TOKEN   cost proxy per token    (default 0.000001)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

LOG = logging.getLogger("ghost.sentinel")

SENTINEL_THRESHOLD = float(os.environ.get("SENTINEL_THRESHOLD", "0.65"))
COST_PER_TOKEN     = float(os.environ.get("SENTINEL_COST_PER_TOKEN", "0.000001"))


@dataclass
class SentinelGate:
    """Immutable snapshot of one cycle's viability inputs."""
    yield_discovery:  float   # Y_d
    prob_of_evidence: float   # P_e
    compute_cost:     float   # C_c
    entropy_drift:    float   # D_ε
    human_sig:        int     # S_human ∈ {0, 1}

    def economic_viability(self) -> float:
        """V_E = (Y_d × P_e) / max(C_c + D_ε, ε) × S_human"""
        numerator   = self.yield_discovery * self.prob_of_evidence
        denominator = self.compute_cost + self.entropy_drift
        return (numerator / max(denominator, 1e-9)) * self.human_sig

    def is_authorized(self, threshold: float = SENTINEL_THRESHOLD) -> bool:
        return self.human_sig == 1 and self.economic_viability() >= threshold


def evaluate_cycle(
    sage_score:    float,
    response:      str,
    entropy_drift: float = 0.0,
    human_sig:     int   = 1,
    threshold:     float = SENTINEL_THRESHOLD,
) -> dict:
    """Build a SentinelGate from SAGE output and return the evaluation result.

    Args:
        sage_score:    SAGE quality score  — used as Y_d and P_e proxy
        response:      generated text      — token count drives C_c
        entropy_drift: cosine distance from agent's baseline embedding
        human_sig:     1 = normal, 0 = kill switch engaged
        threshold:     minimum V_E required for PASS verdict

    Returns dict with all components for KAIROS logging and dashboard display.
    """
    token_count = len(response.split())
    cost        = token_count * COST_PER_TOKEN

    gate = SentinelGate(
        yield_discovery  = sage_score,
        prob_of_evidence = sage_score,
        compute_cost     = cost,
        entropy_drift    = entropy_drift,
        human_sig        = human_sig,
    )

    viability  = gate.economic_viability()
    authorized = gate.is_authorized(threshold)

    if not authorized:
        if human_sig == 0:
            LOG.warning("[sentinel] BLOCKED by kill switch (human_sig=0)")
        else:
            LOG.warning(
                "[sentinel] BLOCKED — V_E=%.4f below threshold=%.2f "
                "(Y_d=%.3f, drift=%.4f, cost=%.6f)",
                viability, threshold, sage_score, entropy_drift, cost,
            )

    return {
        "authorized":        authorized,
        "viability":         round(viability, 4),
        "threshold":         threshold,
        "yield_discovery":   round(gate.yield_discovery, 4),
        "prob_of_evidence":  round(gate.prob_of_evidence, 4),
        "compute_cost":      round(cost, 8),
        "entropy_drift":     round(entropy_drift, 4),
        "human_sig":         human_sig,
    }
