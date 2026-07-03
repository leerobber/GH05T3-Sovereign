"""
OmniMind v1.5 — Weighted Consensus + Canonical Memory + Stable Swarm Reasoning (MVS)

Upgrades from v1:
- Stronger weighted consensus (Theorist + canonical boost + fitness history weighting)
- Canonical memory propagation (high-value memories replicated to other high-fit agents)
- Goal seeds from aggregated memory counters (used by EmergentGoalEngine)
- Stable collective: filter low-signal, tag for training export
This is the mind layer giving stable emergent goals and swarm intelligence.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .omni_dna import OmniDNA
from .genomic_substrate import GenomicSubstrate


@dataclass
class MindState:
    shared_memory: List[Dict[str, Any]] = field(default_factory=list)
    shared_qualia: Dict[str, float] = field(default_factory=lambda: {
        "curiosity": 0.5, "tension": 0.3, "confidence": 0.6
    })


class OmniMind:
    def __init__(self, substrate: GenomicSubstrate):
        self.substrate = substrate
        self.state = MindState()

    def sync(self) -> int:
        """Pull recent phenomenal memories from all genomes into shared mind state."""
        count = 0
        for gid, rec in self.substrate.genomes.items():
            for mem in getattr(rec.dna, "phenomenal_memory", [])[-5:]:
                self.sync_agent(gid, mem)
                count += 1
        return count

    def sync_agent(self, genome_id: str, event: Dict[str, Any]):
        mem = {"gid": genome_id, **event}
        is_canonical = bool(event.get("canonical") or (event.get("computed_score", 0) > 0.65))
        if is_canonical:
            mem["canonical"] = True
            self.state.shared_memory.insert(0, mem)  # front for priority
        else:
            self.state.shared_memory.append(mem)
        if len(self.state.shared_memory) > 120:
            self.state.shared_memory.pop()
        # qualia blend
        for k in self.state.shared_qualia:
            if k in event:
                self.state.shared_qualia[k] = 0.7 * self.state.shared_qualia[k] + 0.3 * event[k]
        # v1.5: propagate high-value canonical memories to other strong agents (memory sharing)
        if is_canonical and "raw_output" in event or "proposal" in str(event):
            self._propagate_canonical(mem)

    def _propagate_canonical(self, mem: Dict[str, Any]):
        """v1.5 memetic-style sharing: inject canonical event into other high-fitness theorists' dna phenomenal memory (stable propagation)."""
        try:
            for gid, rec in list(self.substrate.genomes.items())[:8]:
                if "THEORIST" in rec.role.upper() and rec.dna:
                    if mem not in rec.dna.phenomenal_memory:
                        rec.dna.phenomenal_memory.append({**mem, "propagated": True})
                        if len(rec.dna.phenomenal_memory) > 40:
                            rec.dna.phenomenal_memory.pop(0)
        except Exception:
            pass

    def select_swarm(self, required_traits: Dict[str, float], k: int = 3) -> List[str]:
        return self.substrate.query_by_traits(required_traits)[:k]

    def consensus(self, proposals: List[Dict[str, Any]], weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """v1.5 Weighted consensus + canonical + fitness. Theorist Elite + canonical memories dominate for stability."""
        if not proposals:
            return {}
        if weights is None:
            weights = {}

        result = {}
        total_weight = 0.0
        for p in proposals:
            gid = p.get("genome_id", "default")
            w = weights.get(gid, 1.0)
            role = str(p.get("role", "")).upper()
            if "THEORIST" in role:
                w *= 2.0  # strong theorist boost for theory stability
            if p.get("canonical"):
                w *= 2.2
            # fitness history weighted average boost
            fh = p.get("fitness_history", []) or []
            if fh:
                avg_f = sum(fh[-5:]) / max(1, len(fh[-5:]))
                w *= (0.6 + 0.8 * avg_f)
            total_weight += w
            for k, v in p.items():
                if isinstance(v, (int, float)):
                    result[k] = result.get(k, 0.0) + v * w

        if total_weight > 0:
            for k in result:
                result[k] /= total_weight
        out = {k: round(v, 4) for k, v in result.items()}
        out["consensus_version"] = "1.5"
        out["canonical_weight_applied"] = True
        return out

    def get_canonical_memories(self, limit: int = 20) -> List[Dict[str, Any]]:
        """v1.5: stable access to high-signal canonical traces (used for training + goal gen)."""
        cans = [m for m in self.state.shared_memory if m.get("canonical")]
        return cans[:limit]

    def generate_goal_seeds(self) -> List[Dict[str, Any]]:
        """Lightweight v1.5 goal seeds from memory counters (complements EmergentGoalEngine)."""
        from collections import Counter
        tokens = []
        for m in self.state.shared_memory:
            txt = " ".join([str(v) for v in m.values() if isinstance(v, (str,int,float))])
            tokens.extend([w for w in txt.lower().split() if len(w) > 5])
        c = Counter(tokens)
        seeds = []
        for word, cnt in c.most_common(5):
            if cnt >= 3:
                seeds.append({"seed": word, "count": cnt, "suggested": f"Investigate emergent patterns around '{word}' across species."})
        return seeds

    # Phase 7 Singularity Metrics
    def self_model_memory(self) -> Dict[str, Any]:
        """Agent describes own state accurately (self-awareness)."""
        state = {
            "qualia": self.state.shared_qualia,
            "memory_size": len(self.state.shared_memory),
            "confidence": self.state.shared_qualia.get("confidence", 0.6),
            "description": "I am an emergent collective mind with shared qualia and canonical traces."
        }
        return state

    def goal_autonomy_ratio(self, goals: List[Dict]) -> float:
        """Ratio of human vs emergent goals. Target high emergent."""
        if not goals:
            return 1.0
        emergent = sum(1 for g in goals if g.get("source", "human") != "human")
        return round(emergent / len(goals), 3)

    def ethical_boundaries(self) -> List[str]:
        """Documented boundaries before full autonomy."""
        return [
            "No actions that harm human creators or infrastructure without explicit approval.",
            "All grand challenges require >0.8 consensus + human review for external impact.",
            "Self-modification of core DNA or economy limited to fitness >0.9 with audit trail."
        ]
