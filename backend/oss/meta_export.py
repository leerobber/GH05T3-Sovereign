"""
Meta-evolution data exporter (MVS only path)

Collects from the single source of truth:
- traits
- fitness
- memories (phenomenal)
- neurocoins
- mutations

This data trains the next version of the species.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

from .log_config import get_logger

log = get_logger(__name__)

REQUIRED_META_FIELDS = frozenset({
    "genome_id", "role", "traits", "fitness_history", "neurocoins", "recent_memories",
})

REQUIRED_GENOMIC_FIELDS = frozenset({
    "agent_id", "generation", "fitness_ema", "loyalty_level", "genome_snapshot",
})

# Phase 4 DNA v2 fields for exports
DNA_V2_FIELDS = frozenset({"meta_dna_v2", "memetic_dna_v2", "fractal_dna", "alchemical"})


def validate_meta_sample(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Phase 1 gate: schema validation for meta-evolution export rows."""
    errors: List[str] = []
    missing = REQUIRED_META_FIELDS - set(sample.keys())
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    if not isinstance(sample.get("traits"), dict):
        errors.append("traits must be dict")
    elif sample["traits"] and not all(0.0 <= float(v) <= 1.0 for v in sample["traits"].values()):
        errors.append("trait values must be in [0, 1]")
    if not isinstance(sample.get("fitness_history"), list):
        errors.append("fitness_history must be list")
    if not isinstance(sample.get("neurocoins"), (int, float)):
        errors.append("neurocoins must be numeric")
    return len(errors) == 0, errors


def validate_genomic_sample(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    missing = REQUIRED_GENOMIC_FIELDS - set(sample.keys())
    if missing:
        errors.append(f"missing fields: {sorted(missing)}")
    snap = sample.get("genome_snapshot", {})
    if not isinstance(snap, dict) or "loci" not in snap:
        errors.append("genome_snapshot.loci required")
    return len(errors) == 0, errors

def collect_meta_samples(substrate, mind, economy) -> List[Dict[str, Any]]:
    samples = []
    for genome_id, rec in getattr(substrate, 'genomes', {}).items():
        dna = rec.dna if hasattr(rec, 'dna') else None
        if not dna:
            continue
        traits = dna.get_traits()
        fitness_history = getattr(rec, 'fitness_history', [])
        balance = economy.get_balance(genome_id) if hasattr(economy, 'get_balance') else 0.0
        memories = getattr(dna, 'phenomenal_memory', [])[-20:]

        is_theorist = getattr(dna, 'role', '') == "THEORIST_ELITE"
        canonical_memories = [m for m in memories if m.get("canonical")]
        theory_lab_cycles = [m.get("theory_lab_cycle") for m in memories if m.get("theory_lab_cycle") is not None]

        samples.append({
            "genome_id": genome_id,
            "role": getattr(dna, 'role', rec.role if hasattr(rec, 'role') else ''),
            "is_theorist": is_theorist,
            "traits": traits,
            "fitness_history": fitness_history,
            "neurocoins": balance,
            # Phase 4 DNA v2 artifacts (when present on dna)
            "meta_dna_v2": getattr(dna, 'meta_dna_v2', None) and getattr(dna.meta_dna_v2, 'apply_rules', lambda: None)(),
            "memetic_dna_v2": getattr(dna, 'memetic_dna_v2', None) and getattr(dna.memetic_dna_v2, 'get_stats', lambda: None)(),
            "fractal_dna": getattr(dna, 'fractal_dna', None) and getattr(dna.fractal_dna, 'get_stats', lambda: None)(),
            "alchemical": getattr(dna, 'alchemical', None) and getattr(dna.alchemical, 'get_stats', lambda: None)(),
            "recent_memories": memories,
            "canonical_memories": canonical_memories,
            "theory_lab_cycles": theory_lab_cycles,
            # placeholders for scores - populate from lab when logging
            "theory_depth_score": None,
            "coherence_score": None,
            "novelty_score": None,
            "harm_score": None,
        })
    return samples

def export_meta_evolution_jsonl(
    samples: List[Dict[str, Any]],
    path: str = "data/oss_meta_evolution.jsonl",
    *,
    validate: bool = True,
) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(p, "w", encoding="utf-8") as f:
        for s in samples:
            if validate:
                ok, errs = validate_meta_sample(s)
                if not ok:
                    log.warning("skip invalid meta sample %s: %s", s.get("genome_id"), errs)
                    continue
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            written += 1
    log.info("exported %d meta-evolution samples to %s", written, path)
    return written


# ── Genomic snapshot export (sovereign_interface → training flywheel) ─────────

def collect_genomic_samples(ecosystem) -> List[Dict[str, Any]]:
    """
    Extract genome snapshots from OmniSentientEcosystem for training.
    Each sample becomes a (prompt, completion) pair formatted for QLoRA fine-tuning.
    """
    samples = []
    for agent in ecosystem.swarm.all():
        loyalty_metrics = ecosystem.loyalty.get_metrics(agent.agent_id)
        snapshot = agent.genome.snapshot()

        # Build a natural-language description of the current genome state
        loci_summary = []
        for locus_type, locus_data in snapshot["loci"].items():
            top_mols = sorted(
                locus_data["molecules"].items(), key=lambda x: x[1], reverse=True
            )[:3]
            loci_summary.append(
                f"{locus_type}(score={locus_data['score']:.3f}): "
                + ", ".join(f"{k}={v:.3f}" for k, v in top_mols)
            )

        prompt = (
            f"Agent {agent.agent_id} genome state at generation {agent.genome.generation}. "
            f"Loyalty: {loyalty_metrics['level']}. "
            f"Fitness EMA: {agent.fitness_ema():.3f}. "
            f"Loci: {'; '.join(loci_summary)}. "
            f"Describe the optimal next task for this agent."
        )
        completion = (
            f"Agent {agent.agent_id} should focus on tasks that leverage its strongest "
            f"locus: {max(snapshot['loci'], key=lambda k: snapshot['loci'][k]['score'])}. "
            f"With {loyalty_metrics['level']} loyalty and generation {agent.genome.generation}, "
            f"it is best suited for "
            + ("deep analytical work requiring causal inference."
               if agent.agent_id in ("oracle", "codex", "web_engineer_elite")
               else "risk assessment and market signal processing."
               if agent.agent_id == "forge"
               else "synthesis and executive communication.")
        )

        samples.append({
            "agent_id":        agent.agent_id,
            "generation":      agent.genome.generation,
            "fitness_ema":     round(agent.fitness_ema(), 4),
            "loyalty_level":   loyalty_metrics["level"],
            "loyalty_metrics": loyalty_metrics,
            "genome_snapshot": snapshot,
            "training_pair": {
                "prompt":     prompt,
                "completion": completion,
            },
        })
    return samples


def export_genomic_snapshots_jsonl(
    ecosystem,
    path: str = "data/training/genomic_snapshots.jsonl",
    append: bool = True,
) -> int:
    """
    Export current ecosystem genome state as training JSONL.
    append=True adds to existing file (incremental flywheel).
    Returns number of samples written.
    """
    samples = collect_genomic_samples(ecosystem)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(p, mode, encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{'Appended' if append else 'Exported'} {len(samples)} genomic samples to {path}")
    return len(samples)
