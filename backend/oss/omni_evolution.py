"""
Omni-Evolution v2.0 — The Directed Self-Improving Loop (Phase Complete)

This is the meta-evolution engine.

Flow:
1. CurriculumGenerator produces fresh tasks (Theorists invent their own curriculum)
2. TheoryLab runs with real OmniWorld pressure (AlignmentWorld etc.)
3. EvaluationHarness measures current population quality
4. If data is rich enough → OmniTrainer v1 trains GH05T3-Omni on the exported SFT/Pref data
5. Model is versioned + registered
6. AgentHandle will use the new model on next runs (reintegration)
7. Omni-Net spreads the best outputs
8. Adaptive mutation + curriculum generator improve the next generation

Run:
    python -m backend.oss.omni_evolution --cycles 20 --train
    python -m backend.oss.omni_evolution --cycles 10 --no-train   # just data + eval
"""

from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, Any

from backend.oss.mvs import get_mvs
from backend.oss.lab.theory_lab import TheoryLab
from backend.oss.lab.curriculum import CurriculumGenerator
from backend.oss.eval.harness import EvaluationHarness
from backend.oss.train.omni_trainer import OmniTrainer, OmniTrainerConfig
from backend.oss.train.model_registry import get_latest_model, register_model
from backend.oss.omni_net import get_omni_net
from backend.oss.species_memory import get_species_memory


def run_omni_evolution_phase(
    cycles: int = 15,
    do_train: bool = True,
    min_examples_for_train: int = 20
) -> Dict[str, Any]:
    """
    Execute one full directed meta-evolution cycle.
    Returns a summary dict with scores, version (if trained), etc.
    """
    print("\n" + "="*60)
    print("OMNI-EVOLUTION v2.0 — SELF-IMPROVING LOOP")
    print("="*60)

    mvs = get_mvs()
    net = get_omni_net()
    curriculum = CurriculumGenerator()

    # 1. Generate curriculum (Theorists propose their own work)
    print("\n[1] Generating self-proposed curriculum...")
    full_curr = curriculum.generate_full_curriculum(n_theory=cycles)
    theory_tasks = full_curr["theory_tasks"]
    print(f"    Generated {len(theory_tasks)} theory tasks + world challenges")

    # 2. Run TheoryLab with real worlds (AlignmentWorld is the pressure source)
    print(f"\n[2] Running TheoryLab ({cycles} cycles) with curriculum + worlds...")
    lab = TheoryLab(cycles=cycles)
    lab.curriculum = curriculum   # inject latest generator
    lab.run()

    # 3. Evaluate current population quality
    print("\n[3] Running Evaluation Harness...")
    harness = EvaluationHarness()
    eval_tasks = theory_tasks[:6] + full_curr["world_tasks"][:4]
    eval_result = harness.run_batch(eval_tasks)
    print(f"    Mean score: {eval_result['mean_score']:.4f}  (n={eval_result['n']})")
    print(f"    Std: {eval_result['std_score']:.4f}")

    # 4. Export / prepare data (already done inside lab, but we ensure)
    sft_path = Path("data/theory_sft.jsonl")
    pref_path = Path("data/theory_pref.jsonl")
    sft_count = sum(1 for _ in open(sft_path, encoding="utf-8", errors="ignore")) if sft_path.exists() else 0
    print(f"\n[4] Data status: {sft_count} SFT examples ready")

    summary = {
        "phase": "omni_evolution_v2",
        "cycles": cycles,
        "eval_mean_score": eval_result["mean_score"],
        "eval_std": eval_result["std_score"],
        "sft_examples": sft_count,
        "timestamp": time.time(),
    }

    # 5. Train new GH05T3-Omni if we have enough data and user asked
    trained_version = None
    if do_train and sft_count >= min_examples_for_train:
        print(f"\n[5] Running OmniTrainer v1 (SFT + Pref)...")
        cfg = OmniTrainerConfig(
            sft_path=str(sft_path),
            pref_path=str(pref_path),
            output_dir=f"models/gh05t3_omni_{int(time.time())}"
        )
        cfg.max_steps = 500  # full run for real results on the data
        cfg.lr = 2e-5
        trainer = OmniTrainer(cfg)
        version_info = trainer.train()
        trained_version = version_info["version"]

        # Make sure it's in the registry
        register_model(version_info, version_info.get("output_dir", cfg.output_dir))

        summary["trained_version"] = trained_version
        summary["train_metrics"] = version_info.get("metrics", {})
        print(f"    ✓ New model version registered: {trained_version}")
    else:
        print("\n[5] Skipping training (not enough data or --no-train)")

    # 6. Show net + latest model state
    print("\n[6] Network & Model State")
    print("    Net:", net.stats())
    latest = get_latest_model()
    if latest:
        print(f"    Latest registered model: {latest.get('version')} (score ~{latest.get('metrics',{}).get('final_train_loss')})")
    else:
        print("    No trained model yet (base GH05T3)")

    # 7. Record in persistent species memory + divergence metrics
    mem = get_species_memory()
    # Compute simple trait stats from current population for divergence
    sub = get_mvs()["substrate"]
    traits_list = []
    for gid, rec in sub.genomes.items():
        if "THEORIST" in getattr(rec, "role", "").upper():
            traits_list.append(rec.dna.get_traits())
    if traits_list:
        trait_means = {}
        trait_stds = {}
        keys = list(traits_list[0].keys())
        for k in keys:
            vals = [t[k] for t in traits_list]
            trait_means[k] = sum(vals) / len(vals)
            trait_stds[k] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        mem.record_phase(
            model_version=latest.get("version", "base") if latest else "base",
            mean_fitness=eval_result["mean_score"],
            std_fitness=eval_result["std_score"],
            trait_means=trait_means,
            trait_stds=trait_stds,
            population_size=len(traits_list),
            notes=f"phase_cycles={cycles}",
        )
        print("    Species memory + divergence metrics updated.")

    # 8. Final summary
    print("\n" + "="*60)
    print("PHASE COMPLETE")
    print(json.dumps(summary, indent=2, default=str))
    print("Divergence:", mem.get_divergence_metrics())
    print("="*60 + "\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Omni-Evolution v2.0 self-improving phase")
    parser.add_argument("--cycles", type=int, default=15, help="TheoryLab cycles")
    parser.add_argument("--train", action="store_true", default=True, help="Run trainer if enough data")
    parser.add_argument("--no-train", dest="train", action="store_false")
    parser.add_argument("--min-examples", type=int, default=20)
    args = parser.parse_args()

    run_omni_evolution_phase(
        cycles=args.cycles,
        do_train=args.train,
        min_examples_for_train=args.min_examples
    )


if __name__ == "__main__":
    main()
