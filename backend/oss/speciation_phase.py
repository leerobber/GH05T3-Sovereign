"""
Speciation Phase — Species Divergence Experiments (full phase)

Runs directed experiments that drive populations to speciate using measured
trait + meta-DNA divergence (no forced demo events).

Run:
    python -m backend.oss.speciation_phase --experiments 5 --cycles 20
"""

import argparse
import json
import time
from typing import Dict, Any

from backend.oss.lab.theory_lab import TheoryLab
from backend.oss.speciation import get_speciation_engine, render_phylogeny_ascii
from backend.oss.lab.curriculum import CurriculumGenerator
from backend.oss.omni_net import get_omni_net


def run_speciation_experiments(experiments: int = 5, cycles: int = 20) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("SPECIATION PHASE — SPECIES DIVERGENCE EXPERIMENTS")
    print("=" * 60)

    spec = get_speciation_engine()
    curriculum = CurriculumGenerator()
    get_omni_net()

    results = []

    for exp in range(experiments):
        print(f"\n=== Experiment {exp + 1}/{experiments} ===")
        lab = TheoryLab(cycles=cycles)
        lab.curriculum = curriculum
        lab.run()

        theorists = list(lab.theorists)
        niche = f"experiment-{exp + 1}"
        new_sp, measured_div = spec.attempt_speciation_with_pressure(
            theorists,
            exp * 100,
            niche=niche,
            base_strength=0.10 + exp * 0.02,
            max_rounds=8,
            strength_step=0.05,
        )

        threshold = spec.divergence_threshold
        if new_sp:
            print(f"  Speciation event (measured divergence={measured_div:.3f}, threshold={threshold})")
            print(f"  New species: {new_sp}")
            print("  → New species isolated from parent in future memetic exchanges.")
        else:
            print(
                f"  No speciation (measured divergence={measured_div:.3f} < threshold={threshold})"
            )

        event_summary = {
            "experiment": exp + 1,
            "cycles": cycles,
            "new_species": new_sp,
            "measured_divergence": round(measured_div, 4),
            "threshold": threshold,
            "speciated": bool(new_sp),
            "total_species": len(spec.species),
        }
        results.append(event_summary)
        print(f"  Current species count: {len(spec.species)}")

    from dataclasses import asdict as dc_asdict

    phylogeny_text = render_phylogeny_ascii(spec.events)
    summary = {
        "phase": "speciation",
        "experiments": experiments,
        "final_species": len(spec.species),
        "total_events": len(spec.events),
        "speciation_rate": sum(1 for r in results if r["speciated"]) / max(1, experiments),
        "results": results,
        "phylogeny": [dc_asdict(e) for e in spec.events[-10:]] if spec.events else [],
        "phylogeny_ascii": phylogeny_text,
    }

    print("\n" + "=" * 60)
    print("SPECIATION PHASE COMPLETE")
    print(phylogeny_text)
    print(json.dumps(summary, indent=2, default=str))
    print("=" * 60 + "\n")

    with open("data/speciation_phase_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n[6] Preparing SFT/Pref data and launching full-weight training for GH05T3-Omni...")
    try:
        from backend.oss.lab.prepare_theory_training import main as prepare_main

        prepare_main()
        from backend.oss.train.omni_trainer import OmniTrainer, OmniTrainerConfig

        cfg = OmniTrainerConfig(
            sft_path="data/theory_sft.jsonl",
            pref_path="data/theory_pref.jsonl",
            output_dir=f"models/gh05t3_omni_speciation_{int(time.time())}",
        )
        cfg.max_steps = 500
        cfg.lr = 2e-5
        trainer = OmniTrainer(cfg)
        version_info = trainer.train()
        print(f"    Full-weight training completed for version {version_info['version']}")
        summary["trained_version"] = version_info["version"]
        summary["train_metrics"] = version_info.get("metrics", {})
    except Exception as e:
        print(f"    Training launch note: {e}. Run the trainer manually on prepared data.")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()

    run_speciation_experiments(args.experiments, args.cycles)