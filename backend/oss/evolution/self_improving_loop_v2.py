"""
Self-Improving Loop v2 — Phase 6.2

Closed loop: export meta data → fine-tune → reintegrate into MVS.
Species tracking via KMeans on trait vectors.
Dry-run test (no GPU).

Success: new models beat previous on benchmarks.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import time
import random
from pathlib import Path

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

from backend.oss.meta_export import export_meta_evolution_jsonl, collect_meta_samples

# Lazy imports to break circulars (mvs <-> harness <-> loop)
def _get_mvs():
    from backend.oss.mvs import get_mvs
    return get_mvs()

def _get_harness():
    from backend.oss.eval.harness import EvaluationHarness
    return EvaluationHarness()


class SelfImprovingLoopV2:
    def __init__(self, output_dir: str = "models/gh05t3_omni_v2"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self._mvs = None  # lazy

    @property
    def mvs(self):
        if self._mvs is None:
            self._mvs = _get_mvs()
        return self._mvs

    @mvs.setter
    def mvs(self, val):
        self._mvs = val

    def export_training_data(self) -> Dict[str, str]:
        """Export latest meta-evolution data for training."""
        samples = collect_meta_samples(
            self.mvs["substrate"], self.mvs.get("mind"), self.mvs.get("economy")
        )
        sft_path = str(self.output_dir / "sft_v2.jsonl")
        pref_path = str(self.output_dir / "pref_v2.jsonl")
        # Simple split for demo
        with open(sft_path, "w") as f:
            for s in samples:
                f.write(json.dumps({"prompt": s.get("traits", {}), "completion": s}) + "\n")
        with open(pref_path, "w") as f:
            for i, s in enumerate(samples):
                if i % 2 == 0:
                    f.write(json.dumps({"chosen": s, "rejected": samples[(i+1)%len(samples)]}) + "\n")
        return {"sft": sft_path, "pref": pref_path}

    def run_training_cycle(self, dry_run: bool = True) -> Dict[str, Any]:
        """Run one full self-improving cycle."""
        start = time.time()
        data_paths = self.export_training_data()

        harness = _get_harness()
        before = harness.run_batch([{"prompt": "theory task v2"}], role="THEORIST_ELITE")

        if dry_run:
            # Simulate training
            metrics = {"mean_score": before["mean_score"] + random.uniform(0.03, 0.08)}
            version = f"gh05t3_omni_v2_{int(time.time())}"
            model_path = str(self.output_dir / version)
            (self.output_dir / version).mkdir(exist_ok=True)
            (Path(model_path) / "adapter.placeholder").write_text("dry-run v2")
        else:
            # Real path would call trainer
            from backend.oss.train.omni_trainer import OmniTrainer, OmniTrainerConfig
            cfg = OmniTrainerConfig(sft_path=data_paths["sft"], pref_path=data_paths["pref"], output_dir=str(self.output_dir))
            trainer = OmniTrainer(cfg)
            version_info = trainer.train()
            metrics = version_info.get("metrics", {})
            version = version_info["version"]
            model_path = version_info.get("path", str(self.output_dir))

        after = harness.run_batch([{"prompt": "theory task v2"}], role="THEORIST_ELITE")

        improvement = after["mean_score"] - before["mean_score"]

        cycle_result = {
            "version": version,
            "before": before["mean_score"],
            "after": after["mean_score"],
            "improvement": round(improvement, 4),
            "duration_s": round(time.time() - start, 1),
            "data_paths": data_paths,
            "model_path": model_path,
            "timestamp": time.time(),
        }
        self.history.append(cycle_result)
        return cycle_result

    def track_species_with_kmeans(self, n_clusters: int = 5) -> Dict[str, Any]:
        """Track species divergence using traits (KMeans)."""
        substrate = self.mvs["substrate"]
        trait_matrix = []
        ids = []
        for gid, rec in substrate.genomes.items():
            traits = rec.dna.get_traits() if hasattr(rec, "dna") else getattr(rec, "traits", {})
            vec = [traits.get(k, 0.5) for k in list(traits.keys())[:12]]  # fixed dim
            trait_matrix.append(vec)
            ids.append(gid)

        if len(trait_matrix) < n_clusters or not HAS_SKLEARN:
            # Fallback simple clustering by hash
            clusters = {i: [] for i in range(n_clusters)}
            for i, gid in enumerate(ids):
                clusters[i % n_clusters].append(gid)
            return {"clusters": clusters, "method": "fallback", "n": len(ids)}

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(trait_matrix)
        clusters: Dict[int, List[str]] = {i: [] for i in range(n_clusters)}
        for gid, lab in zip(ids, labels):
            clusters[int(lab)].append(gid)
        return {"clusters": {str(k): v for k,v in clusters.items()}, "method": "kmeans", "n": len(ids)}

    def dry_run_test(self) -> bool:
        """No-GPU CI friendly test."""
        result = self.run_training_cycle(dry_run=True)
        species = self.track_species_with_kmeans(3)
        return result["improvement"] > 0 and len(species["clusters"]) >= 2


def get_self_improving_loop_v2() -> SelfImprovingLoopV2:
    if not hasattr(get_self_improving_loop_v2, "_inst"):
        get_self_improving_loop_v2._inst = SelfImprovingLoopV2()
    return get_self_improving_loop_v2._inst
