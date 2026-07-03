"""
GH05T3 Training Pipeline — orchestrates collection + generation.

Run via API:  POST /api/training/run
Run directly: python -m training.pipeline
Runs overnight using KAIROS nightly slot — zero cost.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger("ghost.training.pipeline")

DATASETS_DIR = Path(__file__).parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)


# Targets — tunable via env so you can do small test runs first
TARGETS = {
    "adversarial_defense": int(os.environ.get("TRAIN_TARGET_DEFENSE",   "5000")),
    "reasoning_chains":    int(os.environ.get("TRAIN_TARGET_REASONING", "3000")),
    "cve_patterns":        int(os.environ.get("TRAIN_TARGET_CVE",       "3000")),
    "bug_bounty":          int(os.environ.get("TRAIN_TARGET_BOUNTY",    "5000")),
}

_running = False
_progress: dict = {}


async def run_pipeline(
    collect: bool = True,
    generate: bool = True,
    ws_broadcast=None,
) -> dict:
    """
    Full pipeline: collect free public data, then generate with local LLM.
    Safe to restart — skips already-generated examples.
    ws_broadcast: optional coroutine to push progress to dashboard.
    """
    global _running, _progress
    if _running:
        return {"status": "already_running", "progress": _progress}

    _running = True
    _progress = {"phase": "starting", "datasets": {}}

    async def _emit(msg: str, data: dict = None):
        _progress.update(data or {})
        LOG.info("[pipeline] %s", msg)
        if ws_broadcast:
            try:
                await ws_broadcast("training_progress", {"msg": msg, **(data or {})})
            except Exception:
                pass

    try:
        from training.collectors import (
            collect_nvd_cves, collect_mitre_attack,
            collect_hf_reasoning, load_raw, raw_stats,
        )
        from training.generators import (
            generate_adversarial_defense, generate_reasoning_chains,
            generate_cve_patterns, generate_bug_bounty, dataset_stats,
            reset_tracker, get_tracker,
        )

        reset_tracker()

        # ── Phase 1: Collect raw public data ─────────────────────────
        nvd_records = mitre_records = hf_records = []

        if collect:
            await _emit("Collecting NVD CVE data (free public API)...")
            _progress["phase"] = "collecting_nvd"
            nvd_records = load_raw("nvd_cves") or await collect_nvd_cves(3000)

            await _emit(f"NVD done: {len(nvd_records)} CVEs. Collecting MITRE ATT&CK...")
            _progress["phase"] = "collecting_mitre"
            mitre_records = load_raw("mitre_attack") or await collect_mitre_attack()

            await _emit(f"ATT&CK done: {len(mitre_records)} techniques. Collecting HF reasoning...")
            _progress["phase"] = "collecting_hf"
            hf_records = load_raw("hf_reasoning") or await collect_hf_reasoning(3000)

            await _emit(f"Collection complete. Raw stats: {raw_stats()}")
        else:
            from training.collectors import load_raw
            nvd_records    = load_raw("nvd_cves")
            mitre_records  = load_raw("mitre_attack")
            hf_records     = load_raw("hf_reasoning")

        # ── Phase 2: Generate training datasets ───────────────────────
        if generate:
            await _emit("Generating adversarial defense dataset...")
            _progress["phase"] = "generating_defense"
            n = await generate_adversarial_defense(
                TARGETS["adversarial_defense"], nvd_records, mitre_records)
            _progress["datasets"]["adversarial_defense"] = n
            await _emit(f"adversarial_defense: {n} examples", {"datasets": _progress["datasets"]})

            await _emit("Generating reasoning chains dataset...")
            _progress["phase"] = "generating_reasoning"
            n = await generate_reasoning_chains(TARGETS["reasoning_chains"], hf_records)
            _progress["datasets"]["reasoning_chains"] = n
            await _emit(f"reasoning_chains: {n} examples", {"datasets": _progress["datasets"]})

            await _emit("Generating CVE pattern dataset...")
            _progress["phase"] = "generating_cve"
            n = await generate_cve_patterns(TARGETS["cve_patterns"], nvd_records)
            _progress["datasets"]["cve_patterns"] = n
            await _emit(f"cve_patterns: {n} examples", {"datasets": _progress["datasets"]})

            await _emit("Generating bug bounty dataset...")
            _progress["phase"] = "generating_bounty"
            n = await generate_bug_bounty(TARGETS["bug_bounty"], mitre_records)
            _progress["datasets"]["bug_bounty"] = n
            await _emit(f"bug_bounty: {n} examples", {"datasets": _progress["datasets"]})

        final = dataset_stats()
        total = sum(final.values())
        cost  = get_tracker().to_dict()
        _progress["phase"] = "complete"
        _progress["datasets"] = final
        _progress["total"] = total
        _progress["cost"] = cost
        await _emit(f"Pipeline complete. Total: {total} training examples.", {"phase": "complete"})
        return {"status": "complete", "datasets": final, "total": total, "cost": cost}

    except Exception as e:
        LOG.exception("pipeline error")
        _progress["phase"] = "error"
        _progress["error"] = str(e)
        return {"status": "error", "error": str(e)}
    finally:
        _running = False


def pipeline_status() -> dict:
    from training.generators import dataset_stats, get_tracker
    from training.collectors import raw_stats
    return {
        "running":  _running,
        "progress": _progress,
        "datasets": dataset_stats(),
        "raw":      raw_stats(),
        "targets":  TARGETS,
        "cost":     get_tracker().to_dict(),
    }


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    collect_only  = "--collect-only"  in sys.argv
    generate_only = "--generate-only" in sys.argv

    asyncio.run(run_pipeline(
        collect  = not generate_only,
        generate = not collect_only,
    ))
