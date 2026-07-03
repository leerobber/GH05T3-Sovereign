"""GH05T3 v7 orchestrator: runs the Rust kernel core loop, fulfills any
MODEL_CALL / multi-model intents it emits via the real ghost_llm cascade,
scores the run, and keeps a running state across cycles.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from backend.integration.gml_kernel_bridge import (
    check_dependencies,
    run_gh05t3_kernel_core_json,
    run_model_via_ghost_llm,
    run_multi_model_via_ghost_llm,
    stream_model_via_ghost_llm,
)


class GH05T3State:
    """Persistent agent/runtime state across kernel cycles."""

    def __init__(self) -> None:
        self.cycles: int = 0
        self.last_kernel_summary: Dict[str, Any] | None = None
        self.last_model_outputs: List[str] = []
        self.last_deps: Dict[str, Any] = {}
        self.evolution_log: List[Dict[str, Any]] = []


def handle_model_calls(kernel_view: Dict[str, Any]) -> List[str]:
    """Scans short_term memory for MODEL_CALL intents the kernel emitted.

    Rust's model_call handler pushes a valid JSON string into short_term
    memory for every MODEL_CALL glyph — either a v2 single-backend payload
    (kernel::payload::ModelCallPayload) or a v4 multi-backend payload
    (kernel::payload::MultiModelCallPayload). This scans for those and
    fulfills each via the real ghost_llm cascade. Non-JSON entries (e.g.
    the DEPENDENCY_CHECK/SENTINEL_DEP placeholder strings) are skipped.

    version:
      - "v1"/"v2": non-streaming via run_model_via_ghost_llm
      - "v3": streaming via stream_model_via_ghost_llm
      - "v4": multi-backend blending via run_multi_model_via_ghost_llm
        (detected by the "backends" key, not by version string, since v4
        payloads carry their own version field independently)
    """
    outputs: List[str] = []

    for entry in kernel_view.get("short_term", []):
        try:
            payload = json.loads(entry)
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(payload, dict):
            continue

        if "backends" in payload:  # v4 multi-model
            result = run_multi_model_via_ghost_llm(
                backends=payload.get("backends", []),
                prompt=payload.get("prompt", ""),
                blend_strategy=payload.get("blend_strategy", "concat"),
                version=payload.get("version", "v4"),
            )
            outputs.append(result["blended"])
        elif "backend" in payload:  # v1/v2/v3 single-backend
            version = payload.get("version", "v2")
            prompt = payload.get("prompt", "")
            backend = payload.get("backend", "claude")

            if version == "v3":
                full_text, _chunks = stream_model_via_ghost_llm(
                    prompt=prompt, backend=backend, version=version,
                )
                outputs.append(full_text)
            else:
                text = run_model_via_ghost_llm(backend=backend, prompt=prompt, version=version)
                outputs.append(text)

    return outputs


def compute_evolution_fitness(
    deps: Dict[str, Any],
    model_outputs: List[str],
) -> Dict[str, Any]:
    """Fitness stub: higher when dependencies are healthy and fewer
    MODEL_ERROR/LOCAL_FALLBACK outputs were produced this cycle."""
    ghost_ok = deps.get("ghost_llm", {}).get("ok", False)
    net_ok = deps.get("net", {}).get("ok", False)
    fs_ok = deps.get("fs", {}).get("ok", False)
    gpu_ok = deps.get("gpu", {}).get("ok", False)

    errors = sum(1 for o in model_outputs if "[MODEL_ERROR]" in o)
    fallbacks = sum(1 for o in model_outputs if "[LOCAL_FALLBACK]" in o)

    fitness = 0.0
    if ghost_ok:
        fitness += 0.3
    if net_ok:
        fitness += 0.2
    if fs_ok:
        fitness += 0.2
    if gpu_ok:
        fitness += 0.3

    fitness -= 0.1 * errors
    fitness -= 0.05 * fallbacks

    return {
        "fitness": max(fitness, 0.0),
        "errors": errors,
        "fallbacks": fallbacks,
        "deps": deps,
    }


def run_gh05t3_cycle(state: GH05T3State, sleep_seconds: float = 0.0) -> None:
    """One full GH05T3 orchestration cycle:
      - check dependencies
      - run the Rust kernel core loop (JSON FFI export)
      - fulfill any MODEL_CALL intents via ghost_llm
      - compute evolution fitness
      - update state
    """
    deps = check_dependencies()
    kernel_view = run_gh05t3_kernel_core_json()

    model_outputs = handle_model_calls(kernel_view)
    evo = compute_evolution_fitness(deps, model_outputs)

    state.cycles += 1
    state.last_kernel_summary = kernel_view
    state.last_model_outputs = model_outputs
    state.last_deps = deps
    state.evolution_log.append(evo)

    print(f"\n=== GH05T3 Cycle {state.cycles} ===")
    print("Dependencies:", json.dumps(deps, indent=2, sort_keys=True))
    print("Kernel tick:", kernel_view.get("tick"))
    print("Model outputs:", model_outputs)
    print("Evolution fitness:", evo)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def run_forever(sleep_seconds: float = 1.0) -> None:
    """Continuous orchestration loop. Call from a CLI entrypoint/supervisor."""
    state = GH05T3State()
    while True:
        run_gh05t3_cycle(state, sleep_seconds=sleep_seconds)
