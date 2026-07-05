"""Real mutation operators over genome traits -- deterministic,
testable perturbations tied to actual traits the real gh05t3_binary
model uses (binary_ratio, stabilizer), not placeholder classes with no
real behavior behind them.
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class Mutation:
    """A concrete, already-decided mutation. apply() performs the exact
    trait transform the operator computed when it judged the base genome
    a mutation candidate."""

    base_id: str
    new_id: str
    description: str
    _transform: Callable[[dict[str, Any]], dict[str, Any]]

    def apply(self, base_traits: dict[str, Any]) -> dict[str, Any]:
        return self._transform(base_traits)


class MutationOperator(Protocol):
    def is_applicable(self, genome: Any, perf: dict[str, Any]) -> bool: ...
    def create_mutation(self, genome: Any, perf: dict[str, Any]) -> Mutation: ...


class BinaryRatioJitterMutation:
    """Nudges a genome's `binary_ratio` trait (see
    gh05t3_binary.core.attention.HybridBinaryAttention's binary_ratio
    param) up or down within [min_ratio, max_ratio] by a bounded random
    step. Only applicable to genomes that actually declare this trait --
    genomes without it are left alone rather than this operator
    inventing a value for them.
    """

    def __init__(
        self,
        step: float = 0.02,
        min_ratio: float = 0.5,
        max_ratio: float = 1.0,
        rng: random.Random | None = None,
    ):
        self.step = step
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.rng = rng or random.Random()

    def is_applicable(self, genome: Any, perf: dict[str, Any]) -> bool:
        return "binary_ratio" in genome.traits

    def create_mutation(self, genome: Any, perf: dict[str, Any]) -> Mutation:
        delta = self.rng.choice([-1.0, 1.0]) * self.step

        def transform(traits: dict[str, Any]) -> dict[str, Any]:
            new_traits = dict(traits)
            current = float(new_traits["binary_ratio"])
            new_traits["binary_ratio"] = max(self.min_ratio, min(self.max_ratio, current + delta))
            return new_traits

        return Mutation(
            base_id=genome.id,
            new_id=f"{genome.id}-mut-{uuid.uuid4().hex[:8]}",
            description=f"binary_ratio {'+' if delta > 0 else ''}{delta:.3f}",
            _transform=transform,
        )


class StabilizerSwitchMutation:
    """Flips a genome's `stabilizer` trait between the two stabilizers
    gh05t3_binary.core.transformer.GH05T3BinaryTransformer actually
    supports (see core/stabilizers.py: "mgc" / "damg"). Only applicable
    to genomes declaring one of these two known values, so it never
    invents a stabilizer name that doesn't exist in the real model.
    """

    _KNOWN = ("mgc", "damg")

    def is_applicable(self, genome: Any, perf: dict[str, Any]) -> bool:
        return genome.traits.get("stabilizer") in self._KNOWN

    def create_mutation(self, genome: Any, perf: dict[str, Any]) -> Mutation:
        current = genome.traits["stabilizer"]
        flipped = "damg" if current == "mgc" else "mgc"

        def transform(traits: dict[str, Any]) -> dict[str, Any]:
            new_traits = dict(traits)
            new_traits["stabilizer"] = flipped
            return new_traits

        return Mutation(
            base_id=genome.id,
            new_id=f"{genome.id}-mut-{uuid.uuid4().hex[:8]}",
            description=f"stabilizer {current} -> {flipped}",
            _transform=transform,
        )


class QuantModeMutation:
    """Flips a genome's `out_proj_quant_mode` trait between the two real
    quantization modes gh05t3_binary.core.attention.HybridBinaryAttention
    actually supports for out_proj ("ternary" / "binary" -- see
    gh05t3_binary/core/attention.py and binary_layers.py's BinaryLinear/
    TernaryLinear). Inspired by a real historical precedent found in the
    original GH05T3 repo's oss/living_loop/genome.py: a KernelGenome had
    an evolvable quant_mode field, wired into mutate()/crossover(), whose
    source no longer survives (same "ghost" pattern as several other
    subsystems investigated this session). Rebuilt here against REAL,
    already-verified components (BinaryLinear/TernaryLinear, and the
    actual out_proj_quant_mode constructor parameter threaded through
    HybridBinaryAttention/BinaryTransformerBlock/GH05T3BinaryTransformer/
    GH05T3BinaryOSS), not reconstructed from bytecode or file names.

    Only applicable to genomes declaring one of these two known values,
    so it never invents a quant mode that doesn't exist in the real
    model -- same convention as StabilizerSwitchMutation above.
    """

    _KNOWN = ("ternary", "binary")

    def is_applicable(self, genome: Any, perf: dict[str, Any]) -> bool:
        return genome.traits.get("out_proj_quant_mode") in self._KNOWN

    def create_mutation(self, genome: Any, perf: dict[str, Any]) -> Mutation:
        current = genome.traits["out_proj_quant_mode"]
        flipped = "binary" if current == "ternary" else "ternary"

        def transform(traits: dict[str, Any]) -> dict[str, Any]:
            new_traits = dict(traits)
            new_traits["out_proj_quant_mode"] = flipped
            return new_traits

        return Mutation(
            base_id=genome.id,
            new_id=f"{genome.id}-mut-{uuid.uuid4().hex[:8]}",
            description=f"out_proj_quant_mode {current} -> {flipped}",
            _transform=transform,
        )
