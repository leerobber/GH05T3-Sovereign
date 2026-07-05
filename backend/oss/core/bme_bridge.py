"""Real bridge from a genome's traits to the actual gh05t3_binary engine:
constructs a live GH05T3BinaryOSS PyTorch model from a genome's
architecture traits.

This is genome-driven ARCHITECTURE construction, not genome-driven
quantization-mode hot-swapping inside a Rust-side context. An earlier
proposed design (a C-ABI GMLContext holding an opaque config struct,
updated/read via ctypes) was rejected after two rounds of real
compile-testing produced fresh compile errors each time -- and, more
fundamentally, even a bug-free version of that design would only have
stored config numbers in a Rust struct with nothing downstream ever
reading them. The real model, its real weights, and the real
binary/ternary quantization decisions all live here, in Python, via
gh05t3_binary.oss.integration.GH05T3BinaryOSS's actual constructor.

Deterministic per-genome seeding + caching: keyed by genome_id, not a
hash of the traits. Genomes are immutable once created (GenomePlane
never mutates a genome's traits in place -- a mutation always produces a
NEW genome with a new id), so re-evaluating the SAME genome_id must give
the same real score every time, not fresh randomness each call.
Deliberately NOT keyed by trait/architecture hash: two DIFFERENT genome
ids that happen to share identical trait values are still meant to be
independent evolutionary samples (their initial weights are part of
what evolution is implicitly exploring); collapsing them onto shared
weights would silently make distinct individuals behave identically.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from gh05t3_binary.oss.integration import GH05T3BinaryOSS

# Matches the real trained tokenizer's vocab (gh05t3_binary/train/
# checkpoints/tokenizer.json) -- used only as a fallback default when a
# genome doesn't declare its own vocab_size trait.
_DEFAULT_VOCAB_SIZE = 4096


def _seed_from_genome_id(genome_id: str) -> int:
    """A stable seed derived from genome_id. Python's builtin hash() is
    randomized per-process (for hash-flooding security) and would give a
    different seed every restart -- not suitable here, since
    reproducibility must survive across process restarts, not just
    within one."""
    digest = hashlib.sha256(genome_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class BMEBridge:
    """Bridges genome traits to a real GH05T3BinaryOSS, cached per
    genome_id. Architecture traits (num_layers/dim/num_heads/vocab_size)
    can't be applied to an existing model in place, since changing any
    of them changes the shape of every weight tensor -- so the first
    call for a given genome_id builds a fresh model; every subsequent
    call for that SAME genome_id returns the identical cached object
    (safe, since genomes are immutable).
    """

    def __init__(self):
        self._cache: dict[str, GH05T3BinaryOSS] = {}

    def apply_genome_to_engine(self, genome_id: str, traits: dict[str, Any]) -> GH05T3BinaryOSS:
        cached = self._cache.get(genome_id)
        if cached is not None:
            return cached

        seed = _seed_from_genome_id(genome_id)
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            model = GH05T3BinaryOSS(
                num_layers=int(traits.get("num_layers", 4)),
                dim=int(traits.get("dim", 256)),
                num_heads=int(traits.get("num_heads", 4)),
                vocab_size=int(traits.get("vocab_size", _DEFAULT_VOCAB_SIZE)),
                binary_ratio=float(traits.get("binary_ratio", 0.95)),
                stabilizer=str(traits.get("stabilizer", "mgc")),
            )
        finally:
            # Restore the caller's global RNG state -- seeding a genome's
            # weights deterministically must not perturb unrelated code
            # that also draws from torch's global RNG.
            torch.set_rng_state(rng_state)

        model.eval()
        self._cache[genome_id] = model
        return model
