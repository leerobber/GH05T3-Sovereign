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
"""
from __future__ import annotations

from typing import Any

from gh05t3_binary.oss.integration import GH05T3BinaryOSS

# Matches the real trained tokenizer's vocab (gh05t3_binary/train/
# checkpoints/tokenizer.json) -- used only as a fallback default when a
# genome doesn't declare its own vocab_size trait.
_DEFAULT_VOCAB_SIZE = 4096


class BMEBridge:
    """Bridges genome traits to a real, freshly-constructed
    GH05T3BinaryOSS. Each call builds a NEW model -- architecture traits
    (num_layers/dim/num_heads/vocab_size) can't be applied to an existing
    model in place, since changing any of them changes the shape of
    every weight tensor. This is real, and relatively expensive; it's
    meant to be called per-genome-evaluation (see swarm/swarm_runtime.py),
    not per-inference-request.
    """

    def apply_genome_to_engine(self, traits: dict[str, Any]) -> GH05T3BinaryOSS:
        model = GH05T3BinaryOSS(
            num_layers=int(traits.get("num_layers", 4)),
            dim=int(traits.get("dim", 256)),
            num_heads=int(traits.get("num_heads", 4)),
            vocab_size=int(traits.get("vocab_size", _DEFAULT_VOCAB_SIZE)),
            binary_ratio=float(traits.get("binary_ratio", 0.95)),
            stabilizer=str(traits.get("stabilizer", "mgc")),
        )
        model.eval()
        return model
