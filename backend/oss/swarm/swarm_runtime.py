"""Real genome evaluation: builds a genome's model via BMEBridge, runs an
actual forward pass on a real (or seeded-random, as a documented
fallback) token batch, and reports the real cross-entropy loss. This
measures architecture/initialization VALIDITY -- does this genome
produce a stable, correctly-shaped forward pass and a sane loss
magnitude -- not trained quality. Each genome gets a fresh random init,
not a trained checkpoint: different genomes can have different shapes
(num_layers/dim), so sharing trained weights across genomes isn't
possible without a real, separate training run per genome, which this
stage does not attempt.
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn.functional as F

from backend.oss.core.bme_bridge import BMEBridge

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CORPUS_PATH = os.path.join(_REPO_ROOT, "gh05t3_binary", "train", "checkpoints", "corpus.txt")
_TOKENIZER_PATH = os.path.join(_REPO_ROOT, "gh05t3_binary", "train", "checkpoints", "tokenizer.json")


def _load_real_eval_batch(
    seq_len: int, batch_size: int, vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (input_ids, targets), both (batch_size, seq_len). Uses the
    real trained BPE tokenizer + real corpus if both are present on disk
    (regenerated build artifacts, gitignored, not repo-tracked) -- falls
    back to a fixed-seed random token tensor otherwise, so genome
    evaluation still works on a machine that hasn't run training yet
    (with the documented caveat that the fallback tests forward-pass
    validity, not real-text loss).

    Targets are a contiguous next-token shift of the same flat sequence
    (matching gh05t3_binary.train.dataset.TokenStreamDataset's real
    convention), not a wraparound of each row -- row r's target is
    exactly the token that follows row r in the real corpus.
    """
    needed = batch_size * seq_len + 1

    if os.path.isfile(_TOKENIZER_PATH) and os.path.isfile(_CORPUS_PATH):
        from gh05t3_binary.train.bpe_tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(_TOKENIZER_PATH)
        with open(_CORPUS_PATH, "r", encoding="utf-8") as f:
            text = f.read()
        ids = tokenizer.encode(text)

        if len(ids) >= needed:
            # Clamp to the genome's own vocab_size in case it differs
            # from the real tokenizer's trained vocab (e.g. a genome
            # testing a smaller vocab) -- real tokens, clamped, never
            # fabricated ones.
            clamped = [min(i, vocab_size - 1) for i in ids[:needed]]
            flat = torch.tensor(clamped, dtype=torch.long)
            input_ids = flat[:-1].reshape(batch_size, seq_len)
            targets = flat[1:].reshape(batch_size, seq_len)
            return input_ids, targets

    generator = torch.Generator().manual_seed(1234)
    flat = torch.randint(0, vocab_size, (needed,), generator=generator)
    input_ids = flat[:-1].reshape(batch_size, seq_len)
    targets = flat[1:].reshape(batch_size, seq_len)
    return input_ids, targets


class SwarmRuntime:
    def __init__(self, bme_bridge: BMEBridge | None = None, seq_len: int = 32, batch_size: int = 4):
        self.bme_bridge = bme_bridge or BMEBridge()
        self.seq_len = seq_len
        self.batch_size = batch_size

    def evaluate_genome(self, genome_id: str, traits: dict[str, Any]) -> dict[str, Any]:
        """Builds (or reuses -- see BMEBridge's genome_id-keyed cache)
        the genome's model and runs one real forward pass + real
        cross-entropy loss on a real (or seeded-random fallback) token
        batch. Returns a real score and real wall-clock latency -- never
        a hardcoded placeholder value (the original rebuild spec's
        SwarmRuntime.evaluate_genome returned a literal
        {"score": 0.75, "latency": 0.01, "stability": 0.99} regardless of
        input; this replaces every one of those fields with a measured
        value). Re-evaluating the SAME genome_id gives the SAME real
        score every time (deterministic per-genome seeding in
        BMEBridge), not fresh randomness each call."""
        vocab_size = int(traits.get("vocab_size", 4096))
        model = self.bme_bridge.apply_genome_to_engine(genome_id, traits)

        input_ids, targets = _load_real_eval_batch(self.seq_len, self.batch_size, vocab_size)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
        latency = time.perf_counter() - t0

        loss_value = loss.item()
        return {
            # HighestScoreSelection (dna/selection_strategies.py) picks
            # the max score -- score is the negative loss so lower real
            # loss means a higher, better score, not the raw loss value.
            "score": -loss_value,
            "loss": loss_value,
            "latency": latency,
        }
