"""Tests for SwarmRuntime.evaluate_genome: the actual correctness gate
for Stage 2 -- confirms score/loss/latency are real, measured values,
not the original rebuild spec's hardcoded {"score": 0.75, "latency":
0.01, "stability": 0.99}.
"""
from __future__ import annotations

import backend.oss.swarm.swarm_runtime as swarm_runtime_module
from backend.oss.swarm.swarm_runtime import SwarmRuntime, _load_real_eval_batch


def _tiny_runtime() -> SwarmRuntime:
    # Small enough to run fast in a test; still a real forward pass
    # through a real (freshly, randomly initialized) model.
    return SwarmRuntime(seq_len=8, batch_size=2)


def test_evaluate_genome_returns_real_measured_values_not_a_placeholder():
    runtime = _tiny_runtime()
    result = runtime.evaluate_genome("g1", {"num_layers": 1, "dim": 32, "num_heads": 2, "vocab_size": 16})

    assert result["loss"] > 0.0
    assert result["score"] == -result["loss"]
    assert result["latency"] > 0.0
    # Regression guard against the original spec's literal stub values.
    assert result["score"] != 0.75
    assert result["latency"] != 0.01
    assert "stability" not in result  # fabricated field from the original spec, not reproduced here


def test_different_genome_shapes_produce_different_real_scores():
    runtime = _tiny_runtime()
    small = runtime.evaluate_genome("small", {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 16})
    big = runtime.evaluate_genome("big", {"num_layers": 2, "dim": 32, "num_heads": 4, "vocab_size": 16})

    # Different architectures, different random inits -- virtually
    # certain to produce different real losses (continuous floats).
    assert small["score"] != big["score"]


def test_re_evaluating_the_same_genome_id_gives_the_same_real_score():
    """Reproducibility: the same unchanged genome evaluated twice (e.g.
    across two evolution cycles it survives unmutated) must not look
    like it randomly changed fitness -- BMEBridge's genome_id-keyed
    deterministic seeding + caching guarantees this."""
    runtime = _tiny_runtime()
    traits = {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 16}

    first = runtime.evaluate_genome("stable-genome", traits)
    second = runtime.evaluate_genome("stable-genome", traits)

    # score/loss must be exactly reproducible; latency is real wall-clock
    # timing and will naturally jitter between calls, so it's excluded.
    assert first["score"] == second["score"]
    assert first["loss"] == second["loss"]


def test_different_genome_ids_with_identical_traits_can_still_score_differently():
    runtime = _tiny_runtime()
    traits = {"num_layers": 1, "dim": 16, "num_heads": 2, "vocab_size": 16}

    a = runtime.evaluate_genome("genome-a", traits)
    b = runtime.evaluate_genome("genome-b", traits)

    assert a["loss"] != b["loss"]


def test_eval_batch_uses_seeded_random_fallback_when_no_real_corpus_present(monkeypatch):
    monkeypatch.setattr(swarm_runtime_module, "_TOKENIZER_PATH", "/nonexistent/tokenizer.json")
    monkeypatch.setattr(swarm_runtime_module, "_CORPUS_PATH", "/nonexistent/corpus.txt")

    input_a, targets_a = _load_real_eval_batch(seq_len=8, batch_size=2, vocab_size=50)
    input_b, targets_b = _load_real_eval_batch(seq_len=8, batch_size=2, vocab_size=50)

    # Fixed seed -> deterministic fallback, not fresh randomness each call.
    assert (input_a == input_b).all()
    assert (targets_a == targets_b).all()


def test_eval_batch_targets_are_a_real_next_token_shift_not_a_wraparound(monkeypatch):
    monkeypatch.setattr(swarm_runtime_module, "_TOKENIZER_PATH", "/nonexistent/tokenizer.json")
    monkeypatch.setattr(swarm_runtime_module, "_CORPUS_PATH", "/nonexistent/corpus.txt")

    seq_len, batch_size, vocab_size = 8, 3, 50
    input_ids, targets = _load_real_eval_batch(seq_len, batch_size, vocab_size)

    # Row r's target must be the real continuation of row r -- i.e. row
    # r's target position i equals row r's input position i+1, and the
    # last target position of row r equals the first input position of
    # row r+1 (contiguous flat sequence, not each row wrapping on itself).
    assert (targets[:, :-1] == input_ids[:, 1:]).all()
    assert (targets[:-1, -1] == input_ids[1:, 0]).all()
