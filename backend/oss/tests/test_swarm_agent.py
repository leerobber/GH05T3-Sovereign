"""Tests for SwarmAgent + AgentSwarmRuntime: proves the real end-to-end
path -- read real personas from a BinaryLedger, run a real batched
forward pass through a real GH05T3BinaryOSS model, get back real,
per-agent logits.
"""
from __future__ import annotations

import torch

from backend.oss.core.binary_ledger import DESIRE_NAMES, NO_PARENT, BinaryLedger
from backend.oss.swarm.agent_swarm_runtime import AgentSwarmRuntime
from backend.oss.swarm.swarm_agent import load_active_agents
from gh05t3_binary.oss.integration import GH05T3BinaryOSS


def _seeded_ledger(tmp_path):
    path = str(tmp_path / "swarm.bin")
    ledger = BinaryLedger(path, capacity=10, create=True)
    ledger.write_agent(0, desires=(0.9, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5), maturity=4, fitness=0.8, parent_offset=NO_PARENT, generation=0)
    ledger.write_agent(1, desires=(0.2, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5), maturity=3, fitness=0.6, parent_offset=0, generation=1)
    ledger.write_zero(2)  # explicitly vacant -- must be excluded
    return ledger


def test_load_active_agents_reads_real_personas_and_skips_vacant_slots(tmp_path):
    ledger = _seeded_ledger(tmp_path)
    try:
        agents = load_active_agents(ledger)

        assert len(agents) == 2  # slot 2 was zeroed, must be excluded
        assert {a.slot_index for a in agents} == {0, 1}

        agent0 = next(a for a in agents if a.slot_index == 0)
        assert set(agent0.persona.keys()) == set(DESIRE_NAMES)
        assert abs(agent0.persona["KNOWLEDGE"] - 0.9) < 0.01
        assert abs(agent0.fitness - 0.8) < 0.01
        assert agent0.parent_offset == NO_PARENT

        agent1 = next(a for a in agents if a.slot_index == 1)
        assert agent1.parent_offset == 0
        assert agent1.generation == 1
    finally:
        ledger.close()


def test_load_active_agents_does_not_mutate_the_ledger(tmp_path):
    ledger = _seeded_ledger(tmp_path)
    try:
        before = ledger.stats()
        load_active_agents(ledger)
        after = ledger.stats()
        assert before == after
    finally:
        ledger.close()


def _tiny_model() -> GH05T3BinaryOSS:
    return GH05T3BinaryOSS(num_layers=1, dim=16, num_heads=2, vocab_size=20, binary_ratio=0.95, stabilizer="mgc")


def test_agent_swarm_runtime_runs_a_real_batched_forward_pass(tmp_path):
    ledger = _seeded_ledger(tmp_path)
    try:
        agents = load_active_agents(ledger)
        model = _tiny_model()
        runtime = AgentSwarmRuntime(model)

        seq_len, vocab_size = 6, 20
        input_ids = torch.randint(0, vocab_size, (len(agents), seq_len))

        outputs = runtime.run_batch(agents, input_ids)

        assert set(outputs.keys()) == {a.slot_index for a in agents}
        for slot_index, logits in outputs.items():
            assert logits.shape == (seq_len, vocab_size)
            assert torch.isfinite(logits).all()
    finally:
        ledger.close()


def test_agent_swarm_runtime_rejects_mismatched_batch_size(tmp_path):
    ledger = _seeded_ledger(tmp_path)
    try:
        agents = load_active_agents(ledger)
        runtime = AgentSwarmRuntime(_tiny_model())

        wrong_input = torch.randint(0, 20, (len(agents) + 1, 6))
        try:
            runtime.run_batch(agents, wrong_input)
            assert False, "expected ValueError for mismatched batch size"
        except ValueError:
            pass
    finally:
        ledger.close()


def test_different_agents_get_different_real_outputs(tmp_path):
    """Sanity check that outputs are actually per-row, not accidentally
    identical/broadcast across the batch."""
    ledger = _seeded_ledger(tmp_path)
    try:
        agents = load_active_agents(ledger)
        runtime = AgentSwarmRuntime(_tiny_model())

        seq_len, vocab_size = 6, 20
        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab_size, (len(agents), seq_len))

        outputs = runtime.run_batch(agents, input_ids)
        values = list(outputs.values())
        assert not torch.equal(values[0], values[1])
    finally:
        ledger.close()
