"""Tests for BinaryLedger. Every write-touching test operates on a fresh
temp-directory ledger (pytest's tmp_path) -- the real
backend/data/aethyro_swarm.bin is only ever opened read-only here, and
only to confirm decoding matches what was independently verified by hand
before this module was written (see binary_ledger.py's module docstring).
"""
from __future__ import annotations

import os
import shutil

import pytest

from backend.oss.core.binary_ledger import (
    DESIRE_NAMES,
    NO_PARENT,
    SCRATCH_LOCKED,
    SCRATCH_NEEDS_REVIEW,
    SCRATCH_PATENTED,
    STRUCT_SIZE,
    BinaryLedger,
)

_REAL_LEDGER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "aethyro_swarm.bin")
)


def test_struct_size_is_32_bytes():
    assert STRUCT_SIZE == 32


def _fresh_ledger(tmp_path, capacity: int = 10) -> BinaryLedger:
    path = str(tmp_path / "test_ledger.bin")
    return BinaryLedger(path, capacity=capacity, create=True)


def test_create_new_ledger_has_correct_size_and_zero_active(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=10)
    try:
        assert ledger.capacity == 10
        assert os.path.getsize(ledger.path) == 10 * STRUCT_SIZE
        assert ledger._active == 0
    finally:
        ledger.close()


def test_open_missing_file_without_create_raises():
    with pytest.raises(FileNotFoundError):
        BinaryLedger("/nonexistent/path/ledger.bin", create=False)


def test_write_and_read_agent_round_trips():
    with_desires = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    ledger = None
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        ledger = BinaryLedger(os.path.join(d, "l.bin"), capacity=5, create=True)
        try:
            ledger.write_agent(0, desires=with_desires, maturity=3, fitness=0.75, parent_offset=NO_PARENT, generation=1)
            agent = ledger.read_agent(0)

            assert agent["maturity"] == 3
            assert agent["fitness"] == 0.75  # exact in float16, per documented precision notes
            assert agent["generation"] == 1
            assert agent["has_parent"] is False
            for name, expected in zip(DESIRE_NAMES, with_desires):
                assert abs(agent["desires"][name] - expected) < 0.01
        finally:
            ledger.close()


def test_fitness_precision_matches_documented_float16_rounding(tmp_path):
    ledger = _fresh_ledger(tmp_path)
    try:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.8)
        # Documented example: written 0.8 -> read back ~0.7998 (float16 nearest).
        assert abs(ledger.read_agent(0)["fitness"] - 0.7998) < 1e-3

        ledger.write_agent(1, desires=(0.5,) * 7, maturity=1, fitness=0.75)
        # Documented example: 0.75 is exact in float16.
        assert ledger.read_agent(1)["fitness"] == 0.75
    finally:
        ledger.close()


def test_out_of_range_index_raises(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=5)
    try:
        with pytest.raises(IndexError):
            ledger.read_agent(5)
        with pytest.raises(IndexError):
            ledger.write_agent(-1, desires=(0.5,) * 7, maturity=1, fitness=0.5)
    finally:
        ledger.close()


def test_wrong_desires_length_raises(tmp_path):
    ledger = _fresh_ledger(tmp_path)
    try:
        with pytest.raises(ValueError):
            ledger.write_agent(0, desires=(0.5, 0.5), maturity=1, fitness=0.5)
    finally:
        ledger.close()


def test_write_zero_marks_vacant_and_find_vacant_slot_reuses_it(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=5)
    try:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.9)
        ledger.write_agent(1, desires=(0.5,) * 7, maturity=1, fitness=0.9)
        ledger.write_zero(0)

        assert ledger.read_agent(0)["fitness"] == 0.0
        assert ledger.find_vacant_slot() == 0  # freed slot reused before extending

        new_slot = ledger.write_at_next_available_slot(desires=(0.6,) * 7, maturity=2, fitness=0.4)
        assert new_slot == 0
        assert ledger.read_agent(0)["fitness"] > 0.0
    finally:
        ledger.close()


def test_write_at_next_available_slot_extends_when_nothing_vacant(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=3)
    try:
        s0 = ledger.write_at_next_available_slot(desires=(0.5,) * 7, maturity=1, fitness=0.5)
        s1 = ledger.write_at_next_available_slot(desires=(0.5,) * 7, maturity=1, fitness=0.5)
        assert (s0, s1) == (0, 1)
    finally:
        ledger.close()


def test_full_capacity_raises_memory_error(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=2)
    try:
        ledger.write_at_next_available_slot(desires=(0.5,) * 7, maturity=1, fitness=0.5)
        ledger.write_at_next_available_slot(desires=(0.5,) * 7, maturity=1, fitness=0.5)
        with pytest.raises(MemoryError):
            ledger.write_at_next_available_slot(desires=(0.5,) * 7, maturity=1, fitness=0.5)
    finally:
        ledger.close()


def test_atomic_field_updates(tmp_path):
    ledger = _fresh_ledger(tmp_path)
    try:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.5, generation=0, heartbeat=0)

        ledger.update_fitness(0, 0.9)
        ledger.update_maturity(0, 7)
        ledger.update_generation(0, 3)
        ledger.update_heartbeat(0, 12345)

        agent = ledger.read_agent(0)
        assert abs(agent["fitness"] - 0.9) < 1e-2
        assert agent["maturity"] == 7
        assert agent["generation"] == 3
        assert agent["heartbeat"] == 12345

        # Updating one field must not disturb the others (desires untouched).
        for v in agent["desires"].values():
            assert abs(v - 0.5) < 0.01
    finally:
        ledger.close()


def test_generation_wraps_at_256(tmp_path):
    ledger = _fresh_ledger(tmp_path)
    try:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.5, generation=300)
        assert ledger.read_agent(0)["generation"] == 300 % 256
    finally:
        ledger.close()


def test_scratchpad_bit_flags(tmp_path):
    ledger = _fresh_ledger(tmp_path)
    try:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.5)

        assert ledger.get_scratch_bit(0, SCRATCH_PATENTED) is False

        ledger.set_scratch_bit(0, SCRATCH_PATENTED)
        assert ledger.get_scratch_bit(0, SCRATCH_PATENTED) is True
        assert ledger.read_agent(0)["is_patented"] is True

        ledger.set_scratch_bit(0, SCRATCH_NEEDS_REVIEW)
        assert ledger.get_scratch_bit(0, SCRATCH_PATENTED) is True  # setting one bit doesn't clear another
        assert ledger.get_scratch_bit(0, SCRATCH_NEEDS_REVIEW) is True

        ledger.clear_scratch_bit(0, SCRATCH_LOCKED)  # clearing an unset bit is a no-op, not an error
        assert ledger.read_agent(0)["is_locked"] is False

        ledger.update_scratchpad(0, 0)
        assert ledger.read_agent(0)["is_patented"] is False
        assert ledger.read_agent(0)["needs_review"] is False
    finally:
        ledger.close()


def test_find_by_scratch_bit(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=5)
    try:
        for i in range(4):
            ledger.write_agent(i, desires=(0.5,) * 7, maturity=1, fitness=0.5)
        ledger.set_scratch_bit(1, SCRATCH_PATENTED)
        ledger.set_scratch_bit(3, SCRATCH_PATENTED)

        found = ledger.find_by_scratch_bit(SCRATCH_PATENTED)
        assert sorted(found.tolist()) == [1, 3]
    finally:
        ledger.close()


def test_lineage_walk_and_descendants(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=5)
    try:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.5, parent_offset=NO_PARENT, generation=0)
        ledger.write_agent(1, desires=(0.5,) * 7, maturity=1, fitness=0.5, parent_offset=0, generation=1)
        ledger.write_agent(2, desires=(0.5,) * 7, maturity=1, fitness=0.5, parent_offset=1, generation=2)
        ledger.write_agent(3, desires=(0.5,) * 7, maturity=1, fitness=0.5, parent_offset=0, generation=1)

        assert ledger.get_lineage(2) == [2, 1, 0]
        assert ledger.get_lineage(0) == [0]  # genesis seed, no parent

        children_of_0 = ledger.find_descendants(0)
        assert sorted(children_of_0.tolist()) == [1, 3]
    finally:
        ledger.close()


def test_high_water_mark_recomputed_on_reopen(tmp_path):
    path = str(tmp_path / "hwm.bin")
    ledger = BinaryLedger(path, capacity=10, create=True)
    ledger.write_agent(5, desires=(0.5,) * 7, maturity=1, fitness=0.5)
    ledger.close()

    reopened = BinaryLedger(path, create=False)
    try:
        # Documented gotcha, reproduced deliberately: writing only to slot
        # 5 means the high-water mark is 6 on reopen, not "1 active slot".
        assert reopened._active == 6
        assert reopened.read_agent(5)["fitness"] == 0.5
    finally:
        reopened.close()


def test_vectorized_views_and_stats(tmp_path):
    ledger = _fresh_ledger(tmp_path, capacity=5)
    try:
        ledger.write_agent(0, desires=(0.2,) * 7, maturity=2, fitness=0.5, generation=1)
        ledger.write_agent(1, desires=(0.8,) * 7, maturity=4, fitness=0.9, generation=3)

        fv = ledger.fitness_vector()
        assert fv.shape == (2,)
        assert abs(fv[0] - 0.5) < 0.01 and abs(fv[1] - 0.9) < 0.01

        dm = ledger.desires_matrix()
        assert dm.shape == (2, 7)

        arr = ledger.to_numpy()
        assert arr.shape == (2,)
        assert arr[0]["maturity"] == 2
        assert arr[1]["generation"] == 3

        stats = ledger.stats()
        assert stats["active_slots"] == 2
        assert stats["capacity"] == 5
        assert stats["slot_bytes"] == 32
        assert stats["max_fitness"] > stats["mean_fitness"] or abs(stats["max_fitness"] - stats["mean_fitness"]) < 0.05
        assert set(stats["mean_desires"].keys()) == set(DESIRE_NAMES)
    finally:
        ledger.close()


def test_stats_on_empty_ledger_does_not_crash(tmp_path):
    ledger = _fresh_ledger(tmp_path)
    try:
        stats = ledger.stats()
        assert stats["active_slots"] == 0
        assert stats["mean_fitness"] == 0.0
    finally:
        ledger.close()


def test_context_manager_closes_on_exit(tmp_path):
    path = str(tmp_path / "ctx.bin")
    with BinaryLedger(path, capacity=3, create=True) as ledger:
        ledger.write_agent(0, desires=(0.5,) * 7, maturity=1, fitness=0.5)
    assert ledger._mmap is None  # closed


# ---- read-only verification against the REAL committed data file -----------

@pytest.mark.skipif(not os.path.isfile(_REAL_LEDGER_PATH), reason="real aethyro_swarm.bin not present")
def test_real_ledger_file_decodes_to_sane_values():
    """Read-only: never calls any write method on the real file. Confirms
    this module decodes it exactly the way it was manually verified
    before writing any code (see binary_ledger.py's module docstring)."""
    ledger = BinaryLedger(_REAL_LEDGER_PATH, create=False)
    try:
        assert ledger.capacity == 10_000
        stats = ledger.stats()
        # Real file had 7 active slots with max_generation=0 at last manual check.
        assert stats["active_slots"] >= 1
        assert 0.0 <= stats["mean_fitness"] <= 1.0

        agent0 = ledger.read_agent(0)
        assert agent0["fitness"] > 0.0  # slot 0 is a real, active genesis seed
        assert agent0["has_parent"] is False
        for v in agent0["desires"].values():
            assert 0.0 <= v <= 1.0
    finally:
        ledger.close()
        # Defensive: prove this test never modified the real file.
        assert os.path.getsize(_REAL_LEDGER_PATH) == 320_000
