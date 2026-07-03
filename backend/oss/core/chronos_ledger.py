"""
ChronosLedger — mmap binary state ledger for the Aethyro Execution Plane.

32-byte agent layout (perfect cache-line alignment, 2 agents per 64-byte L2 line):
  7 × float16  desires       (14 bytes, 0-13)  — KNOWLEDGE/SKILL/STATUS/EXPERIENCE/CREATION/CONNECTION/FREEDOM
  1 × uint16   maturity      ( 2 bytes, 14-15) — context maturity level 1–8
  1 × float16  fitness       ( 2 bytes, 16-17) — current fitness [0,1]
  1 × uint16   parent_offset ( 2 bytes, 18-19) — ledger index of parent; 0xFFFF = genesis seed
  1 × uint8    generation    ( 1 byte,  20)    — lineage depth (wraps at 255)
  3s heartbeat               ( 3 bytes, 21-23) — last-seen tick mod 2^24 (little-endian uint24)
  1 × uint64   scratchpad    ( 8 bytes, 24-31) — bit-flagged agent traits

Total: 32 bytes/agent. 2 agents per 64-byte cache line (no split-load penalty).
10,000 agents = 320KB — fits in 1MB L2 cache with headroom.

Scratchpad bit flags (see SCRATCH_* constants):
  bit 0     — SCRATCH_LOCKED        agent is frozen, skip mutation
  bit 1     — SCRATCH_PATENTED      a LexGenSeal record exists for this slot
  bit 2     — SCRATCH_NEEDS_REVIEW  flagged for patent-office review
  bits 3-5  — SCRATCH_UNIVERSE_MASK universe ID 0-7 (BME)
  bits 6-8  — SCRATCH_ROLE_TIER_MASK role tier 0-7 (BME)
  bit 9     — SCRATCH_MIGRANT       migrated from another universe (BME)
  bit 10    — SCRATCH_SPECIATION    triggered speciation event (BME)
  bit 11    — SCRATCH_ELITE_PROPOSAL has proposed a new skill (BME)
  bit 12    — SCRATCH_BREAKTHROUGH_GENE genome contrib >= 0.80 (BME)
  bits 13-63 — reserved

Lineage: parent_offset stores ledger *index* of parent. Sentinel 0xFFFF = no parent.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("ghost.chronos_ledger")

# ── Binary layout ─────────────────────────────────────────────────────────────
# 7×e(float16) + H(uint16 maturity) + e(float16 fitness) + H(uint16 parent)
# + B(uint8 gen) + 3s(heartbeat) + Q(uint64 scratchpad)
AGENT_STRUCT = "eeeeeeeHeHB3sQ"
STRUCT_SIZE  = struct.calcsize(AGENT_STRUCT)   # 32 bytes — verified

AGENT_DTYPE = np.dtype([
    ("desires",       np.float16, (7,)),
    ("maturity",      np.uint16),
    ("fitness",       np.float16),
    ("parent_offset", np.uint16),
    ("generation",    np.uint8),
    ("_heartbeat",    np.uint8,  (3,)),
    ("scratchpad",    np.uint64),
])
assert AGENT_DTYPE.itemsize == STRUCT_SIZE, (
    f"dtype/struct mismatch: {AGENT_DTYPE.itemsize} != {STRUCT_SIZE}"
)

# Byte offsets inside the 32-byte slot
_OFF_MATURITY       = 14   # bytes 14-15  uint16
_OFF_FITNESS        = 16   # bytes 16-17  float16
_OFF_PARENT_OFFSET  = 18   # bytes 18-19  uint16
_OFF_GENERATION     = 20   # byte  20     uint8
_OFF_HEARTBEAT      = 21   # bytes 21-23  3s (uint24 little-endian)
_OFF_SCRATCHPAD     = 24   # bytes 24-31  uint64

_NO_PARENT          = 0xFFFF           # sentinel: genesis-seed agent
_ZERO_PAD           = b"\x00\x00\x00"  # 3-byte heartbeat zero

# Scratchpad bit flags — bits 0-2 (original)
SCRATCH_LOCKED        = 1 << 0   # agent frozen — skip mutation
SCRATCH_PATENTED      = 1 << 1   # LexGenSeal record exists for this slot
SCRATCH_NEEDS_REVIEW  = 1 << 2   # flagged for patent-office review

# BME extension — bits 3-12 (Binary Multiverse Engine)
SCRATCH_UNIVERSE_MASK     = 0b111 << 3   # bits 3-5: universe ID (0-7)
SCRATCH_ROLE_TIER_MASK    = 0b111 << 6   # bits 6-8: role tier (0-7)
SCRATCH_MIGRANT           = 1 << 9       # migrated from another universe this generation
SCRATCH_SPECIATION        = 1 << 10      # triggered a speciation event (tier >= 4)
SCRATCH_ELITE_PROPOSAL    = 1 << 11      # has proposed a new skill to SkillRegistry
SCRATCH_BREAKTHROUGH_GENE = 1 << 12      # genome fitness contribution >= 0.80

UNIVERSE_SHIFT  = 3   # bit offset for universe ID within SCRATCH_UNIVERSE_MASK
ROLE_TIER_SHIFT = 6   # bit offset for role tier within SCRATCH_ROLE_TIER_MASK

DESIRE_NAMES = ["KNOWLEDGE", "SKILL", "STATUS", "EXPERIENCE", "CREATION", "CONNECTION", "FREEDOM"]

_DEFAULT_LEDGER = Path("data/aethyro_swarm.bin")
_DEFAULT_CAP    = 10_000


def _f16(v: float) -> float:
    """Quantize a float to float16 precision before packing."""
    return float(np.float16(np.clip(v, 0.0, 1.0)))


class ChronosLedger:
    """
    Memory-mapped 32-byte agent state store.

    At 32 bytes/agent, two agents fit exactly in one 64-byte cache line —
    zero split-load penalty. 10,000 agents = 320KB, well within the 1MB L2
    on Ryzen 7 8000-series. Genesis Thread numpy sweeps run at near-register speed.

    The uint64 scratchpad stores bit-flagged trait signals (locked, patented,
    needs_review) without requiring additional ledger memory or a separate map.
    """

    def __init__(
        self,
        filename: Optional[Path] = None,
        capacity: int = _DEFAULT_CAP,
    ):
        self.capacity  = capacity
        self._path     = Path(filename) if filename else _DEFAULT_LEDGER
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._size     = capacity * STRUCT_SIZE
        self._fd: Optional[int]        = None
        self._map: Optional[mmap.mmap] = None
        self._active: int = 0
        self._open()

    @classmethod
    def load(
        cls,
        filename: Optional[Path] = None,
        capacity: int = _DEFAULT_CAP,
    ) -> "ChronosLedger":
        """Compatibility helper for router-style callers."""
        return cls(filename=filename, capacity=capacity)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _open(self) -> None:
        self._fd = os.open(str(self._path), os.O_RDWR | os.O_CREAT)
        os.ftruncate(self._fd, self._size)
        self._map = mmap.mmap(self._fd, self._size)
        LOG.info(
            "ChronosLedger: %s  cap=%d  slot=%dB  total=%dKB  agents/cacheline=2",
            self._path, self.capacity, STRUCT_SIZE, self._size // 1024,
        )

    def close(self) -> None:
        if self._map:
            self._map.flush()
            self._map.close()
            self._map = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── Full-slot write ───────────────────────────────────────────────────────

    def write_agent(
        self,
        index:         int,
        desires:       Tuple[float, ...],
        maturity:      int,
        fitness:       float,
        parent_offset: int = _NO_PARENT,
        generation:    int = 0,
        scratchpad:    int = 0,
    ) -> None:
        if index >= self.capacity:
            raise IndexError(f"Slot {index} >= capacity {self.capacity}")
        d = tuple(_f16(v) for v in list(desires)[:7]) + (0.1,) * max(0, 7 - len(desires))
        struct.pack_into(
            AGENT_STRUCT, self._map, index * STRUCT_SIZE,
            *d,
            int(maturity)      & 0xFFFF,
            _f16(fitness),
            int(parent_offset) & 0xFFFF,
            int(generation)    & 0xFF,
            _ZERO_PAD,
            int(scratchpad)    & 0xFFFFFFFFFFFFFFFF,
        )
        if index >= self._active:
            self._active = index + 1

    def write_zero(self, index: int) -> None:
        """Zero a slot (fitness=0.0 acts as the vacancy sentinel)."""
        struct.pack_into(
            AGENT_STRUCT, self._map, index * STRUCT_SIZE,
            *(0.1,) * 7, 0, 0.0, _NO_PARENT, 0, _ZERO_PAD, 0,
        )

    # ── Vacancy-scan slot allocation ──────────────────────────────────────────

    def find_vacant_slot(self, start: int = 0) -> int:
        """
        Scan from `start` for the first slot with fitness == 0.0 (vacancy marker).

        write_zero() and release_slot() set fitness=0.0. Falls back to the
        monotonic _active counter when no freed slot is found.

        Raises MemoryError when every slot is occupied.
        """
        limit = min(self._active, self.capacity)
        for i in range(start, limit):
            v = struct.unpack_from("e", self._map, i * STRUCT_SIZE + _OFF_FITNESS)[0]
            if v == 0.0:
                return i
        if self._active < self.capacity:
            return self._active
        raise MemoryError(f"Swarm at capacity ({self.capacity} agents). Pruning required.")

    def write_at_next_available_slot(
        self,
        desires:       Tuple[float, ...],
        maturity:      int,
        fitness:       float,
        parent_offset: int = _NO_PARENT,
        generation:    int = 0,
        scratchpad:    int = 0,
        start:         int = 0,
    ) -> int:
        """
        Atomic vacancy-scan write. Finds the first free slot (fitness==0.0),
        writes agent data, returns the slot index.
        """
        slot = self.find_vacant_slot(start=start)
        self.write_agent(slot, desires, maturity, fitness, parent_offset, generation, scratchpad)
        return slot

    # ── Atomic in-place updates ───────────────────────────────────────────────

    def update_fitness(self, index: int, fitness: float) -> None:
        """Write only 2 bytes at the fitness offset — minimal cache footprint."""
        struct.pack_into("e", self._map, index * STRUCT_SIZE + _OFF_FITNESS, _f16(fitness))

    def update_maturity(self, index: int, maturity: int) -> None:
        struct.pack_into("H", self._map, index * STRUCT_SIZE + _OFF_MATURITY, int(maturity) & 0xFFFF)

    def update_generation(self, index: int, generation: int) -> None:
        struct.pack_into("B", self._map, index * STRUCT_SIZE + _OFF_GENERATION, int(generation) & 0xFF)

    def update_heartbeat(self, index: int, tick: int) -> None:
        """Store tick counter (mod 2^24) in the 3-byte heartbeat field."""
        hb = (tick & 0xFFFFFF).to_bytes(3, "little")
        struct.pack_into("3s", self._map, index * STRUCT_SIZE + _OFF_HEARTBEAT, hb)

    # ── Scratchpad bit manipulation ───────────────────────────────────────────

    def set_scratch_bit(self, index: int, bit: int) -> None:
        """Set one or more scratchpad flag bits (OR operation)."""
        current = struct.unpack_from("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD)[0]
        struct.pack_into("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD, current | bit)

    def clear_scratch_bit(self, index: int, bit: int) -> None:
        """Clear one or more scratchpad flag bits (AND NOT operation)."""
        current = struct.unpack_from("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD)[0]
        struct.pack_into("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD, current & ~bit)

    def get_scratch_bit(self, index: int, bit: int) -> bool:
        v = struct.unpack_from("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD)[0]
        return bool(v & bit)

    def update_scratchpad(self, index: int, value: int) -> None:
        """Overwrite the entire scratchpad uint64."""
        struct.pack_into("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD,
                         int(value) & 0xFFFFFFFFFFFFFFFF)

    # Aliases used by BMEBridge (shorter names, same semantics)
    def get_scratchpad(self, index: int) -> int:
        return struct.unpack_from("Q", self._map, index * STRUCT_SIZE + _OFF_SCRATCHPAD)[0]

    def set_scratchpad(self, index: int, value: int) -> None:
        self.update_scratchpad(index, value)

    def get_desires(self, index: int) -> np.ndarray:
        """Return desires as float16 numpy array (7,)."""
        raw = struct.unpack_from("7e", self._map, index * STRUCT_SIZE)
        return np.array(raw, dtype=np.float16)

    def get_fitness(self, index: int) -> float:
        return float(struct.unpack_from("e", self._map, index * STRUCT_SIZE + _OFF_FITNESS)[0])

    def get_generation(self, index: int) -> int:
        return int(struct.unpack_from("B", self._map, index * STRUCT_SIZE + _OFF_GENERATION)[0])

    # BME universe helpers — convenience wrappers over scratchpad bitmasks
    def get_universe(self, index: int) -> int:
        return (self.get_scratchpad(index) & SCRATCH_UNIVERSE_MASK) >> UNIVERSE_SHIFT

    def set_universe(self, index: int, universe_id: int) -> None:
        s = self.get_scratchpad(index) & ~SCRATCH_UNIVERSE_MASK
        self.set_scratchpad(index, s | ((universe_id & 0b111) << UNIVERSE_SHIFT))

    def get_role_tier(self, index: int) -> int:
        return (self.get_scratchpad(index) & SCRATCH_ROLE_TIER_MASK) >> ROLE_TIER_SHIFT

    def set_role_tier(self, index: int, tier: int) -> None:
        s = self.get_scratchpad(index) & ~SCRATCH_ROLE_TIER_MASK
        self.set_scratchpad(index, s | ((tier & 0b111) << ROLE_TIER_SHIFT))

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_agent(self, index: int) -> Dict[str, object]:
        if index >= self.capacity:
            raise IndexError(f"Slot {index} >= capacity {self.capacity}")
        v = struct.unpack_from(AGENT_STRUCT, self._map, index * STRUCT_SIZE)
        return {
            "desires":       dict(zip(DESIRE_NAMES, [round(float(x), 4) for x in v[:7]])),
            "maturity":      int(v[7]),
            "fitness":       round(float(v[8]), 4),
            "parent_offset": int(v[9]),
            "generation":    int(v[10]),
            "has_parent":    int(v[9]) != _NO_PARENT,
            "heartbeat":     int.from_bytes(v[11], "little"),
            "scratchpad":    int(v[12]),
            "is_locked":     bool(int(v[12]) & SCRATCH_LOCKED),
            "is_patented":   bool(int(v[12]) & SCRATCH_PATENTED),
            "needs_review":  bool(int(v[12]) & SCRATCH_NEEDS_REVIEW),
        }

    def get_lineage(self, index: int, max_depth: int = 10) -> List[int]:
        """Walk parent_offset pointers. Returns ancestor chain [index, parent, grandparent ...]."""
        chain, seen, current = [index], {index}, index
        for _ in range(max_depth):
            v      = struct.unpack_from(AGENT_STRUCT, self._map, current * STRUCT_SIZE)
            parent = int(v[9])
            if parent == _NO_PARENT or parent in seen or parent >= self.capacity:
                break
            chain.append(parent)
            seen.add(parent)
            current = parent
        return chain

    # ── Numpy views ───────────────────────────────────────────────────────────

    def to_numpy(self, active_only: bool = True) -> np.ndarray:
        """Structured numpy array. arr['desires'] → shape (n, 7) float16."""
        self._map.seek(0)
        raw = self._map.read(self._size)
        arr = np.frombuffer(raw, dtype=AGENT_DTYPE)
        return arr[: self._active] if (active_only and self._active > 0) else arr

    def desires_matrix(self) -> np.ndarray:
        """(active, 7) float32 desires — upcast from float16, NaN/Inf sanitized."""
        arr = self.to_numpy()
        if len(arr) == 0:
            return np.zeros((1, 7), dtype=np.float32)
        raw = arr["desires"].astype(np.float32)
        return np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)

    def fitness_vector(self) -> np.ndarray:
        """(active,) float32 fitness values."""
        return self.to_numpy()["fitness"].astype(np.float32)

    def scratchpad_vector(self) -> np.ndarray:
        """(active,) uint64 scratchpad values — for bulk flag inspection."""
        return self.to_numpy()["scratchpad"]

    def find_descendants(self, ancestor_index: int) -> np.ndarray:
        """Vectorized: all slots whose parent_offset == ancestor_index."""
        return np.where(self.to_numpy()["parent_offset"] == ancestor_index)[0]

    def find_by_scratch_bit(self, bit: int) -> np.ndarray:
        """Vectorized: all slots with a specific scratchpad bit set."""
        scratch = self.scratchpad_vector()
        return np.where(scratch & bit)[0]

    # ── Misc ──────────────────────────────────────────────────────────────────

    @property
    def active_slots(self) -> int:
        return self._active

    def flush(self) -> None:
        if self._map:
            self._map.flush()

    def stats(self) -> Dict[str, object]:
        if self._active == 0:
            return {"active_slots": 0, "capacity": self.capacity,
                    "utilization_pct": 0.0, "slot_bytes": STRUCT_SIZE}
        arr = self.to_numpy()
        f32 = arr["fitness"].astype(np.float32)
        scratch = arr["scratchpad"]
        return {
            "active_slots":    self._active,
            "capacity":        self.capacity,
            "slot_bytes":      STRUCT_SIZE,
            "total_bytes":     self._active * STRUCT_SIZE,
            "utilization_pct": round(self._active / self.capacity * 100, 2),
            "mean_fitness":    round(float(f32.mean()), 4),
            "max_fitness":     round(float(f32.max()), 4),
            "mean_maturity":   round(float(arr["maturity"].astype(np.float32).mean()), 2),
            "mean_generation": round(float(arr["generation"].astype(np.float32).mean()), 2),
            "mean_desires":    {
                k: round(float(v), 4)
                for k, v in zip(DESIRE_NAMES, arr["desires"].astype(np.float32).mean(axis=0))
            },
            "patented_count":  int(np.sum((scratch & SCRATCH_PATENTED) != 0)),
            "locked_count":    int(np.sum((scratch & SCRATCH_LOCKED)   != 0)),
            "review_count":    int(np.sum((scratch & SCRATCH_NEEDS_REVIEW) != 0)),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_ledger: Optional[ChronosLedger] = None


def get_chronos_ledger() -> ChronosLedger:
    global _ledger
    if _ledger is None:
        _ledger = ChronosLedger()
    return _ledger
