"""Real, mmap-backed binary agent-swarm ledger.

Implements the documented ChronosLedger wire format exactly (see the
binary-ledger skill: /mnt/c/Users/leer4/GH05T3/.claude/skills/binary-ledger/
SKILL.md), verified independently against the real aethyro_swarm.bin file
before writing a line of this: struct.calcsize("eeeeeeeHeHB3sQ") == 32,
confirmed against the file's real size (320,000 = 10,000 * 32), and real
decoded slot values are sane (desires in [0,1] centered at 0.5, fitness
in [0,1], parent_offset == 0xFFFF for genesis-seed agents).

Named BinaryLedger, not ChronosLedger, and lives at a distinct path from
core/chronos_ledger.py -- that name and path are already taken by this
session's real, tested, live-API-wired ChronosLedger, which tracks a
semantically DIFFERENT kind of "genome": neural-network architecture
variants (num_layers/dim/stabilizer/binary_ratio) for the binary/ternary
transformer. This ledger encodes agent SWARM/PERSONA state instead: 7
psychological "desire" dimensions, maturity, fitness, lineage via a
parent slot index, a heartbeat tick, and bit-flagged scratchpad traits.
They share evolutionary vocabulary (fitness, generation, mutation) but
are not the same thing and are not merged here.
"""
from __future__ import annotations

import mmap
import os
import struct
from typing import Any, Optional

import numpy as np

AGENT_STRUCT = "eeeeeeeHeHB3sQ"
STRUCT_SIZE = struct.calcsize(AGENT_STRUCT)
assert STRUCT_SIZE == 32, f"AGENT_STRUCT must pack to 32 bytes, got {STRUCT_SIZE}"

DEFAULT_CAPACITY = 10_000

DESIRE_NAMES = ("KNOWLEDGE", "SKILL", "STATUS", "EXPERIENCE", "CREATION", "CONNECTION", "FREEDOM")

NO_PARENT = 0xFFFF  # genesis-seed sentinel for parent_offset

SCRATCH_LOCKED = 1 << 0
SCRATCH_PATENTED = 1 << 1
SCRATCH_NEEDS_REVIEW = 1 << 2

_ZERO_SLOT = b"\x00" * STRUCT_SIZE


def _f16(value: float) -> float:
    """Clips to [0,1] and round-trips through float16 -- matches the
    documented precision behavior exactly: raw floats outside [0,1] are
    clipped silently, never written as-is."""
    clipped = max(0.0, min(1.0, float(value)))
    return float(np.float16(clipped))


class BinaryLedger:
    """mmap-backed binary agent ledger. One 32-byte slot per agent; file
    size determines slot count (10,000 slots = 320KB matches the real
    aethyro_swarm.bin this was verified against)."""

    def __init__(self, path: str, capacity: int = DEFAULT_CAPACITY, create: bool = False):
        self.path = path
        file_exists = os.path.isfile(path)

        if create and not file_exists:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.write(_ZERO_SLOT * capacity)
        elif not file_exists:
            raise FileNotFoundError(
                f"ledger file not found: {path} (pass create=True to initialize a new one)"
            )

        self._fd = os.open(path, os.O_RDWR)
        file_size = os.fstat(self._fd).st_size
        if file_size == 0 or file_size % STRUCT_SIZE != 0:
            os.close(self._fd)
            raise ValueError(f"ledger file size {file_size} is not a positive multiple of {STRUCT_SIZE} bytes")

        self.capacity = file_size // STRUCT_SIZE
        self._mmap: Optional[mmap.mmap] = mmap.mmap(self._fd, file_size)
        self._active = self._scan_high_water_mark()

    def _scan_high_water_mark(self) -> int:
        """Real documented gotcha: `_active` isn't persisted in the file
        -- it's recomputed on open by scanning for the highest slot index
        whose raw bytes aren't all zero (i.e. has ever been written, even
        if since freed back to a fitness=0.0 vacancy). Everything past
        that index is guaranteed never-touched."""
        high_water = 0
        for i in range(self.capacity):
            if self._mmap[i * STRUCT_SIZE:(i + 1) * STRUCT_SIZE] != _ZERO_SLOT:
                high_water = i + 1
        return high_water

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "BinaryLedger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def active_count(self) -> int:
        """The high-water mark (see _scan_high_water_mark) -- how many
        slots from index 0 have ever been written, not how many are
        currently non-vacant. Use stats()["active_slots"] for the latter."""
        return self._active

    def _check_index(self, index: int) -> None:
        if not (0 <= index < self.capacity):
            raise IndexError(f"slot {index} out of range [0, {self.capacity})")

    def _raw(self, index: int) -> tuple:
        return struct.unpack_from(AGENT_STRUCT, self._mmap, index * STRUCT_SIZE)

    def _unpack(self, index: int) -> dict[str, Any]:
        raw = self._raw(index)
        desires = dict(zip(DESIRE_NAMES, raw[0:7]))
        parent_offset = raw[9]
        scratchpad = raw[12]
        return {
            "desires": desires,
            "maturity": raw[7],
            "fitness": raw[8],
            "parent_offset": parent_offset,
            "has_parent": parent_offset != NO_PARENT,
            "generation": raw[10],
            "heartbeat": int.from_bytes(raw[11], "little"),
            "scratchpad": scratchpad,
            "is_locked": bool(scratchpad & SCRATCH_LOCKED),
            "is_patented": bool(scratchpad & SCRATCH_PATENTED),
            "needs_review": bool(scratchpad & SCRATCH_NEEDS_REVIEW),
        }

    def _pack(
        self,
        desires: tuple,
        maturity: int,
        fitness: float,
        parent_offset: int,
        generation: int,
        heartbeat: int,
        scratchpad: int,
    ) -> bytes:
        packed_desires = tuple(_f16(x) for x in desires)
        heartbeat_bytes = (int(heartbeat) % (1 << 24)).to_bytes(3, "little")
        return struct.pack(
            AGENT_STRUCT,
            *packed_desires,
            int(maturity),
            _f16(fitness),
            int(parent_offset),
            int(generation) % 256,
            heartbeat_bytes,
            int(scratchpad),
        )

    # ---- core read/write -------------------------------------------------

    def read_agent(self, index: int) -> dict[str, Any]:
        self._check_index(index)
        return self._unpack(index)

    def write_agent(
        self,
        index: int,
        desires: tuple,
        maturity: int,
        fitness: float,
        parent_offset: int = NO_PARENT,
        generation: int = 0,
        heartbeat: int = 0,
        scratchpad: int = 0,
    ) -> None:
        self._check_index(index)
        if len(desires) != 7:
            raise ValueError(f"desires must have exactly 7 values (one per {DESIRE_NAMES}), got {len(desires)}")
        packed = self._pack(desires, maturity, fitness, parent_offset, generation, heartbeat, scratchpad)
        offset = index * STRUCT_SIZE
        self._mmap[offset:offset + STRUCT_SIZE] = packed
        if index + 1 > self._active:
            self._active = index + 1

    def write_zero(self, index: int) -> None:
        """Frees a slot: sets fitness=0.0, the documented vacancy
        sentinel. Preserves the rest of the slot's fields (matches the
        documented behavior -- only fitness marks vacancy)."""
        self._check_index(index)
        agent = self._unpack(index)
        self.write_agent(
            index,
            desires=tuple(agent["desires"].values()),
            maturity=agent["maturity"],
            fitness=0.0,
            parent_offset=agent["parent_offset"],
            generation=agent["generation"],
            heartbeat=agent["heartbeat"],
            scratchpad=agent["scratchpad"],
        )

    release_slot = write_zero  # documented alias

    def find_vacant_slot(self, start: int = 0) -> int:
        for i in range(start, self._active):
            if self._raw(i)[8] == 0.0:
                return i
        if self._active >= self.capacity:
            raise MemoryError(f"ledger is at full capacity ({self.capacity} slots)")
        return self._active

    def write_at_next_available_slot(
        self,
        desires: tuple,
        maturity: int,
        fitness: float,
        parent_offset: int = NO_PARENT,
        generation: int = 0,
        heartbeat: int = 0,
        scratchpad: int = 0,
    ) -> int:
        slot = self.find_vacant_slot()
        self.write_agent(slot, desires, maturity, fitness, parent_offset, generation, heartbeat, scratchpad)
        return slot

    # ---- atomic single-field updates --------------------------------------

    def update_fitness(self, index: int, value: float) -> None:
        self._check_index(index)
        offset = index * STRUCT_SIZE + 16
        self._mmap[offset:offset + 2] = struct.pack("e", _f16(value))

    def update_maturity(self, index: int, value: int) -> None:
        self._check_index(index)
        offset = index * STRUCT_SIZE + 14
        self._mmap[offset:offset + 2] = struct.pack("H", int(value))

    def update_generation(self, index: int, value: int) -> None:
        self._check_index(index)
        offset = index * STRUCT_SIZE + 20
        self._mmap[offset:offset + 1] = struct.pack("B", int(value) % 256)

    def update_heartbeat(self, index: int, tick: int) -> None:
        self._check_index(index)
        offset = index * STRUCT_SIZE + 21
        self._mmap[offset:offset + 3] = (int(tick) % (1 << 24)).to_bytes(3, "little")

    def update_scratchpad(self, index: int, value: int) -> None:
        self._check_index(index)
        offset = index * STRUCT_SIZE + 24
        self._mmap[offset:offset + 8] = struct.pack("Q", int(value))

    def get_scratch_bit(self, index: int, bit: int) -> bool:
        return bool(self.read_agent(index)["scratchpad"] & bit)

    def set_scratch_bit(self, index: int, bit: int) -> None:
        current = self.read_agent(index)["scratchpad"]
        self.update_scratchpad(index, current | bit)

    def clear_scratch_bit(self, index: int, bit: int) -> None:
        current = self.read_agent(index)["scratchpad"]
        self.update_scratchpad(index, current & ~bit)

    # ---- lineage -----------------------------------------------------------

    def get_lineage(self, slot: int, max_depth: int = 10) -> list[int]:
        self._check_index(slot)
        chain = [slot]
        current = slot
        for _ in range(max_depth - 1):
            parent = self._raw(current)[9]
            if parent == NO_PARENT:
                break
            chain.append(parent)
            current = parent
        return chain

    def find_descendants(self, ancestor_index: int) -> np.ndarray:
        result = [i for i in range(self._active) if self._raw(i)[9] == ancestor_index]
        return np.array(result, dtype=np.int64)

    def find_by_scratch_bit(self, bit: int) -> np.ndarray:
        result = [i for i in range(self._active) if self._raw(i)[12] & bit]
        return np.array(result, dtype=np.int64)

    # ---- vectorized views ----------------------------------------------------

    def fitness_vector(self) -> np.ndarray:
        values = [self._raw(i)[8] for i in range(self._active)]
        return np.nan_to_num(np.array(values, dtype=np.float32))

    def desires_matrix(self) -> np.ndarray:
        rows = [self._raw(i)[0:7] for i in range(self._active)]
        if not rows:
            return np.zeros((0, 7), dtype=np.float32)
        return np.nan_to_num(np.array(rows, dtype=np.float32))

    def scratchpad_vector(self) -> np.ndarray:
        values = [self._raw(i)[12] for i in range(self._active)]
        return np.array(values, dtype=np.uint64)

    AGENT_DTYPE = np.dtype([
        ("desires", np.float32, (7,)),
        ("maturity", np.uint16),
        ("fitness", np.float32),
        ("parent_offset", np.uint16),
        ("generation", np.uint8),
        ("_heartbeat", np.uint8, (3,)),
        ("scratchpad", np.uint64),
    ])

    def to_numpy(self) -> np.ndarray:
        out = np.zeros(self._active, dtype=self.AGENT_DTYPE)
        for i in range(self._active):
            raw = self._raw(i)
            out[i]["desires"] = raw[0:7]
            out[i]["maturity"] = raw[7]
            out[i]["fitness"] = raw[8]
            out[i]["parent_offset"] = raw[9]
            out[i]["generation"] = raw[10]
            out[i]["_heartbeat"] = list(raw[11])
            out[i]["scratchpad"] = raw[12]
        return out

    # ---- aggregate stats -------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        fv = self.fitness_vector()
        dm = self.desires_matrix()
        active_mask = fv != 0.0
        active_count = int(active_mask.sum())

        if active_count == 0:
            mean_fitness = max_fitness = mean_maturity = mean_generation = 0.0
            mean_desires = {name: 0.0 for name in DESIRE_NAMES}
        else:
            maturities = np.array([self._raw(i)[7] for i in range(self._active) if fv[i] != 0.0])
            generations = np.array([self._raw(i)[10] for i in range(self._active) if fv[i] != 0.0])
            mean_fitness = float(fv[active_mask].mean())
            max_fitness = float(fv[active_mask].max())
            mean_maturity = float(maturities.mean())
            mean_generation = float(generations.mean())
            mean_desires = {name: float(dm[active_mask, idx].mean()) for idx, name in enumerate(DESIRE_NAMES)}

        return {
            "active_slots": active_count,
            "capacity": self.capacity,
            "slot_bytes": STRUCT_SIZE,
            "utilization_pct": round(100.0 * active_count / self.capacity, 4) if self.capacity else 0.0,
            "mean_fitness": mean_fitness,
            "max_fitness": max_fitness,
            "mean_maturity": mean_maturity,
            "mean_generation": mean_generation,
            "mean_desires": mean_desires,
            "patented_count": int(len(self.find_by_scratch_bit(SCRATCH_PATENTED))),
            "locked_count": int(len(self.find_by_scratch_bit(SCRATCH_LOCKED))),
            "review_count": int(len(self.find_by_scratch_bit(SCRATCH_NEEDS_REVIEW))),
        }


_ledger_singleton: Optional[BinaryLedger] = None


def get_binary_ledger(path: Optional[str] = None) -> BinaryLedger:
    """Cached singleton, matching the documented get_chronos_ledger()
    convenience accessor -- renamed to avoid colliding with this
    session's unrelated, already-real ChronosLedger (core/chronos_ledger.py)."""
    global _ledger_singleton
    if _ledger_singleton is None:
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "aethyro_swarm.bin")
        _ledger_singleton = BinaryLedger(path)
    return _ledger_singleton
