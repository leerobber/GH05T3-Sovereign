"""
GenomePlane — agent skill genome, parallel mmap file to ChronosLedger.

genome_plane.bin layout:
  SLOT 0   [64 bytes]  ← 8 genes × 8 bytes
  SLOT 1   [64 bytes]
  ...
  SLOT N   [64 bytes]

Indexed by the same slot index used by ChronosLedger. The two files
share no structural dependency — genome_plane.bin can be rebuilt from
skill_registry + ChronosLedger state at any time.

Gene layout (8 bytes, "<BBHeB1s"):
  universe_id  uint8   1   which universe this gene belongs to (0-7)
  role_tier    uint8   1   minimum tier required to express this skill
  skill_id     uint16  2   index into SkillRegistry (universe*10000 + local)
  expression   float16 2   activation weight [0.0 → 1.0] as float16
  flags        uint8   1   bit0=active bit1=dominant bit2=mutated bit3=inherited
  reserved     1s      1   padding to 8 bytes

expression=0.0 means the gene is latent (present but silent).
expression≥0.9 means the gene drives behaviour (dominant expression).

The genome is the bridge between ChronosLedger (agent state) and
SkillRegistry (available skills). Evolution operates on the genome —
MutationEngine writes new expression values, BMEBridge cross-pollinates
genes between universes, GenesisThread triggers genome writes on
breakthrough events.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("ghost.genome_plane")

# ── Binary constants ───────────────────────────────────────────────────────────
GENE_STRUCT           = "<BBHeB1s"
GENE_SIZE             = struct.calcsize(GENE_STRUCT)   # must be 8
assert GENE_SIZE == 8, f"GENE_SIZE mismatch: {GENE_SIZE}"

GENOME_GENES_PER_SLOT = 8
GENOME_SLOT_SIZE      = GENE_SIZE * GENOME_GENES_PER_SLOT   # 64 bytes
assert GENOME_SLOT_SIZE == 64

_DEFAULT_PATH         = Path("data/genome_plane.bin")

# Gene flags
GENE_FLAG_ACTIVE    = 1 << 0   # gene is currently expressed
GENE_FLAG_DOMINANT  = 1 << 1   # overrides recessive genes for same skill slot
GENE_FLAG_MUTATED   = 1 << 2   # was mutated this generation
GENE_FLAG_INHERITED = 1 << 3   # came from crossover (not spontaneous)


def _empty_gene() -> bytes:
    return struct.pack(GENE_STRUCT, 0, 0, 0, 0.0, 0, b"\x00")


def _make_gene(
    universe_id: int,
    role_tier: int,
    skill_id: int,
    expression: float,
    flags: int = GENE_FLAG_ACTIVE,
) -> bytes:
    return struct.pack(
        GENE_STRUCT,
        universe_id & 0xFF,
        role_tier & 0xFF,
        skill_id & 0xFFFF,
        float(np.float16(expression)),
        flags & 0xFF,
        b"\x00",
    )


def _unpack_gene(raw: bytes) -> Dict:
    uid, role, sid, expr_raw, flags, _ = struct.unpack(GENE_STRUCT, raw)
    return {
        "universe_id":  uid,
        "role_tier":    role,
        "skill_id":     sid,
        "expression":   float(expr_raw),
        "flags":        flags,
        "is_active":    bool(flags & GENE_FLAG_ACTIVE),
        "is_dominant":  bool(flags & GENE_FLAG_DOMINANT),
        "is_mutated":   bool(flags & GENE_FLAG_MUTATED),
        "is_inherited": bool(flags & GENE_FLAG_INHERITED),
    }


class GenomePlane:
    """
    Parallel mmap store of agent skill genomes.

    Indexed by the same slot number as ChronosLedger. Each slot holds
    8 gene records — the agent's current skill DNA. Reads and writes are
    direct memory operations with no JSON, no DB overhead.

    Thread safety: single-process safe with GIL. For multi-process access
    use file-level locking around write_gene().

    The plane auto-extends when a slot outside the current file is written
    — no manual resizing needed.
    """

    def __init__(self, path: Optional[Path] = None, max_slots: int = 65536):
        self._path      = Path(path) if path else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_slots = max_slots
        self._map:   Optional[mmap.mmap] = None
        self._fd:    Optional[int]       = None
        self._slots  = 0

        if not self._path.exists() or self._path.stat().st_size == 0:
            self._init_file(initial_slots=4096)
        self._load()

    # ── Init / load ────────────────────────────────────────────────────────────

    def _init_file(self, initial_slots: int) -> None:
        size = initial_slots * GENOME_SLOT_SIZE
        with open(self._path, "wb") as f:
            f.write(b"\x00" * size)
        LOG.info(
            "GenomePlane: created %s (%d slots, %d bytes)",
            self._path, initial_slots, size,
        )

    def _load(self) -> None:
        if self._map:
            self._map.close()
        if self._fd is not None:
            os.close(self._fd)

        self._fd    = os.open(str(self._path), os.O_RDWR)
        file_size   = os.path.getsize(str(self._path))
        self._slots = file_size // GENOME_SLOT_SIZE
        self._map   = mmap.mmap(self._fd, file_size, access=mmap.ACCESS_WRITE)
        LOG.debug("GenomePlane: loaded %d slots", self._slots)

    def _ensure_slot(self, slot: int) -> None:
        if slot < self._slots:
            return
        needed     = slot + 1
        new_size   = needed * GENOME_SLOT_SIZE
        self._map.close()
        os.close(self._fd)
        with open(self._path, "ab") as f:
            current = os.path.getsize(str(self._path))
            f.write(b"\x00" * (new_size - current))
        self._load()

    def close(self) -> None:
        if self._map:
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

    # ── Low-level byte ops ────────────────────────────────────────────────────

    def _slot_offset(self, slot: int) -> int:
        return slot * GENOME_SLOT_SIZE

    def _gene_offset(self, slot: int, gene_idx: int) -> int:
        return self._slot_offset(slot) + gene_idx * GENE_SIZE

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_gene(self, slot: int, gene_idx: int) -> Optional[Dict]:
        """Read one gene. Returns None if slot doesn't exist."""
        if slot >= self._slots or gene_idx >= GENOME_GENES_PER_SLOT:
            return None
        off = self._gene_offset(slot, gene_idx)
        raw = self._map[off : off + GENE_SIZE]
        if all(b == 0 for b in raw):
            return None
        return _unpack_gene(raw)

    def read_genome(self, slot: int) -> List[Optional[Dict]]:
        """All 8 genes for a slot. Latent genes return None."""
        genes = []
        for i in range(GENOME_GENES_PER_SLOT):
            genes.append(self.read_gene(slot, i))
        return genes

    def active_genes(self, slot: int) -> List[Dict]:
        """Active (expressed) genes only."""
        return [
            g for g in self.read_genome(slot)
            if g is not None and g["is_active"] and g["expression"] > 0.0
        ]

    def expressed_skill_ids(self, slot: int) -> List[int]:
        """List of skill_ids the agent is currently expressing, sorted by expression strength."""
        return [
            g["skill_id"]
            for g in sorted(
                self.active_genes(slot),
                key=lambda x: x["expression"],
                reverse=True,
            )
        ]

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_gene(
        self,
        slot: int,
        gene_idx: int,
        universe_id: int,
        role_tier: int,
        skill_id: int,
        expression: float,
        flags: int = GENE_FLAG_ACTIVE,
    ) -> None:
        """Write one gene into an agent's genome slot."""
        self._ensure_slot(slot)
        off = self._gene_offset(slot, gene_idx)
        raw = _make_gene(universe_id, role_tier, skill_id, expression, flags)
        self._map[off : off + GENE_SIZE] = raw

    def clear_gene(self, slot: int, gene_idx: int) -> None:
        """Zero out one gene (makes it latent)."""
        self._ensure_slot(slot)
        off = self._gene_offset(slot, gene_idx)
        self._map[off : off + GENE_SIZE] = _empty_gene()

    def clear_genome(self, slot: int) -> None:
        """Zero all 8 genes for a slot (fresh offspring)."""
        self._ensure_slot(slot)
        off = self._slot_offset(slot)
        self._map[off : off + GENOME_SLOT_SIZE] = b"\x00" * GENOME_SLOT_SIZE

    def write_genome(self, slot: int, genes: List[Tuple]) -> None:
        """
        Write a list of gene tuples. Tuples: (universe_id, role_tier, skill_id, expression, flags).
        Pads remaining gene slots with zeros. Max GENOME_GENES_PER_SLOT genes.
        """
        self._ensure_slot(slot)
        for i, gene in enumerate(genes[:GENOME_GENES_PER_SLOT]):
            uid, role, sid, expr, flags = gene
            self.write_gene(slot, i, uid, role, sid, expr, flags)
        # Zero remaining
        for i in range(len(genes), GENOME_GENES_PER_SLOT):
            self.clear_gene(slot, i)

    # ── Mutation ──────────────────────────────────────────────────────────────

    def mutate_genome(
        self,
        slot: int,
        mutation_rate: float = 0.05,
        expression_drift: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """
        In-place Bernoulli mutation of gene expressions.
        Returns number of genes mutated.
        """
        if rng is None:
            rng = np.random.default_rng()
        mutated = 0
        for i in range(GENOME_GENES_PER_SLOT):
            gene = self.read_gene(slot, i)
            if gene is None:
                continue
            if rng.random() < mutation_rate:
                drift         = float(rng.normal(0.0, expression_drift))
                new_expr      = float(np.clip(gene["expression"] + drift, 0.0, 1.0))
                new_flags     = gene["flags"] | GENE_FLAG_MUTATED
                self.write_gene(
                    slot, i,
                    gene["universe_id"], gene["role_tier"],
                    gene["skill_id"], new_expr, new_flags,
                )
                mutated += 1
        return mutated

    # ── Crossover ─────────────────────────────────────────────────────────────

    def crossover(
        self,
        parent_a: int,
        parent_b: int,
        child_slot: int,
        crossover_point: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """
        Single-point crossover between two parent genomes → write to child_slot.
        Genes before crossover_point come from parent_a; rest from parent_b.
        crossover_point defaults to random [1, GENOME_GENES_PER_SLOT-1].
        """
        if rng is None:
            rng = np.random.default_rng()
        if crossover_point is None:
            crossover_point = int(rng.integers(1, GENOME_GENES_PER_SLOT))

        self._ensure_slot(child_slot)
        self.clear_genome(child_slot)

        for i in range(GENOME_GENES_PER_SLOT):
            src_slot = parent_a if i < crossover_point else parent_b
            gene = self.read_gene(src_slot, i)
            if gene is None:
                continue
            flags = (gene["flags"] | GENE_FLAG_INHERITED) & ~GENE_FLAG_MUTATED
            self.write_gene(
                child_slot, i,
                gene["universe_id"], gene["role_tier"],
                gene["skill_id"], gene["expression"], flags,
            )

    def inject_gene(
        self,
        slot: int,
        universe_id: int,
        role_tier: int,
        skill_id: int,
        expression: float = 0.5,
    ) -> int:
        """
        Add a new gene to the first empty/latent slot in an agent's genome.
        Returns the gene index used, or -1 if genome is full.
        """
        for i in range(GENOME_GENES_PER_SLOT):
            gene = self.read_gene(slot, i)
            if gene is None or gene["expression"] == 0.0:
                self.write_gene(
                    slot, i,
                    universe_id, role_tier, skill_id, expression,
                    GENE_FLAG_ACTIVE | GENE_FLAG_INHERITED,
                )
                return i
        return -1

    # ── Numpy vectorised reads ─────────────────────────────────────────────────

    def expression_vector(self, slot: int) -> np.ndarray:
        """Float16 array of 8 expression values for the slot. Fast path — no dict alloc."""
        if slot >= self._slots:
            return np.zeros(GENOME_GENES_PER_SLOT, dtype=np.float16)
        off    = self._slot_offset(slot)
        raw    = np.frombuffer(self._map[off : off + GENOME_SLOT_SIZE], dtype=np.uint8)
        # Expression values are at byte offsets 4,5 of each 8-byte gene
        expr_u16 = np.frombuffer(
            raw.reshape(GENOME_GENES_PER_SLOT, GENE_SIZE)[:, 4:6].tobytes(),
            dtype=np.uint16,
        )
        return expr_u16.view(np.float16)

    def genome_fitness_contribution(self, slot: int) -> float:
        """Weighted sum of expression values — proxy for genome 'activation level'."""
        vec = self.expression_vector(slot)
        return float(np.sum(vec[vec > 0.0]))

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self, sample_slots: int = 256) -> Dict:
        active_per_slot  = []
        for s in range(min(sample_slots, self._slots)):
            active_per_slot.append(len(self.active_genes(s)))
        arr = np.array(active_per_slot, dtype=float)
        return {
            "total_slots": self._slots,
            "file":        str(self._path),
            "sample_slots": sample_slots,
            "mean_active_genes": round(float(np.mean(arr)), 2) if arr.size else 0.0,
            "max_active_genes":  int(np.max(arr)) if arr.size else 0,
            "empty_genome_fraction": round(float(np.mean(arr == 0)), 3) if arr.size else 1.0,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_plane: Optional[GenomePlane] = None


def get_genome_plane() -> GenomePlane:
    global _plane
    if _plane is None:
        _plane = GenomePlane()
    return _plane
