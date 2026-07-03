"""
SkillRegistry — Binary Multiverse Engine (BME) skill DNA library.

skills.bin layout (fully pre-compiled, loaded once at startup):
  ┌──────────────────────────────────────────────────────┐
  │ FILE HEADER       32 bytes                           │
  │ UNIVERSE DIR      8 universes × 16 bytes = 128 bytes │
  │ SKILL RECORDS     N × 64 bytes                       │
  └──────────────────────────────────────────────────────┘

Skill record (64 bytes, cache-line aligned — 1 skill per cache line):
  skill_id        uint32   4   unique ID (universe * 10000 + local_id)
  name            char[20] 20  null-padded skill name
  input_sig       uint64   8   bitmask of accepted input IOTypes
  output_sig      uint64   8   bitmask of produced output IOTypes
  dominance       uint8    1   0=recessive 1=additive 2=dominant 3=co-dominant
  _pad            3s       3   alignment
  mutation_rate   float32  4   per-gene mutation probability [0,1]
  reward_weight   float32  4   contribution to fitness [0,1]
  universe_bits   uint64   8   universe-specific encoded state
  flags           uint8    1   bit0=elite_only bit1=experimental bit2=governance_locked
  role_tier       uint8    1   minimum role tier required to access (0=any)
  discovery_count uint16   2   times agents have discovered/used this skill

Zero-latency O(1) access: mmap loaded once, indexed by skill_id via hash table
in _skill_index. No JSON, no DB, no network. Pure memory dereference.

Universe IDs (encoded in scratchpad bits 3-5 of ChronosLedger):
  0 = Physics      (quantum, string theory, spacetime)
  1 = Chemistry    (reactions, catalysts, molecular bonds)
  2 = Biology      (mycelial, evolution, symbiosis)
  3 = Psychology   (attention, resonance, persuasion)
  4 = Fungal       (networks, spores, distributed growth)
  5 = Cosmic       (gravity, dark matter, horizon crossing)
  6 = Entropy      (chaos, stochastic, dissipation)
  7 = Hybrid       (cross-universe synthesists — Meta-Genomic Governors)

Role Tier Ladder (encoded in scratchpad bits 6-8):
  0 = Base            (any agent)
  1 = Specialist      (domain proficiency)
  2 = Elite           (cross-domain synthesis)
  3 = Apex Synthesist (multi-universe trait merger)
  4 = Quantum Architect / Molecular Alchemist / Species Creator (domain apex)
  5 = String Theorist / Cosmic Engineer / Resonance Engineer (frontier designer)
  6 = Meta-Genomic Governor (curates the registry itself)
  7 = Substrate Philosopher / Data God (proposes new universes)

2026 frontier roles above tier 7 are classified by skill-genome composition,
not scratchpad bits. See ROLE_TIER_NAMES for full taxonomy.
"""

from __future__ import annotations

import logging
import mmap
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("ghost.skill_registry")

# ── Binary constants ───────────────────────────────────────────────────────────
MAGIC          = 0x424D4531   # "BME1"
VERSION        = 1

HEADER_SIZE    = 32           # bytes
UNIV_DIR_ENTRY = 16           # bytes per universe in directory
NUM_UNIVERSES  = 8
UNIV_DIR_SIZE  = NUM_UNIVERSES * UNIV_DIR_ENTRY   # 128 bytes
DATA_OFFSET    = HEADER_SIZE + UNIV_DIR_SIZE        # 160 bytes before first skill

SKILL_STRUCT   = "<I20sQQB3sffQBBH"
SKILL_SIZE     = struct.calcsize(SKILL_STRUCT)      # must be 64
assert SKILL_SIZE == 64, f"SKILL_SIZE mismatch: {SKILL_SIZE}"

HEADER_STRUCT  = "<IBB4sII14s"  # magic(4) version(1) num_univ(1) pad(4) total_skills(4) next_id(4) reserved(14)
assert struct.calcsize(HEADER_STRUCT) == 32, struct.calcsize(HEADER_STRUCT)

UNIV_DIR_STRUCT = "<B8sHHBB1s"  # universe_id(1) name(8) skill_offset_idx(2) skill_count(2) flags(1) reserved(1) pad(1)
assert struct.calcsize(UNIV_DIR_STRUCT) == 16, struct.calcsize(UNIV_DIR_STRUCT)

_DEFAULT_PATH  = Path("data/skills.bin")
_MAX_SKILLS    = 4096          # maximum skills in registry

# ── Universe IDs ───────────────────────────────────────────────────────────────
UNIVERSE_PHYSICS    = 0
UNIVERSE_CHEMISTRY  = 1
UNIVERSE_BIOLOGY    = 2
UNIVERSE_PSYCHOLOGY = 3
UNIVERSE_FUNGAL     = 4
UNIVERSE_COSMIC     = 5
UNIVERSE_ENTROPY    = 6
UNIVERSE_HYBRID     = 7

UNIVERSE_NAMES = {
    0: "Physics",    1: "Chemistry",  2: "Biology",   3: "Psychology",
    4: "Fungal",     5: "Cosmic",     6: "Entropy",   7: "Hybrid",
}

# ── IOType bitmasks ────────────────────────────────────────────────────────────
IO_F32    = 1 << 0
IO_F64    = 1 << 1
IO_INT    = 1 << 2
IO_UINT   = 1 << 3
IO_BOOL   = 1 << 4
IO_BINARY = 1 << 5
IO_TEXT   = 1 << 6
IO_VECTOR = 1 << 7   # float16 desire-length vector (7 dims)

# ── Dominance types ────────────────────────────────────────────────────────────
DOM_RECESSIVE   = 0
DOM_ADDITIVE    = 1
DOM_DOMINANT    = 2
DOM_CO_DOMINANT = 3

# ── Skill flags ────────────────────────────────────────────────────────────────
SKILL_FLAG_ELITE_ONLY      = 1 << 0
SKILL_FLAG_EXPERIMENTAL    = 1 << 1
SKILL_FLAG_GOVERNANCE_LOCK = 1 << 2

# ── Role tier taxonomy (full 2026 ladder) ──────────────────────────────────────
ROLE_TIER_NAMES = {
    0: "Base",
    1: "Specialist",
    2: "Elite",
    3: "Apex Synthesist",
    4: "Quantum Architect / Molecular Alchemist / Species Creator",
    5: "String Theorist / Cosmic Engineer / Resonance Engineer",
    6: "Meta-Genomic Governor",
    7: "Substrate Philosopher",
    # 2026 frontier — classified by skill-genome composition, not scratchpad:
    8:  "Dimensional Weaver",
    9:  "Emergence Catalyst",
    10: "Causal Weaver",
    11: "Resonance Architect",
    12: "Attention Alchemist",
    13: "Frontier Economist",
    14: "Entropy Sovereign",
    15: "Data God",
}

# ── Seed skill library ─────────────────────────────────────────────────────────
# (universe_id, local_id, name, input_sig, output_sig, dominance,
#  mutation_rate, reward_weight, universe_bits, flags, role_tier)
SEED_SKILLS: List[Tuple] = [
    # ── Physics Universe ──
    (0, 0, "QuantumEntangle",  IO_VECTOR,         IO_VECTOR,         DOM_DOMINANT,    0.01, 0.90, 0x0000000000000001, SKILL_FLAG_ELITE_ONLY, 4),
    (0, 1, "StringVibration",  IO_F32,            IO_F64,            DOM_CO_DOMINANT, 0.02, 0.80, 0x0000000000000002, SKILL_FLAG_ELITE_ONLY, 5),
    (0, 2, "WaveCollapse",     IO_VECTOR,         IO_F32,            DOM_DOMINANT,    0.05, 0.75, 0x0000000000000003, 0,                     3),
    (0, 3, "TopologyShift",    IO_F32,            IO_VECTOR,         DOM_ADDITIVE,    0.08, 0.60, 0x0000000000000004, SKILL_FLAG_EXPERIMENTAL,2),
    (0, 4, "EnergyQuantize",   IO_F32,            IO_UINT,           DOM_ADDITIVE,    0.10, 0.55, 0x0000000000000005, 0,                     1),
    # ── Chemistry Universe ──
    (1, 0, "CatalystReact",    IO_F32 | IO_TEXT,  IO_F32,            DOM_ADDITIVE,    0.03, 0.70, 0x0000000000000100, 0,                     1),
    (1, 1, "MolecularBond",    IO_VECTOR,         IO_VECTOR,         DOM_DOMINANT,    0.04, 0.85, 0x0000000000000101, SKILL_FLAG_ELITE_ONLY, 4),
    (1, 2, "Polymerize",       IO_F32,            IO_BINARY,         DOM_CO_DOMINANT, 0.06, 0.65, 0x0000000000000102, SKILL_FLAG_EXPERIMENTAL,3),
    (1, 3, "DissolveBarrier",  IO_BOOL,           IO_BOOL,           DOM_RECESSIVE,   0.15, 0.45, 0x0000000000000103, SKILL_FLAG_ELITE_ONLY, 6),
    (1, 4, "EnzymaticCleave",  IO_VECTOR,         IO_VECTOR,         DOM_DOMINANT,    0.07, 0.72, 0x0000000000000104, 0,                     3),
    # ── Biology Universe ──
    (2, 0, "MycelialNetwork",  IO_BINARY,         IO_F32,            DOM_DOMINANT,    0.05, 0.85, 0x0000000000010000, 0,                     2),
    (2, 1, "SymbioticLink",    IO_VECTOR,         IO_F64,            DOM_ADDITIVE,    0.04, 0.78, 0x0000000000010001, 0,                     2),
    (2, 2, "AdaptiveGrowth",   IO_F32,            IO_UINT,           DOM_CO_DOMINANT, 0.06, 0.68, 0x0000000000010002, 0,                     1),
    (2, 3, "SporeRelease",     IO_BINARY,         IO_BINARY,         DOM_RECESSIVE,   0.20, 0.40, 0x0000000000010003, SKILL_FLAG_EXPERIMENTAL,3),
    (2, 4, "NecrosisSignal",   IO_BOOL,           IO_BOOL,           DOM_DOMINANT,    0.12, 0.55, 0x0000000000010004, SKILL_FLAG_ELITE_ONLY, 4),
    # ── Psychology Universe ──
    (3, 0, "AttentionTrigger", IO_TEXT,           IO_F32,            DOM_DOMINANT,    0.03, 0.88, 0x0000000001000000, 0,                     2),
    (3, 1, "EmotionalResonance",IO_VECTOR,        IO_VECTOR,         DOM_CO_DOMINANT, 0.04, 0.82, 0x0000000001000001, SKILL_FLAG_ELITE_ONLY, 4),
    (3, 2, "PersuasionMolecule",IO_TEXT,          IO_BOOL,           DOM_DOMINANT,    0.06, 0.77, 0x0000000001000002, SKILL_FLAG_ELITE_ONLY, 5),
    (3, 3, "CognitiveBias",    IO_VECTOR,         IO_VECTOR,         DOM_ADDITIVE,    0.10, 0.60, 0x0000000001000003, 0,                     1),
    (3, 4, "StatusSignaling",  IO_F32,            IO_VECTOR,         DOM_CO_DOMINANT, 0.08, 0.65, 0x0000000001000004, 0,                     2),
    # ── Fungal Universe ──
    (4, 0, "HyphalExtension",  IO_BINARY,         IO_BINARY,         DOM_ADDITIVE,    0.06, 0.75, 0x0000000100000000, 0,                     2),
    (4, 1, "ResourceCapture",  IO_F32,            IO_F32,            DOM_DOMINANT,    0.05, 0.80, 0x0000000100000001, 0,                     2),
    (4, 2, "NetworkResilience",IO_VECTOR,         IO_F32,            DOM_CO_DOMINANT, 0.04, 0.85, 0x0000000100000002, 0,                     3),
    (4, 3, "DigestionField",   IO_VECTOR,         IO_VECTOR,         DOM_RECESSIVE,   0.12, 0.50, 0x0000000100000003, SKILL_FLAG_EXPERIMENTAL,3),
    (4, 4, "FruitingBody",     IO_BINARY,         IO_BINARY,         DOM_DOMINANT,    0.03, 0.90, 0x0000000100000004, SKILL_FLAG_ELITE_ONLY, 5),
    # ── Cosmic Universe ──
    (5, 0, "GravityWell",      IO_VECTOR,         IO_VECTOR,         DOM_DOMINANT,    0.02, 0.88, 0x0001000000000000, SKILL_FLAG_ELITE_ONLY, 4),
    (5, 1, "TimeDilation",     IO_F32,            IO_F32,            DOM_RECESSIVE,   0.01, 0.95, 0x0001000000000001, SKILL_FLAG_ELITE_ONLY, 6),
    (5, 2, "HorizonCrossing",  IO_BINARY,         IO_BINARY,         DOM_DOMINANT,    0.03, 0.92, 0x0001000000000002, SKILL_FLAG_ELITE_ONLY, 7),
    (5, 3, "EntropicShield",   IO_F32,            IO_F32,            DOM_ADDITIVE,    0.05, 0.70, 0x0001000000000003, 0,                     3),
    (5, 4, "DarkMatterDense",  IO_BINARY,         IO_BOOL,           DOM_DOMINANT,    0.02, 0.93, 0x0001000000000004, SKILL_FLAG_GOVERNANCE_LOCK, 7),
    # ── Entropy Universe ──
    (6, 0, "ChaosInjection",   IO_F32,            IO_VECTOR,         DOM_RECESSIVE,   0.25, 0.40, 0x0100000000000000, 0,                     1),
    (6, 1, "StochResonance",   IO_F32,            IO_F32,            DOM_ADDITIVE,    0.15, 0.60, 0x0100000000000001, 0,                     2),
    (6, 2, "DecayResistance",  IO_BOOL,           IO_F32,            DOM_DOMINANT,    0.08, 0.75, 0x0100000000000002, 0,                     2),
    (6, 3, "BifurcationPoint", IO_VECTOR,         IO_BINARY,         DOM_CO_DOMINANT, 0.10, 0.65, 0x0100000000000003, SKILL_FLAG_EXPERIMENTAL,4),
    (6, 4, "DissipativeStruct",IO_BINARY,         IO_F32,            DOM_DOMINANT,    0.07, 0.80, 0x0100000000000004, SKILL_FLAG_ELITE_ONLY, 5),
    # ── Hybrid Universe (cross-universe synthesists) ──
    (7, 0, "QuantumCatalysis", IO_VECTOR,         IO_VECTOR,         DOM_DOMINANT,    0.02, 0.95, 0xFFFFFFFFFFFFFFFF, SKILL_FLAG_ELITE_ONLY | SKILL_FLAG_GOVERNANCE_LOCK, 7),
    (7, 1, "ChemoBioFusion",   IO_VECTOR | IO_F32,IO_VECTOR,         DOM_CO_DOMINANT, 0.03, 0.92, 0xFFFFFFFFFFFFFFFE, SKILL_FLAG_ELITE_ONLY, 6),
    (7, 2, "PsychoPhysResonance",IO_TEXT,         IO_VECTOR,         DOM_DOMINANT,    0.02, 0.97, 0xFFFFFFFFFFFFFFFD, SKILL_FLAG_ELITE_ONLY | SKILL_FLAG_GOVERNANCE_LOCK, 7),
    (7, 3, "FungalEntropyWeb", IO_BINARY,         IO_BINARY,         DOM_DOMINANT,    0.04, 0.91, 0xFFFFFFFFFFFFFFFC, SKILL_FLAG_ELITE_ONLY, 6),
    (7, 4, "UniverseFounder",  IO_BINARY,         IO_BINARY,         DOM_DOMINANT,    0.01, 0.99, 0xFFFFFFFFFFFFFFFB, SKILL_FLAG_ELITE_ONLY | SKILL_FLAG_GOVERNANCE_LOCK | SKILL_FLAG_EXPERIMENTAL, 7),
]


def _skill_id(universe_id: int, local_id: int) -> int:
    return universe_id * 10000 + local_id


class SkillRegistry:
    """
    Binary Multiverse Engine skill DNA library.

    Loaded once at startup. All lookups are O(1) memory dereferences via
    _skill_index dict built from the mmap'd skills.bin. Zero JSON, zero DB,
    zero network — pure in-memory binary access.

    Skills are the molecules of the BME. Agents carry skill genomes (see
    GenomePlane) — pointers into this registry. Evolution is recombination
    of those pointers under universe-specific mutation operators.
    """

    def __init__(self, path: Optional[Path] = None):
        self._path     = Path(path) if path else _DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._map:     Optional[mmap.mmap]       = None
        self._fd:      Optional[int]             = None
        self._index:   Dict[int, int]            = {}  # skill_id → byte offset in mmap
        self._total    = 0
        self._universe_offsets: Dict[int, Tuple[int,int]] = {}  # univ_id → (start_idx, count)

        if not self._path.exists() or self._path.stat().st_size < HEADER_SIZE:
            self._build_initial(SEED_SKILLS)
        self._load()

    # ── Build ──────────────────────────────────────────────────────────────────

    def _build_initial(self, seeds: List[Tuple]) -> None:
        """Compile seed skills into skills.bin."""
        grouped: Dict[int, List[Tuple]] = {}
        for row in seeds:
            uid = row[0]
            grouped.setdefault(uid, []).append(row)

        total_skills = len(seeds)
        file_size    = HEADER_SIZE + UNIV_DIR_SIZE + total_skills * SKILL_SIZE

        with open(self._path, "wb") as f:
            # ── Header ──
            f.write(struct.pack(HEADER_STRUCT,
                MAGIC, VERSION, NUM_UNIVERSES,
                b"\x00\x00\x00\x00",
                total_skills,
                0,
                b"\x00" * 14,
            ))

            # ── Universe directory ──
            skill_cursor = 0
            for uid in range(NUM_UNIVERSES):
                skills_here = grouped.get(uid, [])
                count       = len(skills_here)
                name        = UNIVERSE_NAMES.get(uid, "unknown").encode("ascii")[:8].ljust(8, b"\x00")
                f.write(struct.pack(UNIV_DIR_STRUCT,
                    uid, name,
                    skill_cursor & 0xFFFF,   # offset within skill array (in skill-count units)
                    count & 0xFFFF,
                    0, 0, b"\x00",
                ))
                skill_cursor += count

            # ── Skill records ──
            for uid in range(NUM_UNIVERSES):
                for row in grouped.get(uid, []):
                    (universe_id, local_id, name, input_sig, output_sig,
                     dominance, mutation_rate, reward_weight,
                     universe_bits, flags, role_tier) = row
                    sid       = _skill_id(universe_id, local_id)
                    name_enc  = name.encode("ascii")[:20].ljust(20, b"\x00")
                    f.write(struct.pack(SKILL_STRUCT,
                        sid, name_enc, input_sig, output_sig,
                        dominance, b"\x00\x00\x00",
                        mutation_rate, reward_weight,
                        universe_bits, flags, role_tier, 0,
                    ))

        LOG.info(
            "SkillRegistry: compiled %d skills into %s (%d bytes)",
            total_skills, self._path, file_size,
        )

    # ── Load ───────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        self._fd  = os.open(str(self._path), os.O_RDWR)
        file_size = os.path.getsize(str(self._path))
        if file_size < DATA_OFFSET or (file_size - DATA_OFFSET) % SKILL_SIZE != 0:
            LOG.warning(
                "SkillRegistry: rebuilding invalid skills.bin layout (%s, %d bytes)",
                self._path, file_size,
            )
            os.close(self._fd)
            self._fd = None
            self._build_initial(SEED_SKILLS)
            self._load()
            return
        self._map = mmap.mmap(self._fd, file_size, access=mmap.ACCESS_READ)

        # Validate header
        magic, version, num_univ, _, total_skills, next_id, _ = struct.unpack_from(
            HEADER_STRUCT, self._map, 0
        )
        if magic != MAGIC:
            self.close()
            self._build_initial(SEED_SKILLS)
            self._load()
            return
        self._total = total_skills

        # Read universe directory → compute universe ranges
        for i in range(num_univ):
            off  = HEADER_SIZE + i * UNIV_DIR_ENTRY
            uid, name, start_idx, count, *_ = struct.unpack_from(
                UNIV_DIR_STRUCT, self._map, off
            )
            self._universe_offsets[uid] = (start_idx, count)

        # Build O(1) index: skill_id → byte offset
        for i in range(total_skills):
            byte_off = DATA_OFFSET + i * SKILL_SIZE
            if byte_off + SKILL_SIZE > file_size:
                break
            sid = struct.unpack_from("<I", self._map, byte_off)[0]
            self._index[sid] = byte_off

        LOG.info(
            "SkillRegistry: loaded %d skills from %s (%d universes indexed)",
            len(self._index), self._path, num_univ,
        )

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

    # ── O(1) lookup ────────────────────────────────────────────────────────────

    def get_skill(self, skill_id: int) -> Optional[Dict]:
        """O(1) lookup by skill_id. Returns None if not found."""
        off = self._index.get(skill_id)
        if off is None:
            return None
        return self._unpack_skill(off)

    def get_skill_by_universe(self, universe_id: int, local_id: int) -> Optional[Dict]:
        return self.get_skill(_skill_id(universe_id, local_id))

    def _unpack_skill(self, byte_off: int) -> Dict:
        sid, name_b, inp_sig, out_sig, dom, _pad, mut, rwd, ubits, flags, role, disc = \
            struct.unpack_from(SKILL_STRUCT, self._map, byte_off)
        return {
            "skill_id":        sid,
            "universe_id":     sid // 10000,
            "local_id":        sid % 10000,
            "name":            name_b.rstrip(b"\x00").decode("ascii", errors="replace"),
            "input_signature": inp_sig,
            "output_signature":out_sig,
            "dominance":       dom,
            "mutation_rate":   float(mut),
            "reward_weight":   float(rwd),
            "universe_bits":   ubits,
            "flags":           flags,
            "role_tier":       role,
            "discovery_count": disc,
            "is_elite_only":   bool(flags & SKILL_FLAG_ELITE_ONLY),
            "is_experimental": bool(flags & SKILL_FLAG_EXPERIMENTAL),
            "governance_locked": bool(flags & SKILL_FLAG_GOVERNANCE_LOCK),
            "universe_name":   UNIVERSE_NAMES.get(sid // 10000, "unknown"),
        }

    # ── Universe-level queries ─────────────────────────────────────────────────

    def get_universe_skills(self, universe_id: int) -> List[Dict]:
        """All skills for a universe, in order."""
        start_idx, count = self._universe_offsets.get(universe_id, (0, 0))
        result = []
        for i in range(count):
            byte_off = DATA_OFFSET + (start_idx + i) * SKILL_SIZE
            if byte_off + SKILL_SIZE <= self._map.size():
                result.append(self._unpack_skill(byte_off))
        return result

    def get_accessible_skills(
        self, universe_id: int, agent_role_tier: int
    ) -> List[Dict]:
        """Skills in a universe accessible to an agent at the given role tier."""
        return [
            s for s in self.get_universe_skills(universe_id)
            if s["role_tier"] <= agent_role_tier and not s["governance_locked"]
        ]

    # ── Skill proposal (speciation event) ─────────────────────────────────────

    def propose_skill(
        self,
        universe_id: int,
        name: str,
        input_sig: int,
        output_sig: int,
        dominance: int,
        mutation_rate: float,
        reward_weight: float,
        universe_bits: int = 0,
        flags: int = SKILL_FLAG_EXPERIMENTAL,
        role_tier: int = 1,
    ) -> int:
        """
        Append a new skill to skills.bin (speciation event).
        Called by elite agents (tier ≥ 6) or Meta-Genomic Governors.
        Returns the new skill_id.

        NOTE: Re-opens file in write mode briefly; all lookups are unaffected
        because the mmap covers the full file and the new record is appended.
        """
        existing_count = len(self.get_universe_skills(universe_id))
        local_id = existing_count  # next sequential ID within universe
        sid      = _skill_id(universe_id, local_id)
        if sid in self._index:
            # collision — bump until clear
            while sid in self._index:
                local_id += 1
                sid = _skill_id(universe_id, local_id)

        name_enc = name.encode("ascii")[:20].ljust(20, b"\x00")
        record   = struct.pack(
            SKILL_STRUCT,
            sid, name_enc, input_sig, output_sig,
            dominance, b"\x00\x00\x00",
            float(mutation_rate), float(reward_weight),
            universe_bits, flags, role_tier, 0,
        )

        # Append to file and reload mmap
        self.close()
        with open(self._path, "ab") as f:
            f.write(record)
        self._load()

        LOG.info(
            "SkillRegistry: speciation — new skill '%s' (id=%d) in universe %s",
            name, sid, UNIVERSE_NAMES.get(universe_id, str(universe_id)),
        )
        return sid

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        by_universe = {}
        for uid in range(NUM_UNIVERSES):
            skills = self.get_universe_skills(uid)
            by_universe[UNIVERSE_NAMES.get(uid, str(uid))] = {
                "count":           len(skills),
                "elite_only":      sum(1 for s in skills if s["is_elite_only"]),
                "experimental":    sum(1 for s in skills if s["is_experimental"]),
                "governance_lock": sum(1 for s in skills if s["governance_locked"]),
                "mean_reward":     round(float(np.mean([s["reward_weight"] for s in skills])), 3) if skills else 0.0,
            }
        return {
            "total_skills":   self._total,
            "indexed_skills": len(self._index),
            "file":           str(self._path),
            "by_universe":    by_universe,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
    return _registry
