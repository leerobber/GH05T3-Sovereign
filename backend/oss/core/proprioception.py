"""
ProprioceptionEngine — read-only hardware sensing for the Aethyro Execution Plane.

Collects system state (GPU, CPU, RAM) and exposes it as three data-only "Hands"
that the rest of the system can read without side effects.

Hands (report-only primitives — no hardware control):
  vram_shunt       → reports VRAM utilisation so the swarm knows when to shed load
  genomic_spike    → flags lineages that should receive higher mutation rate
  lex_gen_seal     → signals PatentOffice to seal a breakthrough (via seal.py)

Hardware access strategy:
  1. pynvml (NVIDIA NVML) — preferred for GPU info on RTX 5050 (SM 120, Blackwell)
  2. psutil  — CPU/RAM/disk, always available
  3. Graceful stub fallback if pynvml is absent or the GPU isn't supported
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

LOG = logging.getLogger("ghost.proprioception")

# ── GPU backend (optional) ────────────────────────────────────────────────────

try:
    import pynvml                                       # type: ignore
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import psutil as _psutil                            # type: ignore
    _PSUTIL_AVAILABLE = True
except Exception:
    _psutil          = None                             # type: ignore
    _PSUTIL_AVAILABLE = False


# ── Data primitives ("Hands") ─────────────────────────────────────────────────

@dataclass
class VRAMShunt:
    """Current VRAM snapshot. All values in MiB."""
    total_mib:  float = 0.0
    used_mib:   float = 0.0
    free_mib:   float = 0.0
    util_pct:   float = 0.0    # 0-100
    gpu_temp_c: float = 0.0
    gpu_util_pct: float = 0.0  # SM occupancy 0-100
    available:  bool  = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mib":    round(self.total_mib, 1),
            "used_mib":     round(self.used_mib, 1),
            "free_mib":     round(self.free_mib, 1),
            "util_pct":     round(self.util_pct, 1),
            "gpu_temp_c":   round(self.gpu_temp_c, 1),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "available":    self.available,
        }


@dataclass
class GenomicSpike:
    """
    Flags lineages that warrant higher mutation intensity this cycle.
    Set when: VRAM > 85%, CPU > 80%, or fitness flatlined for N ticks.
    """
    spike_active:   bool  = False
    reason:         str   = ""
    mutation_scale: float = 1.0   # multiply MutationEngine.intensity by this
    flagged_agents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spike_active":   self.spike_active,
            "reason":         self.reason,
            "mutation_scale": self.mutation_scale,
            "flagged_agents": self.flagged_agents,
        }


@dataclass
class LexGenSealSignal:
    """
    Data-only signal: 'PatentOffice, you should call seal.seal_breakthrough().'
    The engine never calls seal directly — it only sets this flag.
    GenesisThread reads it each tick and acts.
    """
    should_seal:   bool = False
    agent_id:      str  = ""
    parent_id:     str  = ""
    trigger_reason: str = ""
    data_payload:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_seal":    self.should_seal,
            "agent_id":       self.agent_id,
            "parent_id":      self.parent_id,
            "trigger_reason": self.trigger_reason,
        }


# ── Engine ────────────────────────────────────────────────────────────────────

class ProprioceptionEngine:
    """
    Polls hardware once per call (lazy — no background thread).
    All outputs are immutable snapshots; no hardware state is modified.
    """

    def __init__(self, gpu_index: int = 0):
        self._gpu_index = gpu_index
        self._last_sample: Optional[Dict[str, Any]] = None

    # ── Primary sensing ───────────────────────────────────────────────────────

    def sense(self) -> Dict[str, Any]:
        """Full hardware snapshot: VRAM + CPU + RAM."""
        vram  = self.read_vram()
        cpu   = self.read_cpu()
        spike = self.compute_genomic_spike(vram, cpu)
        sample = {
            "vram":          vram.to_dict(),
            "cpu":           cpu,
            "genomic_spike": spike.to_dict(),
            "latency_ms":    0,   # filled below
        }
        self._last_sample = sample
        return sample

    def read_vram(self) -> VRAMShunt:
        """GPU VRAM snapshot. Gracefully stubs if pynvml unavailable."""
        if not _NVML_AVAILABLE:
            return VRAMShunt(available=False)
        try:
            handle    = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            mem_info  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util  = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temp      = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            total_mib = mem_info.total / 1024 / 1024
            used_mib  = mem_info.used  / 1024 / 1024
            free_mib  = mem_info.free  / 1024 / 1024
            return VRAMShunt(
                total_mib    = total_mib,
                used_mib     = used_mib,
                free_mib     = free_mib,
                util_pct     = round(used_mib / max(total_mib, 1) * 100, 1),
                gpu_temp_c   = float(temp),
                gpu_util_pct = float(gpu_util.gpu),
                available    = True,
            )
        except Exception as exc:
            LOG.debug("proprioception: GPU read failed: %s", exc)
            return VRAMShunt(available=False)

    def read_cpu(self) -> Dict[str, Any]:
        """CPU + RAM snapshot via psutil."""
        if not _PSUTIL_AVAILABLE:
            return {"cpu_pct": 0.0, "ram_used_pct": 0.0, "ram_avail_mib": 0.0, "available": False}
        try:
            cpu_pct   = _psutil.cpu_percent(interval=None)
            vm        = _psutil.virtual_memory()
            return {
                "cpu_pct":       round(cpu_pct, 1),
                "ram_used_pct":  round(vm.percent, 1),
                "ram_avail_mib": round(vm.available / 1024 / 1024, 1),
                "available":     True,
            }
        except Exception as exc:
            LOG.debug("proprioception: CPU read failed: %s", exc)
            return {"cpu_pct": 0.0, "ram_used_pct": 0.0, "ram_avail_mib": 0.0, "available": False}

    # ── Derived Hands ─────────────────────────────────────────────────────────

    def compute_genomic_spike(
        self,
        vram: Optional[VRAMShunt] = None,
        cpu:  Optional[Dict[str, Any]] = None,
    ) -> GenomicSpike:
        """
        Determine if hardware pressure warrants elevated mutation.
        High VRAM → agents need to shed weights faster.
        High CPU  → evolutionary pressure: prune slow lineages faster.
        """
        if vram is None:
            vram = self.read_vram()
        if cpu is None:
            cpu = self.read_cpu()

        reasons: List[str] = []
        scale = 1.0

        if vram.available and vram.util_pct > 85.0:
            reasons.append(f"VRAM={vram.util_pct:.0f}%")
            scale = max(scale, 1.5)
        if cpu.get("available") and cpu.get("cpu_pct", 0) > 80.0:
            reasons.append(f"CPU={cpu['cpu_pct']:.0f}%")
            scale = max(scale, 1.3)

        if reasons:
            return GenomicSpike(
                spike_active   = True,
                reason         = ", ".join(reasons),
                mutation_scale = round(scale, 2),
            )
        return GenomicSpike(spike_active=False, mutation_scale=1.0)

    def build_seal_signal(
        self,
        agent_id:  str,
        parent_id: str,
        payload:   Dict[str, Any],
        reason:    str = "breakthrough",
    ) -> LexGenSealSignal:
        """
        Produce a seal request signal. The engine does NOT call seal.py itself —
        GenesisThread reads this and calls seal.seal_breakthrough().
        """
        return LexGenSealSignal(
            should_seal    = True,
            agent_id       = agent_id,
            parent_id      = parent_id,
            trigger_reason = reason,
            data_payload   = payload,
        )

    def last_sample(self) -> Optional[Dict[str, Any]]:
        return self._last_sample


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[ProprioceptionEngine] = None


def get_proprioception_engine() -> ProprioceptionEngine:
    global _engine
    if _engine is None:
        _engine = ProprioceptionEngine()
    return _engine
