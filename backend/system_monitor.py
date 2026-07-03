"""Real system hardware monitoring for GH05T3.

Reads actual CPU, RAM, disk, and GPU stats. Injected into the system prompt
every chat so GH05T3 reports real numbers instead of hallucinated ones.
"""
from __future__ import annotations
import logging
import os
import subprocess
import time
from functools import lru_cache
from typing import Optional

import psutil

LOG = logging.getLogger("ghost.monitor")

_nvml_ready: bool | None = None  # None=untried, True=ok, False=unavailable


def _get_gpu_handle():
    global _nvml_ready
    if _nvml_ready is False:
        return None, None
    try:
        import warnings, pynvml
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if _nvml_ready is None:
                pynvml.nvmlInit()
                _nvml_ready = True
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return pynvml, h
    except Exception:
        _nvml_ready = False
        return None, None


def gpu_stats() -> dict:
    """Return real NVIDIA GPU stats. Returns empty dict if no GPU."""
    pynvml, h = _get_gpu_handle()
    if h is None:
        return {}
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            name = pynvml.nvmlDeviceGetName(h)
        return {
            "name":        name,
            "util_pct":    util.gpu,
            "mem_used_gb": round(mem.used  / 1024**3, 1),
            "mem_total_gb": round(mem.total / 1024**3, 1),
            "temp_c":      temp,
        }
    except Exception as e:
        LOG.debug("GPU stat error: %s", e)
        return {}


def cpu_stats() -> dict:
    """Return real CPU stats."""
    freq = psutil.cpu_freq()
    return {
        "util_pct": psutil.cpu_percent(interval=0.2),
        "cores":    psutil.cpu_count(logical=False),
        "threads":  psutil.cpu_count(logical=True),
        "freq_mhz": round(freq.current) if freq else None,
    }


def ram_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "used_gb":  round(vm.used  / 1024**3, 1),
        "total_gb": round(vm.total / 1024**3, 1),
        "util_pct": vm.percent,
    }


def disk_stats() -> dict:
    try:
        d = psutil.disk_usage("C:\\")
        return {
            "used_gb":  round(d.used  / 1024**3, 1),
            "total_gb": round(d.total / 1024**3, 1),
            "free_gb":  round(d.free  / 1024**3, 1),
        }
    except Exception:
        return {}


def snapshot() -> dict:
    """Full hardware snapshot — called once per chat to inject into system prompt."""
    return {
        "cpu": cpu_stats(),
        "ram": ram_stats(),
        "disk": disk_stats(),
        "gpu": gpu_stats(),
    }


def format_for_prompt(snap: dict | None = None) -> str:
    """Return a compact one-liner for the system prompt."""
    if snap is None:
        snap = snapshot()
    parts = []

    cpu = snap.get("cpu", {})
    ram = snap.get("ram", {})
    gpu = snap.get("gpu", {})
    disk = snap.get("disk", {})

    if cpu:
        parts.append(f"CPU {cpu.get('util_pct', '?')}%")
    if ram:
        parts.append(f"RAM {ram.get('used_gb','?')}/{ram.get('total_gb','?')}GB ({ram.get('util_pct','?')}%)")
    if gpu:
        parts.append(
            f"GPU({gpu.get('name','?')}) {gpu.get('util_pct','?')}% "
            f"{gpu.get('mem_used_gb','?')}/{gpu.get('mem_total_gb','?')}GB "
            f"{gpu.get('temp_c','?')}°C"
        )
    if disk:
        parts.append(f"Disk C: {disk.get('free_gb','?')}GB free")

    return "LIVE HARDWARE: " + " | ".join(parts) if parts else ""
