import os
from typing import Dict, List, Optional

import torch


class HardwareDetector:
    """Detects real available compute devices (CPU flags, CUDA GPUs) for
    HardwareAwareBinaryDispatcher to route operations to. No NPU support —
    nothing in this environment exposes one, and none of the kernel paths
    below have real NPU-specific code beneath them anyway."""

    def __init__(self):
        self.devices: List[Dict] = []
        self._detect_cpu()
        self._detect_gpu()

    def _cpu_flags(self) -> set:
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("flags"):
                        return set(line.split(":", 1)[1].split())
        except Exception:
            pass
        return set()

    def _detect_cpu(self):
        flags = self._cpu_flags()
        has_amx = any(f.startswith("amx") for f in flags)
        has_avx512 = any(f.startswith("avx512") for f in flags)
        cores = os.cpu_count() or 4

        cpu_type = "cpu_ai" if (has_amx or has_avx512) else "cpu_regular"

        self.devices.append({
            "type": cpu_type,
            "name": "CPU",
            "cores": cores,
            "vram": 0,
            "training_support": False,
            "ai_extensions": has_amx or has_avx512,
            "index": 0,
        })

    def _detect_gpu(self):
        if not torch.cuda.is_available():
            return

        for idx in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(idx)
            vram = int(torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3))
            self.devices.append({
                "type": "gpu_discrete",
                "name": name,
                "vram": vram,
                "training_support": True,
                "ai_extensions": True,
                "index": idx,
            })

    def get_devices(self) -> List[Dict]:
        return self.devices

    def get_best_device(self, workload: Optional[Dict] = None) -> Optional[Dict]:
        if not self.devices:
            return None
        gpus = [d for d in self.devices if d["type"] == "gpu_discrete"]
        return gpus[0] if gpus else self.devices[0]
