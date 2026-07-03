from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .detector import HardwareDetector


class HardwareAwareBinaryDispatcher:
    """Routes matmul/attention calls to the best available real device
    (GPU if present, else CPU), moving tensors there first.

    All kernel bodies below are plain torch ops — there's no actual
    hardware-specific intrinsic underneath (no real AMX/AVX512/NPU code
    exists to call into), so this dispatcher's value today is correct
    device placement, not a compute speedup. It's a real seam to drop
    actual optimized kernels into later, not a performance claim now.
    """

    def __init__(self, detector: HardwareDetector):
        self.detector = detector
        self.kernels: Dict[str, Dict[str, Callable]] = {
            "gpu_discrete": {"matmul": self._matmul, "attention": self._attention},
            "cpu_ai": {"matmul": self._matmul, "attention": self._attention},
            "cpu_regular": {"matmul": self._matmul, "attention": self._attention},
        }

    def dispatch(self, op: str, *args, workload: Optional[Dict] = None, **kwargs):
        device = self.detector.get_best_device(workload)
        if device is None:
            raise RuntimeError("No hardware available")

        device_type = device["type"]
        if device_type not in self.kernels or op not in self.kernels[device_type]:
            raise ValueError(f"No kernel for op={op!r} on device_type={device_type!r}")

        args = [self._to_device(a, device) for a in args]
        return self.kernels[device_type][op](*args, **kwargs)

    def _to_device(self, x, device: Dict):
        if not isinstance(x, torch.Tensor):
            return x
        if device["type"] == "gpu_discrete":
            return x.to(f"cuda:{device.get('index', 0)}")
        return x.to("cpu")

    def _matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    def _attention(self, q, k, v, mask=None):
        scale = 1.0 / (q.size(-1) ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, v)
