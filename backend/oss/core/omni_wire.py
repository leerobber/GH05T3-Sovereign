"""
OmniWire — binary inter-agent communication protocol.

Every inter-agent message is a fixed 64-byte packet:
  header  (uint64) : protocol version / flags
  src_id  (uint64) : hash of sending agent_id
  task_id (uint64) : hash of task_id
  payload (40 bytes): raw binary payload (padded with null bytes)

BinaryBus: in-process ring buffer for passing OmniWire packets between
agents without JSON serialization or HTTP overhead.

Latency target: 700ns–2μs per message (vs. 7,000ns+ for JSON/REST).
"""

from __future__ import annotations

import struct
import threading
from collections import deque
from typing import Any, Dict, List, Optional

# ── Wire format ───────────────────────────────────────────────────────────────
OMNI_WIRE_FORMAT = "QQQ40s"
WIRE_SIZE        = struct.calcsize(OMNI_WIRE_FORMAT)   # 64 bytes
_PROTO_VERSION   = 1


def _hash64(s: str) -> int:
    """Stable 64-bit hash of a string, always non-negative."""
    return hash(s) & 0xFFFFFFFFFFFFFFFF


def pack_packet(src_agent_id: str, task_id: str, payload: bytes) -> bytes:
    """Encode an inter-agent message into a 64-byte OmniWire packet."""
    padded = (payload[:40]).ljust(40, b"\x00")
    return struct.pack(
        OMNI_WIRE_FORMAT,
        _PROTO_VERSION,
        _hash64(src_agent_id),
        _hash64(task_id),
        padded,
    )


def unpack_packet(raw: bytes) -> Dict[str, Any]:
    """Decode a 64-byte OmniWire packet. src/task returned as hashes."""
    if len(raw) < WIRE_SIZE:
        raise ValueError(f"Packet too short: {len(raw)} < {WIRE_SIZE}")
    header, src_hash, task_hash, payload = struct.unpack(OMNI_WIRE_FORMAT, raw[:WIRE_SIZE])
    return {
        "proto_version": header,
        "src_hash":      src_hash,
        "task_hash":     task_hash,
        "payload":       payload.rstrip(b"\x00"),
    }


# ── BinaryBus ─────────────────────────────────────────────────────────────────

class BinaryBus:
    """
    In-process ring-buffer message bus for OmniWire packets.

    Agents call send() to post a packet; receivers call recv() or drain()
    to consume. Thread-safe via a reentrant lock.

    This replaces HTTP/REST for in-process agent-to-agent signaling.
    External API endpoints (FastAPI on 8099) remain for cross-process
    and client communication — they are Control Plane.
    """

    def __init__(self, capacity: int = 1000):
        self._capacity = capacity
        self._bus: deque = deque(maxlen=capacity)
        self._lock = threading.RLock()
        self._stats = {"sent": 0, "dropped": 0, "consumed": 0}

    def send(self, src_agent_id: str, task_id: str, payload: bytes) -> bool:
        """
        Post a packet onto the bus. Returns True on success.
        If bus is at capacity, oldest packet is evicted (ring buffer).
        """
        packet = pack_packet(src_agent_id, task_id, payload)
        with self._lock:
            self._bus.append(packet)
            self._stats["sent"] += 1
        return True

    def send_dict(self, src_agent_id: str, task_id: str, data: Dict[str, Any]) -> bool:
        """Convenience: serialize a small dict into the 40-byte payload."""
        import json
        raw = json.dumps(data, separators=(",", ":")).encode()[:40]
        return self.send(src_agent_id, task_id, raw)

    def recv(self) -> Optional[Dict[str, Any]]:
        """Pop and decode one packet. Returns None if bus is empty."""
        with self._lock:
            if not self._bus:
                return None
            raw = self._bus.popleft()
            self._stats["consumed"] += 1
        return unpack_packet(raw)

    def drain(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Drain up to `limit` packets from the bus."""
        results = []
        for _ in range(limit):
            pkt = self.recv()
            if pkt is None:
                break
            results.append(pkt)
        return results

    def size(self) -> int:
        with self._lock:
            return len(self._bus)

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)


# ── Singleton ─────────────────────────────────────────────────────────────────

_bus: Optional[BinaryBus] = None


def get_binary_bus() -> BinaryBus:
    global _bus
    if _bus is None:
        _bus = BinaryBus()
    return _bus
