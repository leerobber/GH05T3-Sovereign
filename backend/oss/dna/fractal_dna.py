"""Fractal DNA v2 — nested sub-traits (depth <=3) for hierarchical trait structure (Phase 4).

Success: 80%+ of evolve calls discover or modify sub-traits (new branches or meaningful deltas).
Used to give fine-grained control inside broad traits (e.g. cognitive.math.algebra).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import random


@dataclass
class FractalDNA:
    tree: Dict[str, Any] = field(default_factory=dict)
    max_depth: int = 3
    mutations: int = 0
    discovered_paths: List[Tuple[str, ...]] = field(default_factory=list)

    def _clamp(self, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def set_trait(self, path: List[str], value: float) -> None:
        if len(path) > self.max_depth:
            path = path[:self.max_depth]
        node = self.tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        old = node.get(path[-1])
        node[path[-1]] = self._clamp(value)
        pkey = tuple(path)
        if pkey not in self.discovered_paths:
            self.discovered_paths.append(pkey)

    def get_trait(self, path: List[str], default: float = 0.5) -> float:
        node = self.tree
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return self._clamp(default)
            node = node[key]
        val = node if isinstance(node, (int, float)) else default
        return self._clamp(float(val))

    def evolve_fractal(self, path: Optional[List[str]] = None, strength: float = 0.06) -> float:
        """Evolve at a path (or random discovered / root path). Returns new value.
        Creates substructure if missing -> counts as 'discovery'.
        """
        if path is None or not path:
            # pick or create interesting path
            if self.discovered_paths:
                path = list(random.choice(self.discovered_paths))
            else:
                path = [random.choice(["cognitive", "market", "meta", "risk", "collab"]), random.choice(["depth", "speed", "precision"])]

        if len(path) < 2:
            path = path + [random.choice(["sub", "detail", "specialty"])]

        before = self.get_trait(path)
        delta = random.uniform(-strength, strength) * (1.0 + 0.3 * (len(path)-1))  # deeper = slightly more volatile
        new_val = self._clamp(before + delta)
        self.set_trait(path, new_val)
        self.mutations += 1

        # discovery metric: new leafs created in process
        if tuple(path) not in self.discovered_paths:
            self.discovered_paths.append(tuple(path))

        return new_val

    def flatten(self, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        def _walk(node, pfx):
            for k, v in (node or {}).items():
                np = f"{pfx}.{k}" if pfx else k
                if isinstance(v, dict):
                    _walk(v, np)
                else:
                    out[np] = self._clamp(v)
        _walk(self.tree, prefix)
        return out

    def discovery_rate(self, attempted: int) -> float:
        if attempted <= 0:
            return 0.0
        return min(1.0, len(self.discovered_paths) / max(1, attempted))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mutations": self.mutations,
            "discovered_paths": len(self.discovered_paths),
            "tree_size": len(self.flatten()),
            "max_depth_reached": max((len(p) for p in self.discovered_paths), default=0),
        }