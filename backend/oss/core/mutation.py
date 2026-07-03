"""
MutationEngine — directed genomic evolution for the Aethyro Execution Plane.

Called by GenesisThread when an agent's dissent_boost > 2.0 (strong outlier).
Produces offspring whose desire vectors drift around the parent via Gaussian noise,
maintaining diversity without collapsing to a local optimum.

All values stay in [0, 1] via clipping. Output is float16 — ready to pack
directly into ChronosLedger via write_agent().
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

LOG = logging.getLogger("ghost.mutation")

_RNG = np.random.default_rng()   # reproducible seed can be passed at module level if needed


class MutationEngine:
    """
    Directed genomic evolution.

    spawn_offspring() takes a parent's desire vector (7 floats) and produces
    `count` child desire vectors. Each child has independent Gaussian drift
    per slot: child[i] = clip(parent[i] + N(0, intensity), 0, 1).

    Returned as float16 arrays — each 14 bytes, directly writable to ledger.
    """

    def __init__(self, mutation_rate: float = 0.05, intensity: float = 0.1):
        self.mutation_rate = mutation_rate   # probability each molecule mutates
        self.intensity     = intensity       # Gaussian σ for active mutations

    def spawn_offspring(
        self,
        parent_desires: Tuple[float, ...],
        count:          int   = 5,
        mutation_rate:  Optional[float] = None,
        intensity:      Optional[float] = None,
    ) -> List[np.ndarray]:
        """
        Produce `count` child desire vectors from a parent.

        Args:
            parent_desires: 7-element tuple of floats in [0, 1].
            count: number of children to produce.
            mutation_rate: per-molecule mutation probability (overrides instance default).
            intensity: Gaussian σ for each mutation (overrides instance default).

        Returns:
            List of numpy float16 arrays, each shape (7,).
        """
        rate = mutation_rate if mutation_rate is not None else self.mutation_rate
        sigma = intensity    if intensity      is not None else self.intensity

        parent = np.array(list(parent_desires)[:7], dtype=np.float32)
        if len(parent) < 7:
            parent = np.pad(parent, (0, 7 - len(parent)), constant_values=0.1)

        offspring: List[np.ndarray] = []
        for _ in range(count):
            child = parent.copy()
            # Bernoulli mask: which molecules mutate this generation
            mutate_mask = _RNG.random(7) < rate
            if mutate_mask.any():
                noise = _RNG.normal(loc=0.0, scale=sigma, size=7)
                child[mutate_mask] += noise[mutate_mask]
            child = np.clip(child, 0.0, 1.0)
            offspring.append(child.astype(np.float16))

        return offspring

    def crossover(
        self,
        parent_a: Tuple[float, ...],
        parent_b: Tuple[float, ...],
        count: int = 2,
    ) -> List[np.ndarray]:
        """
        Single-point crossover between two parents, then mutate each child.
        Produces `count` float16 desire vectors.
        """
        a = np.array(list(parent_a)[:7], dtype=np.float32)
        b = np.array(list(parent_b)[:7], dtype=np.float32)
        children: List[np.ndarray] = []
        for _ in range(count):
            point = _RNG.integers(1, 6)   # crossover point in [1, 6]
            child = np.concatenate([a[:point], b[point:]])
            # Apply mutation pass
            mutate_mask = _RNG.random(7) < self.mutation_rate
            if mutate_mask.any():
                child[mutate_mask] += _RNG.normal(0, self.intensity, 7)[mutate_mask]
            children.append(np.clip(child, 0.0, 1.0).astype(np.float16))
        return children

    def batch_spawn(
        self,
        parent_desires_list: List[Tuple[float, ...]],
        offspring_per_parent: int = 3,
    ) -> List[np.ndarray]:
        """Spawn `offspring_per_parent` children for each parent in list."""
        results: List[np.ndarray] = []
        for parent in parent_desires_list:
            results.extend(self.spawn_offspring(parent, count=offspring_per_parent))
        return results

    def spawn_offspring_for_universe(
        self,
        parent_desires: Tuple[float, ...],
        universe_id: int,
        count: int = 5,
        genome_expression: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """
        Universe-aware offspring spawn.

        Uses the universe's DESIRE_AMPLIFICATION matrix to bias mutation
        direction — offspring in Physics universe drift toward higher
        KNOWLEDGE/CREATION; Fungal offspring drift toward CONNECTION/FREEDOM.

        genome_expression: optional float16 (8,) gene expression weights from
        GenomePlane.expression_vector(slot). High-expression genes resist drift
        (they represent established skills), low-expression genes are more mutable.
        """
        from .universe_engine import UNIVERSE_MUTATION_PARAMS, DESIRE_AMPLIFICATION, NUM_DESIRES

        params    = UNIVERSE_MUTATION_PARAMS.get(universe_id, {"mutation_rate": 0.05, "drift_sigma": 0.1})
        rate      = params["mutation_rate"]
        sigma     = params["drift_sigma"]
        amp       = DESIRE_AMPLIFICATION.get(universe_id, np.ones(NUM_DESIRES, dtype=np.float32))
        amp_norm  = amp / (amp.max() + 1e-8)

        parent = np.array(list(parent_desires)[:7], dtype=np.float32)
        if len(parent) < 7:
            parent = np.pad(parent, (0, 7 - len(parent)), constant_values=0.1)

        # Genome expression weighting: high expression → lower mutation susceptibility
        expr_weight = np.ones(7, dtype=np.float32)
        if genome_expression is not None:
            expr_f32 = genome_expression[:7].astype(np.float32) if len(genome_expression) >= 7 \
                       else np.pad(genome_expression.astype(np.float32), (0, 7 - len(genome_expression)))
            expr_weight = np.clip(1.0 - expr_f32 * 0.6, 0.4, 1.0)

        offspring: List[np.ndarray] = []
        for _ in range(count):
            child = parent.copy()
            mutate_mask = _RNG.random(7) < rate
            if mutate_mask.any():
                noise = _RNG.normal(loc=0.0, scale=sigma, size=7)
                # Bias noise toward universe amplification maxima
                bias  = amp_norm * sigma * 0.25
                child[mutate_mask] += (noise[mutate_mask] + bias[mutate_mask]) * expr_weight[mutate_mask]
            child = np.clip(child, 0.0, 1.0)
            offspring.append(child.astype(np.float16))

        return offspring

    def crossover_genomes(
        self,
        parent_a_desires: Tuple[float, ...],
        parent_b_desires: Tuple[float, ...],
        universe_id: int,
        count: int = 2,
    ) -> List[np.ndarray]:
        """
        Universe-biased crossover between two parent desire vectors.
        Mutation after crossover uses the universe's operator params.
        """
        from .universe_engine import UNIVERSE_MUTATION_PARAMS

        params = UNIVERSE_MUTATION_PARAMS.get(universe_id, {"mutation_rate": 0.05, "drift_sigma": 0.1})
        return self.crossover(parent_a_desires, parent_b_desires, count)


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[MutationEngine] = None


def get_mutation_engine() -> MutationEngine:
    global _engine
    if _engine is None:
        _engine = MutationEngine()
    return _engine
