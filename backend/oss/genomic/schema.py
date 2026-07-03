"""
Omni-Sentient Genomic Schema — alleles, molecules, loci, genomes, mutation maps.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple


class DominanceType(Enum):
    DOMINANT = auto()
    RECESSIVE = auto()
    CO_DOMINANT = auto()
    ADDITIVE = auto()
    EPISTATIC = auto()
    POLYGENIC = auto()


class MutationStrategy(Enum):
    GAUSSIAN = auto()
    JUMP = auto()
    DECAY = auto()
    REINFORCE = auto()
    CONTEXTUAL = auto()
    QUANTUM = auto()
    ADAPTIVE = auto()


class LocusType(Enum):
    COGNITIVE = auto()
    PSYCHOLOGY = auto()
    MARKET = auto()
    LOYALTY = auto()
    CREATIVITY = auto()
    MEMORY = auto()
    COMMUNICATION = auto()
    META = auto()
    CONTEXT = auto()  # context-awareness as a genetic trait (Phase 3)
    DESIRE  = auto()  # intrinsic desire drive — DDRS (Phase 4)
    LEX     = auto()  # legal-precision / patent-office trait (Phase 5)


@dataclass
class Allele:
    name: str
    value: float
    dominance: DominanceType = DominanceType.ADDITIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "dominance": self.dominance.name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Allele:
        return cls(
            name=data["name"],
            value=float(data["value"]),
            dominance=DominanceType[data["dominance"]],
            metadata=data.get("metadata", {}),
        )


@dataclass
class MutationRule:
    molecule_name: str
    rate: float
    strategy: str  # gaussian | jump | decay | reinforce | contextual
    step: float = 0.05
    context_key: Optional[str] = None


@dataclass
class Molecule:
    id: str
    name: str
    locus: LocusType
    alleles: Dict[str, Allele] = field(default_factory=dict)
    current_allele: str = "default"
    mutation_rate: float = 0.01
    mutation_strategy: MutationStrategy = MutationStrategy.GAUSSIAN
    min_value: float = 0.0
    max_value: float = 1.0
    interaction_map: List[str] = field(default_factory=list)
    epistasis_rules: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.alleles:
            self.alleles["default"] = Allele("default", 0.5, DominanceType.ADDITIVE)
            self.current_allele = "default"

    def get_value(self) -> float:
        return self.alleles[self.current_allele].value

    def set_value(self, value: float) -> None:
        self.alleles[self.current_allele].value = max(
            self.min_value, min(self.max_value, round(value, 4))
        )

    def set_allele(self, allele_name: str) -> bool:
        if allele_name in self.alleles:
            self.current_allele = allele_name
            return True
        return False

    def mutate(self, context: Optional[Dict[str, Any]] = None) -> bool:
        ctx = context or {}
        effective_rate = min(0.95, self.mutation_rate + ctx.get("mutation_rate_boost", 0.0))
        if random.random() > effective_rate:
            return False

        allele = self.alleles[self.current_allele]
        new_value = allele.value
        ctx = context or {}

        if self.mutation_strategy == MutationStrategy.GAUSSIAN:
            new_value += random.gauss(0, 0.1)
        elif self.mutation_strategy == MutationStrategy.JUMP:
            new_value += random.uniform(-0.2, 0.2)
        elif self.mutation_strategy == MutationStrategy.DECAY:
            new_value *= 0.95
        elif self.mutation_strategy == MutationStrategy.REINFORCE:
            if ctx.get("performance", 0.0) > 0.8:
                new_value += random.uniform(0.05, 0.15)
            else:
                new_value += random.uniform(-0.1, 0.05)
        elif self.mutation_strategy == MutationStrategy.CONTEXTUAL:
            sigma = 0.2 if ctx.get("novelty_reward", 0.0) > 0.7 else 0.05
            new_value += random.gauss(0, sigma)
        elif self.mutation_strategy == MutationStrategy.QUANTUM:
            if random.random() < 0.3:
                new_value = random.uniform(self.min_value, self.max_value)
            else:
                return False
        elif self.mutation_strategy == MutationStrategy.ADAPTIVE:
            hist = ctx.get("fitness_history", [])
            if hist:
                recent = sum(hist[-5:]) / min(5, len(hist))
                self.mutation_rate = min(0.1, self.mutation_rate * 1.1) if recent > 0.8 else max(0.001, self.mutation_rate * 0.9)
            new_value += random.gauss(0, self.mutation_rate)

        self._apply_epistasis(ctx)
        allele.value = max(self.min_value, min(self.max_value, round(new_value, 4)))
        return True

    def _apply_epistasis(self, ctx: Dict[str, Any]) -> None:
        """Apply interaction modifiers from epistasis_rules when partner values supplied."""
        partners = ctx.get("molecule_values", {})
        for partner_id, modifier in self.epistasis_rules.items():
            if partner_id in partners:
                self.alleles[self.current_allele].value += partners[partner_id] * modifier * 0.05

    def add_allele(self, allele: Allele) -> bool:
        if allele.name not in self.alleles:
            self.alleles[allele.name] = allele
            return True
        return False

    def crossover(self, other: Molecule) -> Tuple[Molecule, Molecule]:
        def _child(suffix: str) -> Molecule:
            return Molecule(
                id=f"{self.id}_{suffix}_{uuid.uuid4().hex[:6]}",
                name=self.name,
                locus=self.locus,
                mutation_rate=(self.mutation_rate + other.mutation_rate) / 2,
                mutation_strategy=random.choice([self.mutation_strategy, other.mutation_strategy]),
                min_value=self.min_value,
                max_value=self.max_value,
                interaction_map=list(set(self.interaction_map + other.interaction_map)),
                epistasis_rules={**self.epistasis_rules, **other.epistasis_rules},
            )

        child1, child2 = _child("c1"), _child("c2")
        for allele_name in set(self.alleles) | set(other.alleles):
            if random.random() < 0.5:
                child1.add_allele(deepcopy(self.alleles.get(allele_name, Allele(allele_name, 0.5))))
                child2.add_allele(deepcopy(other.alleles.get(allele_name, Allele(allele_name, 0.5))))
            else:
                child1.add_allele(deepcopy(other.alleles.get(allele_name, Allele(allele_name, 0.5))))
                child2.add_allele(deepcopy(self.alleles.get(allele_name, Allele(allele_name, 0.5))))
        child1.set_allele(_select_dominant_allele(child1.alleles))
        child2.set_allele(_select_dominant_allele(child2.alleles))
        return child1, child2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "locus": self.locus.name,
            "current_allele": self.current_allele,
            "mutation_rate": self.mutation_rate,
            "mutation_strategy": self.mutation_strategy.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "alleles": {n: a.to_dict() for n, a in self.alleles.items()},
            "interaction_map": self.interaction_map,
            "epistasis_rules": self.epistasis_rules,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Molecule:
        mol = cls(
            id=data["id"],
            name=data["name"],
            locus=LocusType[data["locus"]],
            current_allele=data["current_allele"],
            mutation_rate=float(data["mutation_rate"]),
            mutation_strategy=MutationStrategy[data["mutation_strategy"]],
            min_value=float(data["min_value"]),
            max_value=float(data["max_value"]),
            interaction_map=data.get("interaction_map", []),
            epistasis_rules=data.get("epistasis_rules", {}),
        )
        for name, ad in data.get("alleles", {}).items():
            mol.add_allele(Allele.from_dict({**ad, "name": name}))
        return mol


def _select_dominant_allele(alleles: Dict[str, Allele]) -> str:
    dominant = [n for n, a in alleles.items() if a.dominance == DominanceType.DOMINANT]
    if dominant:
        return random.choice(dominant)
    co_dom = [n for n, a in alleles.items() if a.dominance == DominanceType.CO_DOMINANT]
    if co_dom and len(co_dom) >= 2:
        v = sum(alleles[n].value for n in co_dom[:2]) / 2
        alleles[co_dom[0]].value = v
        return co_dom[0]
    return random.choice(list(alleles.keys()))


@dataclass
class Locus:
    name: str
    type: LocusType
    molecules: Dict[str, Molecule] = field(default_factory=dict)
    weight: float = 1.0

    def get_molecule(self, molecule_id: str) -> Optional[Molecule]:
        return self.molecules.get(molecule_id)

    def add_molecule(self, molecule: Molecule) -> bool:
        if molecule.id not in self.molecules:
            self.molecules[molecule.id] = molecule
            return True
        return False

    def mutate(self, context: Optional[Dict[str, Any]] = None) -> int:
        changed = 0
        ctx = dict(context or {})
        ctx["molecule_values"] = {mid: m.get_value() for mid, m in self.molecules.items()}
        for mol in self.molecules.values():
            if mol.mutate(ctx):
                changed += 1
        return changed

    def crossover(self, other: Locus) -> Tuple[Locus, Locus]:
        c1 = Locus(f"{self.name}_c1", self.type, weight=(self.weight + other.weight) / 2)
        c2 = Locus(f"{self.name}_c2", self.type, weight=(self.weight + other.weight) / 2)
        for mid in set(self.molecules) | set(other.molecules):
            if mid in self.molecules and mid in other.molecules:
                m1, m2 = self.molecules[mid].crossover(other.molecules[mid])
                c1.add_molecule(m1)
                c2.add_molecule(m2)
            elif mid in self.molecules:
                c1.add_molecule(deepcopy(self.molecules[mid]))
                c2.add_molecule(deepcopy(self.molecules[mid]))
            else:
                c1.add_molecule(deepcopy(other.molecules[mid]))
                c2.add_molecule(deepcopy(other.molecules[mid]))
        return c1, c2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.name,
            "weight": self.weight,
            "molecules": {mid: m.to_dict() for mid, m in self.molecules.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Locus:
        loc = cls(data["name"], LocusType[data["type"]], weight=float(data.get("weight", 1.0)))
        for mid, md in data.get("molecules", {}).items():
            loc.add_molecule(Molecule.from_dict(md))
        return loc


@dataclass
class Genome:
    lineage_id: str
    role: str = "generalist"
    loci: Dict[str, Locus] = field(default_factory=dict)
    fitness: float = 0.0
    mutation_map: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_molecule(self, locus_name: str, molecule_id: str) -> Optional[Molecule]:
        loc = self.loci.get(locus_name)
        return loc.get_molecule(molecule_id) if loc else None

    def get_value(self, locus_name: str, molecule_id: str) -> float:
        mol = self.get_molecule(locus_name, molecule_id)
        return mol.get_value() if mol else 0.0

    def set_value(self, locus_name: str, molecule_id: str, value: float) -> bool:
        mol = self.get_molecule(locus_name, molecule_id)
        if mol:
            mol.set_value(value)
            return True
        return False

    def _desire_mutation_boosts(self) -> Dict[str, float]:
        """Return per-locus mutation rate boosts driven by the agent's dominant desire."""
        desire_locus = self.loci.get("desire")
        if not desire_locus:
            return {}
        best_mid, best_val = None, 0.0
        for mid, mol in desire_locus.molecules.items():
            v = mol.get_value()
            if v > best_val:
                best_val, best_mid = v, mid
        if best_mid is None or best_val < 0.6:
            return {}
        _MAP: Dict[str, Dict[str, float]] = {
            "M_DESIRE_KNOWLEDGE":  {"cognitive": 0.03, "context": 0.02},
            "M_DESIRE_SKILL":      {"cognitive": 0.02, "market":  0.02},
            "M_DESIRE_STATUS":     {"loyalty":   0.03, "psychology": 0.01},
            "M_DESIRE_EXPERIENCE": {"context":   0.03, "meta":    0.02},
            "M_DESIRE_CREATION":   {"cognitive": 0.03, "context": 0.01},
            "M_DESIRE_CONNECTION": {"loyalty":   0.02, "communication": 0.03},
            "M_DESIRE_FREEDOM":    {"context":   0.02, "cognitive": 0.01},
        }
        return _MAP.get(best_mid, {})

    def mutate(self, context: Optional[Dict[str, Any]] = None) -> int:
        ctx = dict(context or {})
        if self.mutation_map:
            ctx.setdefault("novelty_reward", self.mutation_map.get("psychology", {}).get("novelty_boost", 0.0))
        desire_boosts = self._desire_mutation_boosts()
        total = 0
        for locus_name, locus in self.loci.items():
            boost = desire_boosts.get(locus_name, 0.0)
            locus_ctx = {**ctx, "mutation_rate_boost": ctx.get("mutation_rate_boost", 0.0) + boost} if boost else ctx
            total += locus.mutate(locus_ctx)
        return total

    def crossover(self, other: Genome) -> Tuple[Genome, Genome]:
        c1 = Genome(
            lineage_id=f"{self.lineage_id}_c1_{uuid.uuid4().hex[:6]}",
            role=random.choice([self.role, other.role]),
            mutation_map={**self.mutation_map, **other.mutation_map},
            metadata={"parents": [self.lineage_id, other.lineage_id], "created_at": _utc_now()},
        )
        c2 = Genome(
            lineage_id=f"{self.lineage_id}_c2_{uuid.uuid4().hex[:6]}",
            role=random.choice([self.role, other.role]),
            mutation_map={**self.mutation_map, **other.mutation_map},
            metadata={"parents": [self.lineage_id, other.lineage_id], "created_at": _utc_now()},
        )
        for name in set(self.loci) | set(other.loci):
            if name in self.loci and name in other.loci:
                l1, l2 = self.loci[name].crossover(other.loci[name])
                c1.loci[name] = l1
                c2.loci[name] = l2
            elif name in self.loci:
                c1.loci[name] = deepcopy(self.loci[name])
                c2.loci[name] = deepcopy(self.loci[name])
            else:
                c1.loci[name] = deepcopy(other.loci[name])
                c2.loci[name] = deepcopy(other.loci[name])
        return c1, c2

    def calculate_fitness(self, metrics: Dict[str, float]) -> float:
        total, weight_sum = 0.0, 0.0
        for locus in self.loci.values():
            if not locus.molecules:
                continue
            locus_score = sum(m.get_value() for m in locus.molecules.values()) / len(locus.molecules)
            total += locus_score * locus.weight
            weight_sum += locus.weight
        fitness = total / weight_sum if weight_sum else 0.0
        for key, mult in (
            ("task_success", (0.7, 0.3)),
            ("novelty", (0.8, 0.2)),
            ("impact", (0.85, 0.15)),
            ("engagement", (0.9, 0.1)),
        ):
            if key in metrics:
                fitness *= mult[0] + mult[1] * metrics[key]
        # context_efficiency boosts fitness when agent uses context well
        if "context_efficiency" in metrics:
            fitness *= 0.9 + 0.1 * metrics["context_efficiency"]
        # desire_alignment boosts fitness when a task matches the agent's intrinsic drives
        if "desire_alignment" in metrics:
            fitness *= 0.88 + 0.12 * metrics["desire_alignment"]
        self.fitness = max(0.0, min(1.0, round(fitness, 4)))
        return self.fitness

    def apply_interactions(self) -> None:
        """Boost molecules per interaction_map (psychology gene network)."""
        values: Dict[str, float] = {}
        for loc in self.loci.values():
            for mid, mol in loc.molecules.items():
                values[mid] = mol.get_value()
        for loc in self.loci.values():
            for mol in loc.molecules.values():
                for partner in mol.interaction_map:
                    if partner in values:
                        boost = (values[partner] - 0.5) * 0.02
                        mol.set_value(mol.get_value() + boost)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "role": self.role,
            "fitness": self.fitness,
            "loci": {n: l.to_dict() for n, l in self.loci.items()},
            "mutation_map": self.mutation_map,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Genome:
        g = cls(
            lineage_id=data["lineage_id"],
            role=data.get("role", "generalist"),
            fitness=float(data.get("fitness", 0.0)),
            mutation_map=data.get("mutation_map", {}),
            metadata=data.get("metadata", {}),
        )
        for name, ld in data.get("loci", {}).items():
            g.loci[name] = Locus.from_dict(ld)
        return g


def create_molecule(
    molecule_id: str,
    name: str,
    locus_type: LocusType,
    alleles: Optional[Dict[str, Dict[str, Any]]] = None,
    mutation_rate: float = 0.01,
    mutation_strategy: MutationStrategy = MutationStrategy.GAUSSIAN,
    interaction_map: Optional[List[str]] = None,
) -> Molecule:
    mol = Molecule(
        id=molecule_id,
        name=name,
        locus=locus_type,
        mutation_rate=mutation_rate,
        mutation_strategy=mutation_strategy,
        interaction_map=interaction_map or [],
    )
    if alleles:
        for aname, spec in alleles.items():
            dom = DominanceType[spec.get("dominance", "ADDITIVE")]
            if isinstance(spec.get("dominance"), str) and spec["dominance"] in DominanceType.__members__:
                dom = DominanceType[spec["dominance"]]
            mol.add_allele(Allele(aname, float(spec["value"]), dom))
    mol.set_allele("default" if "default" in mol.alleles else next(iter(mol.alleles)))
    return mol


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()