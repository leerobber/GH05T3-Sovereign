"""
Novelty Reward Engine — discovery × impact × rarity × resonance.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NoveltyScore:
    discovery: float = 0.0
    impact: float = 0.0
    rarity: float = 0.0
    resonance: float = 0.0
    weighted: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "discovery": self.discovery,
            "impact": self.impact,
            "rarity": self.rarity,
            "resonance": self.resonance,
            "weighted": self.weighted,
        }


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0
    ng_a = {a[i : i + n] for i in range(max(0, len(a) - n + 1))}
    ng_b = {b[i : i + n] for i in range(max(0, len(b) - n + 1))}
    if not ng_a or not ng_b:
        return 0.0
    return len(ng_a & ng_b) / len(ng_a | ng_b)


class NoveltyDetector:
    def __init__(self, max_history: int = 500):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history

    def add_to_history(self, output: Dict[str, Any]) -> None:
        self.history.append(output)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def score(self, output: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> float:
        if not self.history:
            return 1.0
        text = str(output.get("raw_output", output.get("content", output)))
        sims = [_ngram_jaccard(text, str(h.get("raw_output", h.get("content", h)))) for h in self.history]
        return max(0.0, min(1.0, 1.0 - (sum(sims) / len(sims))))


class ImpactEvaluator:
    def score(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return 0.5
        weights = {
            "click_through_rate": 0.3,
            "dwell_time": 0.2,
            "conversion_rate": 0.4,
            "engagement": 0.1,
            "revenue": 0.3,
            "task_success": 0.5,
        }
        total, wsum = 0.0, 0.0
        for k, w in weights.items():
            if k in metrics:
                total += float(metrics[k]) * w
                wsum += w
        return max(0.0, min(1.0, total / wsum if wsum else 0.5))


class RarityEstimator:
    def __init__(self, max_population: int = 500):
        self.population: List[Dict[str, Any]] = []
        self.max_population = max_population
        self.feature_counts: defaultdict = defaultdict(int)

    def add_to_population(self, output: Dict[str, Any]) -> None:
        if len(self.population) >= self.max_population:
            old = self.population.pop(0)
            self._count(old, -1)
        self.population.append(output)
        self._count(output, 1)

    def _count(self, output: Dict[str, Any], delta: int) -> None:
        text = str(output.get("raw_output", output.get("content", output))).lower()
        for w in text.split():
            self.feature_counts[w] += delta

    def score(self, output: Dict[str, Any]) -> float:
        text = str(output.get("raw_output", output.get("content", output))).lower()
        words = text.split()
        if not words:
            return 0.5
        freq = sum(self.feature_counts.get(w, 0) for w in words) / len(words)
        return max(0.0, min(1.0, 1.0 - freq / (freq + 1.0)))


class ResonanceAnalyzer:
    EMOTION_WEIGHTS = {
        "joy": 0.3, "trust": 0.25, "fear": -0.1, "surprise": 0.2,
        "curiosity": 0.2, "excitement": 0.25, "anger": -0.3, "disgust": -0.3,
    }

    def score(self, metrics: Dict[str, Any]) -> float:
        emotions = metrics.get("emotions", {})
        if not emotions:
            return 0.5
        total, wsum = 0.0, 0.0
        for em, w in self.EMOTION_WEIGHTS.items():
            if em in emotions:
                total += float(emotions[em]) * w
                wsum += abs(w)
        if not wsum:
            return 0.5
        return max(0.0, min(1.0, (total / wsum + 1.0) / 2.0))


@dataclass
class NoveltyRewardEngine:
    novelty_detector: NoveltyDetector = field(default_factory=NoveltyDetector)
    impact_evaluator: ImpactEvaluator = field(default_factory=ImpactEvaluator)
    rarity_estimator: RarityEstimator = field(default_factory=RarityEstimator)
    resonance_analyzer: ResonanceAnalyzer = field(default_factory=ResonanceAnalyzer)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "discovery": 0.3, "impact": 0.25, "rarity": 0.2, "resonance": 0.25,
    })

    def compute_reward(
        self,
        agent_id: str,
        output: Dict[str, Any],
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> NoveltyScore:
        discovery = self.novelty_detector.score(output, context)
        impact = self.impact_evaluator.score(metrics)
        rarity = self.rarity_estimator.score(output)
        resonance = self.resonance_analyzer.score(metrics)

        self.novelty_detector.add_to_history(output)
        self.rarity_estimator.add_to_population(output)

        base = discovery * rarity
        performance = impact * resonance
        weighted = 0.6 * base + 0.4 * performance

        return NoveltyScore(
            discovery=round(discovery, 4),
            impact=round(impact, 4),
            rarity=round(rarity, 4),
            resonance=round(resonance, 4),
            weighted=round(max(0.0, min(1.0, weighted)), 4),
        )