"""
VolatilityWorld v1 — primary environmental pressure surface (Phase 2).

Generates regime-switching volatility series, accepts model submissions,
and evaluates proposals with weighted metrics.
"""
from __future__ import annotations

import math
import random
import statistics
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from ..log_config import get_logger

log = get_logger(__name__)


@dataclass
class VolatilitySeries:
    series_id: str
    timestamps: List[float]
    values: List[float]
    regimes: List[int]
    regime_labels: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def preview(self, n: int = 20) -> List[float]:
        return self.values[:n]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VolatilityModel:
    agent_id: str
    description: str
    code: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    model_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = f"vm_{uuid.uuid4().hex[:10]}"


@dataclass
class VolatilityChallenge:
    challenge_id: str
    series_id: str
    series_preview: List[float]
    regime_count: int
    difficulty: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VolatilityWorld:
    """Regime-switching synthetic volatility environment."""

    REGIME_SIGMAS = (0.1, 0.3, 0.6)
    REGIME_LABELS = ("low", "medium", "high")

    def __init__(self, length: int = 1000, seed: Optional[int] = None) -> None:
        self.length = length
        self._rng = random.Random(seed)
        self._series: Dict[str, VolatilitySeries] = {}
        self._models: Dict[str, VolatilityModel] = {}
        self._evaluations: Dict[str, Dict[str, Any]] = {}

    def generate_series(self, length: Optional[int] = None) -> Dict[str, Any]:
        """Generate regime-switching series; returns dict for backward compatibility."""
        series = self._build_series(length or self.length)
        self._series[series.series_id] = series
        log.info("generated series %s len=%d regimes=%s", series.series_id, len(series.values), series.regime_labels)
        return {
            "series_id": series.series_id,
            "series": series.values,
            "regimes": list(self.REGIME_SIGMAS),
            "regime_labels": series.regime_labels,
            "timestamps": series.timestamps,
        }

    def _build_series(self, length: int) -> VolatilitySeries:
        regimes_idx: List[int] = []
        values: List[float] = []
        current = self._rng.randint(0, len(self.REGIME_SIGMAS) - 1)

        for _ in range(length):
            if self._rng.random() < 0.05:
                current = self._rng.randint(0, len(self.REGIME_SIGMAS) - 1)
            regimes_idx.append(current)
            sigma = self.REGIME_SIGMAS[current]
            values.append(abs(self._rng.gauss(0, sigma)))

        series_id = f"vs_{uuid.uuid4().hex[:10]}"
        return VolatilitySeries(
            series_id=series_id,
            timestamps=[float(i) for i in range(length)],
            values=values,
            regimes=regimes_idx,
            regime_labels=list(self.REGIME_LABELS),
            metadata={"length": length, "switch_prob": 0.05},
        )

    def generate_challenge(self, series_id: Optional[str] = None) -> VolatilityChallenge:
        if series_id and series_id in self._series:
            series = self._series[series_id]
        else:
            series = self._build_series(self.length)
            self._series[series.series_id] = series

        unique_regimes = len(set(series.regimes))
        difficulty = min(1.0, unique_regimes / len(self.REGIME_SIGMAS))
        challenge = VolatilityChallenge(
            challenge_id=f"vc_{uuid.uuid4().hex[:8]}",
            series_id=series.series_id,
            series_preview=series.preview(32),
            regime_count=unique_regimes,
            difficulty=round(difficulty, 3),
            metadata={"mean_vol": round(statistics.mean(series.values), 4)},
        )
        log.debug("challenge %s series=%s difficulty=%.2f", challenge.challenge_id, series.series_id, difficulty)
        return challenge

    def submit_model(self, model: VolatilityModel) -> str:
        self._models[model.model_id] = model
        log.info("submitted model %s agent=%s", model.model_id, model.agent_id)
        return model.model_id

    def evaluate_model(
        self,
        model_or_text: Any,
        data: Optional[Dict[str, Any]] = None,
        *,
        model_id: Optional[str] = None,
        series_id: Optional[str] = None,
    ) -> float:
        """
        Evaluate by model_id + series_id (v1) or legacy (model_text, data dict).
        Returns weighted_score in [0, 1].
        """
        if model_id and series_id:
            result = self._evaluate_by_ids(model_id, series_id)
            return result["weighted_score"]

        # Legacy heuristic path
        model_text = str(model_or_text)
        payload = data or {}
        score = self._heuristic_text_score(model_text)
        if payload.get("series"):
            stat_bonus = self._statistical_fit_bonus(model_text, payload["series"])
            score = min(1.0, score * 0.6 + stat_bonus * 0.4)
        return round(score, 4)

    def _evaluate_by_ids(self, model_id: str, series_id: str) -> Dict[str, Any]:
        model = self._models.get(model_id)
        series = self._series.get(series_id)
        if not model:
            raise KeyError(f"model {model_id!r} not found")
        if not series:
            raise KeyError(f"series {series_id!r} not found")

        text = f"{model.description}\n{model.code}"
        keyword = self._heuristic_text_score(text)
        stat_fit = self._statistical_fit_bonus(text, series.values)
        regime_aware = self._regime_awareness_score(text, series)
        parsimony = self._parsimony_score(model.parameters)

        weighted = (
            keyword * 0.25
            + stat_fit * 0.30
            + regime_aware * 0.30
            + parsimony * 0.15
        )
        weighted = round(min(1.0, max(0.0, weighted)), 4)

        result = {
            "model_id": model_id,
            "series_id": series_id,
            "metrics": {
                "keyword_score": round(keyword, 4),
                "statistical_fit": round(stat_fit, 4),
                "regime_awareness": round(regime_aware, 4),
                "parsimony": round(parsimony, 4),
            },
            "weighted_score": weighted,
            "feedback": self._generate_feedback(weighted, model, series),
            "evaluated_at": time.time(),
        }
        self._evaluations[f"{model_id}:{series_id}"] = result
        log.info("evaluated %s on %s score=%.3f", model_id, series_id, weighted)
        return result

    @staticmethod
    def _heuristic_text_score(model_text: str) -> float:
        t = model_text.lower()
        score = 0.0
        if "regime" in t:
            score += 0.3
        if "volatility" in t or "variance" in t:
            score += 0.3
        if "switch" in t or "transition" in t or "markov" in t:
            score += 0.2
        if any(k in t for k in ("distribution", "stochastic", "process", "garch")):
            score += 0.2
        return min(score, 1.0)

    @staticmethod
    def _statistical_fit_bonus(model_text: str, series: List[float]) -> float:
        if not series:
            return 0.0
        mean_v = statistics.mean(series)
        std_v = statistics.pstdev(series) if len(series) > 1 else 0.0
        t = model_text.lower()
        bonus = 0.3
        if "mean" in t or "average" in t:
            bonus += 0.2
        if "std" in t or "standard deviation" in t:
            bonus += 0.2
        if std_v > 0 and ("heteroskedastic" in t or "cluster" in t):
            bonus += 0.2
        # Reward mentioning scale near actual stats
        if f"{mean_v:.2f}"[:3] in t or f"{std_v:.2f}"[:3] in t:
            bonus += 0.1
        return min(1.0, bonus)

    def _regime_awareness_score(self, model_text: str, series: VolatilitySeries) -> float:
        t = model_text.lower()
        n_regimes = len(set(series.regimes))
        score = 0.2
        if n_regimes >= 2 and "regime" in t:
            score += 0.4
        if "switch" in t or "transition" in t:
            score += 0.2
        if any(label in t for label in series.regime_labels):
            score += 0.2
        return min(1.0, score)

    @staticmethod
    def _parsimony_score(parameters: Dict[str, Any]) -> float:
        if not parameters:
            return 0.5
        n = len(parameters)
        if n <= 3:
            return 1.0
        if n <= 6:
            return 0.7
        return 0.4

    def _generate_feedback(
        self, score: float, model: VolatilityModel, series: VolatilitySeries
    ) -> str:
        if score >= 0.85:
            tier = "canonical_candidate"
            advice = "Promote to canonical memory; reproduce this lineage."
        elif score >= 0.65:
            tier = "viable"
            advice = "Refine regime-switch detection; add statistical validation."
        elif score >= 0.40:
            tier = "developing"
            advice = "Explicitly model regime transitions and volatility clustering."
        else:
            tier = "weak"
            advice = "Ground model in observed mean/std and named regime states."

        return (
            f"[{tier}] score={score:.3f} agent={model.agent_id} "
            f"series={series.series_id} regimes={len(set(series.regimes))}. {advice}"
        )

    def get_evaluation(self, model_id: str, series_id: str) -> Optional[Dict[str, Any]]:
        return self._evaluations.get(f"{model_id}:{series_id}")