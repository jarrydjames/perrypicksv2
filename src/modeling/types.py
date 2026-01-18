from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class HeadPrediction:
    mu: float
    sigma: float
    pi80: Interval


@dataclass(frozen=True)
class ModelOutput:
    game_id: str
    timestamp_utc: str

    model_name: str
    model_version: str
    feature_version: str
    calibration_id: str

    mu: Dict[str, float]  # team_a/team_b/total/margin
    sigma: Dict[str, float]  # total/margin
    pi80: Dict[str, Dict[str, float]]  # {metric: {low, high}}

    probabilities: Dict[str, Any]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class FitArtifacts:
    feature_names: List[str]
    feature_version: str


@dataclass(frozen=True)
class TrainedHead:
    features: List[str]
    model: Any
    residual_sigma: float
