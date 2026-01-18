from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from src.modeling.types import TrainedHead


@dataclass
class TwoHeadFitResult:
    total: TrainedHead
    margin: TrainedHead


class BaseTwoHeadModel(ABC):
    """Predicts TOTAL and MARGIN with separate heads.

    This mirrors your framework:
      - Train separate models for total and margin
      - Convert to team scores downstream

    Implementations should:
    - store `feature_version`
    - train two heads
    - estimate residual sigmas per head
    """

    name: str
    version: str
    feature_version: str

    def __init__(self, *, feature_version: str = "v1"):
        self.feature_version = feature_version

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y_total: np.ndarray,
        y_margin: np.ndarray,
    ) -> "BaseTwoHeadModel":
        raise NotImplementedError

    @abstractmethod
    def predict_heads(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (mu_total, mu_margin)."""
        raise NotImplementedError

    @abstractmethod
    def trained_heads(self) -> TwoHeadFitResult:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        return {"model": self.name, "version": self.version, "feature_version": self.feature_version}
