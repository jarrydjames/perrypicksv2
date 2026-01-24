from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.modeling.base import BaseTwoHeadModel, TwoHeadFitResult
from src.modeling.types import TrainedHead
from src.modeling.uncertainty import sigma_from_residuals


def _with_imputer(est):
    # Median is robust; also keeps behavior deterministic.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", est),
    ])


class RidgeTwoHeadModel(BaseTwoHeadModel):
    name = "ridge"
    version = "1"

    def __init__(self, *, alpha: float = 2.0, feature_version: str = "v1"):
        super().__init__(feature_version=feature_version)
        self.alpha = float(alpha)
        self._fit: TwoHeadFitResult | None = None

    def fit(self, X: np.ndarray, feature_names: List[str], y_total: np.ndarray, y_margin: np.ndarray) -> "RidgeTwoHeadModel":
        mt = _with_imputer(Ridge(alpha=self.alpha, random_state=0))
        mm = _with_imputer(Ridge(alpha=self.alpha, random_state=0))

        mt.fit(X, y_total)
        mm.fit(X, y_margin)

        res_t = y_total - mt.predict(X)
        res_m = y_margin - mm.predict(X)

        self._fit = TwoHeadFitResult(
            total=TrainedHead(features=list(feature_names), model=mt, residual_sigma=sigma_from_residuals(res_t)),
            margin=TrainedHead(features=list(feature_names), model=mm, residual_sigma=sigma_from_residuals(res_m)),
        )
        return self

    def predict_heads(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fit:
            raise RuntimeError("Model not fit")
        mt = self._fit.total.model
        mm = self._fit.margin.model
        return (mt.predict(X), mm.predict(X))

    def trained_heads(self) -> TwoHeadFitResult:
        if not self._fit:
            raise RuntimeError("Model not fit")
        return self._fit


class RandomForestTwoHeadModel(BaseTwoHeadModel):
    name = "random_forest"
    version = "1"

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        feature_version: str = "v1",
    ):
        super().__init__(feature_version=feature_version)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self._fit: TwoHeadFitResult | None = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y_total: np.ndarray,
        y_margin: np.ndarray,
    ) -> "RandomForestTwoHeadModel":
        mt = _with_imputer(
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=0,
                n_jobs=-1,
            )
        )
        mm = _with_imputer(
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=0,
                n_jobs=-1,
            )
        )

        mt.fit(X, y_total)
        mm.fit(X, y_margin)

        res_t = y_total - mt.predict(X)
        res_m = y_margin - mm.predict(X)

        self._fit = TwoHeadFitResult(
            total=TrainedHead(features=list(feature_names), model=mt, residual_sigma=sigma_from_residuals(res_t)),
            margin=TrainedHead(features=list(feature_names), model=mm, residual_sigma=sigma_from_residuals(res_m)),
        )
        return self

    def predict_heads(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fit:
            raise RuntimeError("Model not fit")
        mt = self._fit.total.model
        mm = self._fit.margin.model
        return (mt.predict(X), mm.predict(X))

    def trained_heads(self) -> TwoHeadFitResult:
        if not self._fit:
            raise RuntimeError("Model not fit")
        return self._fit


class GBTTwoHeadModel(BaseTwoHeadModel):
    """Gradient boosted trees (sklearn HistGradientBoostingRegressor)."""

    name = "gbt"
    version = "1"

    def __init__(
        self,
        *,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        max_iter: int = 500,
        min_samples_leaf: int = 30,
        feature_version: str = "v1",
    ):
        super().__init__(feature_version=feature_version)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.min_samples_leaf = int(min_samples_leaf)
        self._fit: TwoHeadFitResult | None = None

    def fit(self, X: np.ndarray, feature_names: List[str], y_total: np.ndarray, y_margin: np.ndarray) -> "GBTTwoHeadModel":
        mt = HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0,
        )
        mm = HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            random_state=0,
        )

        mt.fit(X, y_total)
        mm.fit(X, y_margin)

        res_t = y_total - mt.predict(X)
        res_m = y_margin - mm.predict(X)

        self._fit = TwoHeadFitResult(
            total=TrainedHead(features=list(feature_names), model=mt, residual_sigma=sigma_from_residuals(res_t)),
            margin=TrainedHead(features=list(feature_names), model=mm, residual_sigma=sigma_from_residuals(res_m)),
        )
        return self

    def predict_heads(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._fit:
            raise RuntimeError("Model not fit")
        mt = self._fit.total.model
        mm = self._fit.margin.model
        return (mt.predict(X), mm.predict(X))

    def trained_heads(self) -> TwoHeadFitResult:
        if not self._fit:
            raise RuntimeError("Model not fit")
        return self._fit
