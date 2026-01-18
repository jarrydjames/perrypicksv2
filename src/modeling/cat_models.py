from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.modeling.base import BaseTwoHeadModel, TwoHeadFitResult
from src.modeling.types import TrainedHead
from src.modeling.uncertainty import sigma_from_residuals


class CatBoostTwoHeadModel(BaseTwoHeadModel):
    """Backtest-only CatBoost two-head regressor.

    Lazy-imports catboost so Streamlit runtime doesn’t need the dependency.
    """

    name = "catboost"
    version = "1"

    def __init__(
        self,
        *,
        iterations: int = 2500,
        learning_rate: float = 0.03,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        subsample: float = 0.8,
        feature_version: str = "v1",
        random_seed: int = 0,
    ):
        super().__init__(feature_version=feature_version)
        self.params = {
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "depth": int(depth),
            "l2_leaf_reg": float(l2_leaf_reg),
            "subsample": float(subsample),
            "loss_function": "RMSE",
            "random_seed": int(random_seed),
            "verbose": False,
        }
        self._fit: TwoHeadFitResult | None = None

    def fit(self, X: np.ndarray, feature_names: List[str], y_total: np.ndarray, y_margin: np.ndarray) -> "CatBoostTwoHeadModel":
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "catboost is not installed. Install dev deps: pip install -r requirements-dev.txt"
            ) from e

        mt = CatBoostRegressor(**self.params)
        mm = CatBoostRegressor(**self.params)

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
        return (self._fit.total.model.predict(X), self._fit.margin.model.predict(X))

    def trained_heads(self) -> TwoHeadFitResult:
        if not self._fit:
            raise RuntimeError("Model not fit")
        return self._fit
