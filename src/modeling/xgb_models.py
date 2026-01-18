from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.modeling.base import BaseTwoHeadModel, TwoHeadFitResult
from src.modeling.types import TrainedHead
from src.modeling.uncertainty import sigma_from_residuals


class XGBoostTwoHeadModel(BaseTwoHeadModel):
    """Backtest-only XGBoost two-head regressor.

    Lazy-imports xgboost so Streamlit runtime doesn’t need the dependency.
    """

    name = "xgboost"
    version = "1"

    def __init__(
        self,
        *,
        n_estimators: int = 1200,
        learning_rate: float = 0.03,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        min_child_weight: float = 5.0,
        feature_version: str = "v1",
        n_jobs: int = -1,
    ):
        super().__init__(feature_version=feature_version)
        self.params = {
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "subsample": float(subsample),
            "colsample_bytree": float(colsample_bytree),
            "reg_lambda": float(reg_lambda),
            "min_child_weight": float(min_child_weight),
            "random_state": 0,
            "n_jobs": int(n_jobs),
            "objective": "reg:squarederror",
        }
        self._fit: TwoHeadFitResult | None = None

    def fit(self, X: np.ndarray, feature_names: List[str], y_total: np.ndarray, y_margin: np.ndarray) -> "XGBoostTwoHeadModel":
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as e:  # pragma: no cover
            msg = (
                "Failed to import xgboost. Common macOS cause: missing OpenMP runtime (libomp). "
                "Try: `brew install libomp`. Also ensure dev deps are installed: "
                "`pip install -r requirements-dev.txt`."
            )
            raise RuntimeError(msg) from e

        mt = XGBRegressor(**self.params)
        mm = XGBRegressor(**self.params)

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
