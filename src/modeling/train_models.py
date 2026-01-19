from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.data.training_loader import DEFAULT_TRAINING_PARQUET, TrainingDataSpec, load_training_df

from src.modeling.feature_columns import feature_columns
from src.modeling.sklearn_models import GBTTwoHeadModel, RandomForestTwoHeadModel, RidgeTwoHeadModel


TARGET_TOTAL = "h2_total"
TARGET_MARGIN = "h2_margin"


def train_from_parquet(
    parquet_path: Path,
    out_dir: Path,
    *,
    include_xgb: bool = False,
    include_cat: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_df(TrainingDataSpec(path=parquet_path))
    feats = feature_columns(df)

    X = df[feats].to_numpy(dtype=float)
    y_total = df[TARGET_TOTAL].to_numpy(dtype=float)
    y_margin = df[TARGET_MARGIN].to_numpy(dtype=float)

    models = [
        RidgeTwoHeadModel(alpha=2.0, feature_version="v1"),
        RandomForestTwoHeadModel(feature_version="v1"),
        GBTTwoHeadModel(feature_version="v1"),
    ]

    # Backtest-only optional models (do NOT add to Streamlit runtime deps)
    if include_xgb:
        from src.modeling.xgb_models import XGBoostTwoHeadModel

        models.append(XGBoostTwoHeadModel(feature_version="v1"))

    if include_cat:
        from src.modeling.cat_models import CatBoostTwoHeadModel

        models.append(CatBoostTwoHeadModel(feature_version="v1"))

    for m in models:
        m.fit(X, feats, y_total, y_margin)
        heads = m.trained_heads()

        payload = {
            "model_name": m.name,
            "model_version": m.version,
            "feature_version": m.feature_version,
            "features": feats,
            "total": {"model": heads.total.model, "residual_sigma": heads.total.residual_sigma},
            "margin": {"model": heads.margin.model, "residual_sigma": heads.margin.residual_sigma},
        }

        out_path = out_dir / f"{m.name}_twohead.joblib"
        joblib.dump(payload, out_path)
        print(f"Saved model: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_TRAINING_PARQUET,
        help=f"Path to training parquet (default: {DEFAULT_TRAINING_PARQUET})",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("models_v2"))
    ap.add_argument("--include-xgb", action="store_true", help="Backtest-only: train XGBoost two-head")
    ap.add_argument("--include-cat", action="store_true", help="Backtest-only: train CatBoost two-head")
    args = ap.parse_args()

    train_from_parquet(args.data, args.out_dir, include_xgb=args.include_xgb, include_cat=args.include_cat)


if __name__ == "__main__":
    main()
