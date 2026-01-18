from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.modeling.backtest_utils import FoldSpec, attach_game_time_utc, brier, coverage, iter_walkforward_indices, mae, p_home_win, rmse
from src.modeling.base import BaseTwoHeadModel
from src.modeling.feature_columns import feature_columns
from src.modeling.sklearn_models import GBTTwoHeadModel, RandomForestTwoHeadModel, RidgeTwoHeadModel


TARGET_TOTAL = "h2_total"
TARGET_MARGIN = "h2_margin"


def default_models(*, include_xgb: bool, include_cat: bool) -> List[BaseTwoHeadModel]:
    models: List[BaseTwoHeadModel] = [
        RidgeTwoHeadModel(alpha=2.0, feature_version="v1"),
        RandomForestTwoHeadModel(feature_version="v1"),
        GBTTwoHeadModel(feature_version="v1"),
    ]

    if include_xgb:
        from src.modeling.xgb_models import XGBoostTwoHeadModel

        models.append(XGBoostTwoHeadModel(feature_version="v1"))

    if include_cat:
        from src.modeling.cat_models import CatBoostTwoHeadModel

        models.append(CatBoostTwoHeadModel(feature_version="v1"))

    return models


def run_backtest(
    *,
    parquet_path: Path,
    box_dir: Path,
    out_csv: Path,
    spec: FoldSpec,
    include_xgb: bool,
    include_cat: bool,
) -> None:
    df = pd.read_parquet(parquet_path)
    df = attach_game_time_utc(df, box_dir=box_dir)
    df = df.sort_values("gameTimeUTC").reset_index(drop=True)

    feats = feature_columns(df, ignore={"gameTimeUTC"})

    X_all = df[feats].to_numpy(dtype=float)
    y_total_all = df[TARGET_TOTAL].to_numpy(dtype=float)
    y_margin_all = df[TARGET_MARGIN].to_numpy(dtype=float)

    models = default_models(include_xgb=include_xgb, include_cat=include_cat)

    rows: List[Dict[str, object]] = []

    for fold_i, (tr, te) in enumerate(iter_walkforward_indices(len(df), spec=spec), start=1):
        X_tr, X_te = X_all[tr], X_all[te]
        yt_tr, yt_te = y_total_all[tr], y_total_all[te]
        ym_tr, ym_te = y_margin_all[tr], y_margin_all[te]

        for m in models:
            # Fit fresh each fold (proper backtest)
            mf = m.__class__(feature_version=m.feature_version)  # type: ignore[call-arg]
            mf.fit(X_tr, feats, yt_tr, ym_tr)
            mu_t, mu_m = mf.predict_heads(X_te)

            heads = mf.trained_heads()
            sig_t = float(heads.total.residual_sigma)
            sig_m = float(heads.margin.residual_sigma)

            # 80% normal PI ~ z=1.28155
            z = 1.2815515655446004
            t_lo, t_hi = mu_t - z * sig_t, mu_t + z * sig_t
            m_lo, m_hi = mu_m - z * sig_m, mu_m + z * sig_m

            # Win prob from margin distribution
            p_win = p_home_win(mu_m, sig_m)
            y_win = (ym_te > 0).astype(float)

            rows.append(
                {
                    "fold": fold_i,
                    "model": mf.name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "mae_total": mae(yt_te, mu_t),
                    "rmse_total": rmse(yt_te, mu_t),
                    "mae_margin": mae(ym_te, mu_m),
                    "rmse_margin": rmse(ym_te, mu_m),
                    "pi80_cov_total": coverage(yt_te, t_lo, t_hi),
                    "pi80_cov_margin": coverage(ym_te, m_lo, m_hi),
                    "pi80_width_total": float(np.mean(t_hi - t_lo)),
                    "pi80_width_margin": float(np.mean(m_hi - m_lo)),
                    "brier_win": brier(y_win, p_win),
                }
            )

    res = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    print("\n=== Walk-forward backtest (fold-averaged) ===")
    g = res.groupby("model")
    summary = g[[
        "mae_total",
        "rmse_total",
        "mae_margin",
        "rmse_margin",
        "pi80_cov_total",
        "pi80_cov_margin",
        "pi80_width_total",
        "pi80_width_margin",
        "brier_win",
    ]].mean().sort_values("rmse_total")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved fold metrics -> {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/halftime_training_23_24_enriched.parquet"),
        help="Leakage-safe enriched parquet",
    )
    ap.add_argument("--box-dir", type=Path, default=Path("data/raw/box"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/walkforward_backtest.csv"))

    ap.add_argument("--train-min", type=int, default=500)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--step-size", type=int, default=200)

    ap.add_argument("--include-xgb", action="store_true")
    ap.add_argument("--include-cat", action="store_true")

    args = ap.parse_args()

    run_backtest(
        parquet_path=args.data,
        box_dir=args.box_dir,
        out_csv=args.out,
        spec=FoldSpec(train_min=args.train_min, test_size=args.test_size, step_size=args.step_size),
        include_xgb=args.include_xgb,
        include_cat=args.include_cat,
    )


if __name__ == "__main__":
    main()
