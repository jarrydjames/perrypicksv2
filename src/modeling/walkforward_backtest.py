from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.modeling.backtest_utils import FoldSpec, attach_game_time_utc, brier, coverage, ece, iter_walkforward_indices, mae, p_home_win, rmse
from src.modeling.roi_sim import SimConfig, roi, simulate_threshold_strategy
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
    drop_market_priors: bool = False,
    run_roi: bool = False,
    roi_edge_threshold: float = 0.06,
    roi_odds: int = -110,
) -> None:
    df = pd.read_parquet(parquet_path)
    df = attach_game_time_utc(df, box_dir=box_dir)

    if drop_market_priors:
        drop_cols = [
            "market_total_line",
            "market_home_spread_line",
            "market_home_team_total_line",
            "market_away_team_total_line",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
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

            sim = None
            if run_roi:
                # NOTE: We do not have historic moneyline prices in the dataset.
                # This sim assumes a fixed -110 price (synthetic), so treat ROI as relative.
                cfg = SimConfig(stake=100.0, edge_threshold=float(roi_edge_threshold), odds=int(roi_odds))
                sim = simulate_threshold_strategy(p=p_win, y=y_win, line=np.zeros_like(y_win), cfg=cfg, bet_over=True)

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
                    "ece_win": ece(y_win, p_win, n_bins=10),
                    "roi": roi(sim) if sim else np.nan,
                    "n_bets": float(sim.n_bets) if sim else np.nan,
                    "max_drawdown": float(sim.max_drawdown) if sim else np.nan,
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
        "ece_win",
        "roi",
        "n_bets",
        "max_drawdown",
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
    ap.add_argument(
        "--drop-market-priors",
        action="store_true",
        help="Ablation: remove market_* features before training/backtest",
    )

    ap.add_argument("--roi", action="store_true", help="Simulate (synthetic) betting ROI")
    ap.add_argument("--roi-edge", type=float, default=0.06, help="Edge threshold to place bet")
    ap.add_argument("--roi-odds", type=int, default=-110, help="American odds used for ROI sim")

    args = ap.parse_args()

    run_backtest(
        parquet_path=args.data,
        box_dir=args.box_dir,
        out_csv=args.out,
        spec=FoldSpec(train_min=args.train_min, test_size=args.test_size, step_size=args.step_size),
        include_xgb=args.include_xgb,
        include_cat=args.include_cat,
        drop_market_priors=bool(args.drop_market_priors),
        run_roi=bool(args.roi),
        roi_edge_threshold=float(args.roi_edge),
        roi_odds=int(args.roi_odds),
    )


if __name__ == "__main__":
    main()
