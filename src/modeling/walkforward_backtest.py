from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.modeling.backtest_utils import FoldSpec, attach_game_time_utc, brier, coverage, ece, iter_walkforward_indices, mae, p_home_win, rmse
from src.modeling.roi_sim import SimConfig, roi, simulate_threshold_strategy
from src.betting import prob_moneyline_win_from_mean_sd, prob_over_under_from_mean_sd, prob_spread_cover_from_mean_sd
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
    pi_method: str = "normal",
    calibration: bool = False,
    calibration_outdir: Path | None = None,
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

    # For calibration evaluation vs market (full game)
    h1_total_all = df.get("h1_total", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    h1_margin_all = df.get("h1_margin", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    final_total_all = df.get("final_total", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    final_margin_all = df.get("final_margin", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    market_total_all = df.get("market_total_line", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    market_spread_all = df.get("market_home_spread_line", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    models = default_models(include_xgb=include_xgb, include_cat=include_cat)

    pi_method = str(pi_method or "normal").strip().lower()
    if pi_method not in {"normal", "quantile"}:
        raise ValueError("pi_method must be one of: normal, quantile")

    if calibration and calibration_outdir is None:
        calibration_outdir = Path("reports/calibration")

    cal: Dict[str, Dict[str, Dict[str, list]]] = {}
    if calibration:
        for m in models:
            cal[m.name] = {
                "win": {"p": [], "y": []},
                "total_over": {"p": [], "y": []},
                "spread_home": {"p": [], "y": []},
            }

    rows: List[Dict[str, object]] = []

    for fold_i, (tr, te) in enumerate(iter_walkforward_indices(len(df), spec=spec), start=1):
        X_tr, X_te = X_all[tr], X_all[te]
        yt_tr, yt_te = y_total_all[tr], y_total_all[te]
        ym_tr, ym_te = y_margin_all[tr], y_margin_all[te]

        h1_total_te = h1_total_all[te]
        h1_margin_te = h1_margin_all[te]
        final_total_te = final_total_all[te]
        final_margin_te = final_margin_all[te]
        market_total_te = market_total_all[te]
        market_spread_te = market_spread_all[te]

        for m in models:
            # Fit fresh each fold (proper backtest)
            mf = m.__class__(feature_version=m.feature_version)  # type: ignore[call-arg]
            mf.fit(X_tr, feats, yt_tr, ym_tr)
            mu_t, mu_m = mf.predict_heads(X_te)

            heads = mf.trained_heads()
            sig_t = float(heads.total.residual_sigma)
            sig_m = float(heads.margin.residual_sigma)

            # 80% PI for 2H targets
            if pi_method == "quantile":
                from sklearn.ensemble import GradientBoostingRegressor
                from sklearn.impute import SimpleImputer
                from sklearn.pipeline import Pipeline

                def q_model(alpha: float):
                    return Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            (
                                "model",
                                GradientBoostingRegressor(loss="quantile", alpha=float(alpha), random_state=0),
                            ),
                        ]
                    )

                q10_t = q_model(0.10).fit(X_tr, yt_tr)
                q90_t = q_model(0.90).fit(X_tr, yt_tr)
                q10_m = q_model(0.10).fit(X_tr, ym_tr)
                q90_m = q_model(0.90).fit(X_tr, ym_tr)

                t_lo, t_hi = q10_t.predict(X_te), q90_t.predict(X_te)
                m_lo, m_hi = q10_m.predict(X_te), q90_m.predict(X_te)
            else:
                z = 1.2815515655446004
                t_lo, t_hi = mu_t - z * sig_t, mu_t + z * sig_t
                m_lo, m_hi = mu_m - z * sig_m, mu_m + z * sig_m

            # Full-game derived distributions: halftime is known constant
            mu_final_total = h1_total_te + mu_t
            mu_final_margin = h1_margin_te + mu_m

            # Market calibration (only where we have market lines + actual finals)
            over_mask = np.isfinite(market_total_te) & np.isfinite(final_total_te)
            spread_mask = np.isfinite(market_spread_te) & np.isfinite(final_margin_te)
            win_mask = np.isfinite(final_margin_te)

            # Vectorized win probability from final margin distribution
            p_win = p_home_win(mu_final_margin, sig_m)
            y_win = (final_margin_te > 0).astype(float)

            p_over = np.full_like(p_win, np.nan)
            y_over = np.full_like(y_win, np.nan)
            if np.any(over_mask):
                p_over[over_mask] = [
                    prob_over_under_from_mean_sd(mu=float(mu_final_total[i]), sd=float(sig_t), line=float(market_total_te[i]))
                    for i in np.where(over_mask)[0]
                ]
                y_over[over_mask] = (final_total_te[over_mask] > market_total_te[over_mask]).astype(float)

            p_cover = np.full_like(p_win, np.nan)
            y_cover = np.full_like(y_win, np.nan)
            if np.any(spread_mask):
                p_cover[spread_mask] = [
                    prob_spread_cover_from_mean_sd(
                        mu_margin=float(mu_final_margin[i]),
                        sd_margin=float(sig_m),
                        spread_line_home=float(market_spread_te[i]),
                    )
                    for i in np.where(spread_mask)[0]
                ]
                y_cover[spread_mask] = ((final_margin_te[spread_mask] + market_spread_te[spread_mask]) > 0).astype(float)

            # Optional ROI sim (still synthetic odds)
            sim = None
            if run_roi:
                cfg = SimConfig(stake=100.0, edge_threshold=float(roi_edge_threshold), odds=int(roi_odds))
                sim = simulate_threshold_strategy(p=p_win, y=y_win, line=np.zeros_like(y_win), cfg=cfg, bet_over=True)

            # Accumulate calibration curves (per model)
            if calibration:
                cal[mf.name]["win"]["p"].extend([float(x) for x in p_win])
                cal[mf.name]["win"]["y"].extend([float(x) for x in y_win])
                cal[mf.name]["total_over"]["p"].extend([float(x) for x in p_over[over_mask]])
                cal[mf.name]["total_over"]["y"].extend([float(x) for x in y_over[over_mask]])
                cal[mf.name]["spread_home"]["p"].extend([float(x) for x in p_cover[spread_mask]])
                cal[mf.name]["spread_home"]["y"].extend([float(x) for x in y_cover[spread_mask]])

            rows.append(
                {
                    "fold": fold_i,
                    "model": mf.name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "pi_method": pi_method,
                    # 2H accuracy
                    "mae_total": mae(yt_te, mu_t),
                    "rmse_total": rmse(yt_te, mu_t),
                    "mae_margin": mae(ym_te, mu_m),
                    "rmse_margin": rmse(ym_te, mu_m),
                    # PI quality (2H)
                    "pi80_cov_total": coverage(yt_te, t_lo, t_hi),
                    "pi80_cov_margin": coverage(ym_te, m_lo, m_hi),
                    "pi80_width_total": float(np.mean(t_hi - t_lo)),
                    "pi80_width_margin": float(np.mean(m_hi - m_lo)),
                    # Calibration vs market (full game)
                    "brier_ml": brier(y_win, p_win),
                    "ece_ml": ece(y_win, p_win, n_bins=10),
                    "brier_total_over": brier(y_over[over_mask], p_over[over_mask]) if np.any(over_mask) else np.nan,
                    "ece_total_over": ece(y_over[over_mask], p_over[over_mask], n_bins=10) if np.any(over_mask) else np.nan,
                    "brier_spread_home": brier(y_cover[spread_mask], p_cover[spread_mask]) if np.any(spread_mask) else np.nan,
                    "ece_spread_home": ece(y_cover[spread_mask], p_cover[spread_mask], n_bins=10) if np.any(spread_mask) else np.nan,
                    # ROI sim
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
        "brier_ml",
        "ece_ml",
        "brier_total_over",
        "ece_total_over",
        "brier_spread_home",
        "ece_spread_home",
        "roi",
        "n_bets",
        "max_drawdown",
    ]].mean().sort_values("rmse_total")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved fold metrics -> {out_csv}")

    if calibration:
        from src.modeling.calibration import calibration_curve_bins, save_reliability_plot

        assert calibration_outdir is not None

        for model_name, by_market in cal.items():
            for market_name, d in by_market.items():
                y = d.get("y") or []
                p = d.get("p") or []
                if len(y) < 25:
                    continue
                curve = calibration_curve_bins(y_true=y, p_pred=p, n_bins=10)
                save_reliability_plot(
                    curve=curve,
                    out_path=calibration_outdir / f"reliability_{model_name}_{market_name}.png",
                    title=f"{model_name} — {market_name}",
                )

        print(f"Saved calibration plots -> {calibration_outdir}")


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

    ap.add_argument(
        "--pi-method",
        choices=["normal", "quantile"],
        default="normal",
        help="PI method for 2H intervals (normal uses residual sigma; quantile fits q10/q90 per fold)",
    )
    ap.add_argument("--calibration", action="store_true", help="Generate reliability plots (dev-only)")
    ap.add_argument("--calibration-outdir", type=Path, default=Path("reports/calibration"))

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
        pi_method=str(args.pi_method),
        calibration=bool(args.calibration),
        calibration_outdir=Path(args.calibration_outdir) if args.calibration else None,
    )


if __name__ == "__main__":
    main()
