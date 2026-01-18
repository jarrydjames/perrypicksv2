from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.modeling.backtest_utils import FoldSpec, attach_game_time_utc, brier, coverage, iter_walkforward_indices, mae, p_home_win, rmse
from src.modeling.base import BaseTwoHeadModel
from src.modeling.feature_columns import feature_columns
from src.modeling.sklearn_models import GBTTwoHeadModel, RandomForestTwoHeadModel, RidgeTwoHeadModel


TARGET_TOTAL = "h2_total"
TARGET_MARGIN = "h2_margin"


@dataclass(frozen=True)
class NestedSpec:
    inner_folds: int
    trials: int
    seed: int


def _score_objective(met: Dict[str, float]) -> float:
    """Objective for inner-loop tuning.

    This should reflect what we actually care about for betting:
    - point accuracy (RMSE)
    - calibration (PI80 coverage near 0.80)
    - win-prob calibration (Brier)

    Lower is better.
    """

    rmse_total = float(met.get("rmse_total", 1e9))
    rmse_margin = float(met.get("rmse_margin", 1e9))

    cov_total = float(met.get("pi80_cov_total", 0.0))
    cov_margin = float(met.get("pi80_cov_margin", 0.0))
    brier_win = float(met.get("brier_win", 1.0))

    # With post-hoc sigma calibration, coverage matters less in tuning.
    # Keep a small nudge so we don't learn completely insane sigmas.
    w_cov = 3.0
    w_brier = 2.0

    cov_pen = abs(cov_total - 0.80) + abs(cov_margin - 0.80)

    return (rmse_total + rmse_margin) + (w_cov * cov_pen) + (w_brier * brier_win)


def _fmt_elapsed(start: float) -> str:
    secs = max(0.0, time.perf_counter() - float(start))
    if secs < 60:
        return f"{secs:.1f}s"
    mins = secs / 60.0
    if mins < 60:
        return f"{mins:.1f}m"
    hrs = mins / 60.0
    return f"{hrs:.2f}h"


def _inner_splits(n: int, *, inner_folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create time-series inner folds over [0..n).

    We do expanding window with equal-sized test blocks.
    """

    inner_folds = int(max(1, inner_folds))
    test_size = max(1, n // (inner_folds + 1))
    train_min = max(50, test_size)

    spec = FoldSpec(train_min=train_min, test_size=test_size, step_size=test_size)
    return list(iter_walkforward_indices(n, spec=spec))[:inner_folds]


def _sample_xgb_params(rng: np.random.Generator) -> Dict[str, Any]:
    # Keep the search space sane for laptop runtimes.
    return {
        "n_estimators": int(rng.integers(400, 1400)),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "max_depth": int(rng.integers(3, 7)),
        "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        "min_child_weight": float(rng.choice([1.0, 3.0, 5.0, 8.0])),
        "reg_lambda": float(rng.choice([0.5, 1.0, 2.0, 5.0])),
        "n_jobs": -1,
    }


def _sample_cat_params(rng: np.random.Generator) -> Dict[str, Any]:
    return {
        "iterations": int(rng.integers(800, 3000)),
        "learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
        "depth": int(rng.integers(4, 9)),
        "l2_leaf_reg": float(rng.choice([2.0, 3.0, 5.0, 8.0])),
        "subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
    }


def _abs_residual_quantile(residuals: np.ndarray, *, target_cov: float = 0.80) -> float:
    """Return q where q ~= quantile(|residual|, target_cov).

    We use this to calibrate sigma post-hoc:
      want z*sigma*k ~= q  =>  k ~= q/(z*sigma)

    Returns q in the same units as the target (points).
    """

    target_cov = float(target_cov)
    target_cov = min(0.99, max(0.50, target_cov))

    abs_res = np.abs(np.asarray(residuals, dtype=float))
    abs_res = abs_res[np.isfinite(abs_res)]
    if abs_res.size < 10:
        return 1.0

    # central symmetric coverage target -> quantile at target_cov
    q = float(np.quantile(abs_res, target_cov))
    return max(0.5, min(3.0, q))


def _fit_eval_model(
    model: BaseTwoHeadModel,
    *,
    X_tr: np.ndarray,
    ytot_tr: np.ndarray,
    ymar_tr: np.ndarray,
    X_te: np.ndarray,
    ytot_te: np.ndarray,
    ymar_te: np.ndarray,
    feature_names: List[str],
    calib_frac: float = 0.15,
) -> Dict[str, float]:
    """Fit model, calibrate sigma on a tail slice, evaluate on test.

    We keep time-series integrity by calibrating on the *most recent* part
    of the training window.
    """

    n_tr = int(len(X_tr))
    n_cal = int(max(50, round(n_tr * float(calib_frac))))
    n_cal = min(n_cal, max(0, n_tr - 50))

    if n_cal > 0:
        X_fit, X_cal = X_tr[:-n_cal], X_tr[-n_cal:]
        yt_fit, yt_cal = ytot_tr[:-n_cal], ytot_tr[-n_cal:]
        ym_fit, ym_cal = ymar_tr[:-n_cal], ymar_tr[-n_cal:]

        model.fit(X_fit, feature_names, yt_fit, ym_fit)
        mu_t_cal, mu_m_cal = model.predict_heads(X_cal)

        # Get raw sigmas from the fit-only model
        heads_fit = model.trained_heads()
        sig_t_raw = float(heads_fit.total.residual_sigma)
        sig_m_raw = float(heads_fit.margin.residual_sigma)

        z = 1.2815515655446004
        # Compute required sigma so that z*sigma ~= quantile(|residual|)
        q_t = _abs_residual_quantile(yt_cal - mu_t_cal, target_cov=0.80)
        q_m = _abs_residual_quantile(ym_cal - mu_m_cal, target_cov=0.80)

        # q is in "points"; convert to multiplicative factor on sigma
        k_t = q_t / max(1e-6, (z * sig_t_raw))
        k_m = q_m / max(1e-6, (z * sig_m_raw))
        k_t = float(max(0.5, min(3.0, k_t)))
        k_m = float(max(0.5, min(3.0, k_m)))
    else:
        k_t = 1.0
        k_m = 1.0

    # Refit on full training for best means, then apply calibrated sigma scaling.
    model.fit(X_tr, feature_names, ytot_tr, ymar_tr)
    mu_t, mu_m = model.predict_heads(X_te)

    heads = model.trained_heads()
    sig_t = float(heads.total.residual_sigma) * float(k_t)
    sig_m = float(heads.margin.residual_sigma) * float(k_m)

    z = 1.2815515655446004
    t_lo, t_hi = mu_t - z * sig_t, mu_t + z * sig_t
    m_lo, m_hi = mu_m - z * sig_m, mu_m + z * sig_m

    p_win = p_home_win(mu_m, sig_m)
    y_win = (ymar_te > 0).astype(float)

    return {
        "mae_total": mae(ytot_te, mu_t),
        "rmse_total": rmse(ytot_te, mu_t),
        "mae_margin": mae(ymar_te, mu_m),
        "rmse_margin": rmse(ymar_te, mu_m),
        "pi80_cov_total": coverage(ytot_te, t_lo, t_hi),
        "pi80_cov_margin": coverage(ymar_te, m_lo, m_hi),
        "pi80_width_total": float(np.mean(t_hi - t_lo)),
        "pi80_width_margin": float(np.mean(m_hi - m_lo)),
        "brier_win": brier(y_win, p_win),
        "sigma_total": sig_t,
        "sigma_margin": sig_m,
        "sigma_k_total": float(k_t),
        "sigma_k_margin": float(k_m),
        "n_cal": float(n_cal),
    }


def _tune_xgb(
    *,
    X: np.ndarray,
    ytot: np.ndarray,
    ymar: np.ndarray,
    feature_names: List[str],
    inner_folds: int,
    trials: int,
    rng: np.random.Generator,
    log_prefix: str = "",
    start_ts: float | None = None,
) -> Tuple[Dict[str, Any], float]:
    from src.modeling.xgb_models import XGBoostTwoHeadModel

    splits = _inner_splits(len(X), inner_folds=inner_folds)

    best_params: Dict[str, Any] | None = None
    best_score = float("inf")

    start_ts = start_ts if start_ts is not None else time.perf_counter()

    for trial_i in range(int(trials)):
        params = _sample_xgb_params(rng)

        fold_scores: List[float] = []
        for tr, te in splits:
            m = XGBoostTwoHeadModel(feature_version="v1", **params)
            met = _fit_eval_model(
                m,
                X_tr=X[tr],
                ytot_tr=ytot[tr],
                ymar_tr=ymar[tr],
                X_te=X[te],
                ytot_te=ytot[te],
                ymar_te=ymar[te],
                feature_names=feature_names,
            )
            fold_scores.append(_score_objective({k: float(v) for k, v in met.items()}))

        s = float(np.mean(fold_scores)) if fold_scores else float("inf")
        if s < best_score:
            best_score = s
            best_params = params

        if (trial_i + 1) == 1 or (trial_i + 1) == int(trials) or (trial_i + 1) % 5 == 0:
            print(
                f"{log_prefix}XGB tune trial {trial_i+1}/{int(trials)}  best={best_score:.4f}  elapsed={_fmt_elapsed(start_ts)}",
                flush=True,
            )

    if best_params is None:
        best_params = _sample_xgb_params(rng)

    return best_params, best_score


def _tune_cat(
    *,
    X: np.ndarray,
    ytot: np.ndarray,
    ymar: np.ndarray,
    feature_names: List[str],
    inner_folds: int,
    trials: int,
    rng: np.random.Generator,
    log_prefix: str = "",
    start_ts: float | None = None,
) -> Tuple[Dict[str, Any], float]:
    from src.modeling.cat_models import CatBoostTwoHeadModel

    splits = _inner_splits(len(X), inner_folds=inner_folds)

    best_params: Dict[str, Any] | None = None
    best_score = float("inf")

    start_ts = start_ts if start_ts is not None else time.perf_counter()

    for trial_i in range(int(trials)):
        params = _sample_cat_params(rng)

        fold_scores: List[float] = []
        for tr, te in splits:
            m = CatBoostTwoHeadModel(feature_version="v1", **params)
            met = _fit_eval_model(
                m,
                X_tr=X[tr],
                ytot_tr=ytot[tr],
                ymar_tr=ymar[tr],
                X_te=X[te],
                ytot_te=ytot[te],
                ymar_te=ymar[te],
                feature_names=feature_names,
            )
            fold_scores.append(_score_objective({k: float(v) for k, v in met.items()}))

        s = float(np.mean(fold_scores)) if fold_scores else float("inf")
        if s < best_score:
            best_score = s
            best_params = params

        if (trial_i + 1) == 1 or (trial_i + 1) == int(trials) or (trial_i + 1) % 5 == 0:
            print(
                f"{log_prefix}CAT tune trial {trial_i+1}/{int(trials)}  best={best_score:.4f}  elapsed={_fmt_elapsed(start_ts)}",
                flush=True,
            )

    if best_params is None:
        best_params = _sample_cat_params(rng)

    return best_params, best_score


def run_nested_backtest(
    *,
    parquet_path: Path,
    box_dir: Path,
    out_csv: Path,
    outer: FoldSpec,
    nested: NestedSpec,
    include_xgb: bool,
    include_cat: bool,
) -> None:
    run_start = time.perf_counter()

    df = pd.read_parquet(parquet_path)
    df = attach_game_time_utc(df, box_dir=box_dir)
    df = df.sort_values("gameTimeUTC").reset_index(drop=True)

    feats = feature_columns(df, ignore={"gameTimeUTC"})

    X_all = df[feats].to_numpy(dtype=float)
    y_total_all = df[TARGET_TOTAL].to_numpy(dtype=float)
    y_margin_all = df[TARGET_MARGIN].to_numpy(dtype=float)

    rng = np.random.default_rng(int(nested.seed))

    base_models: List[BaseTwoHeadModel] = [
        RidgeTwoHeadModel(alpha=2.0, feature_version="v1"),
        RandomForestTwoHeadModel(feature_version="v1"),
        GBTTwoHeadModel(feature_version="v1"),
    ]

    rows: List[Dict[str, object]] = []

    outer_splits = list(iter_walkforward_indices(len(df), spec=outer))

    print(
        f"Nested backtest starting  n_rows={len(df)}  n_feats={len(feats)}  outer_folds={len(outer_splits)}  "
        f"inner_folds={nested.inner_folds}  trials={nested.trials}",
        flush=True,
    )

    for fold_i, (tr, te) in enumerate(outer_splits, start=1):
        fold_start = time.perf_counter()
        print(
            f"\n[fold {fold_i}/{len(outer_splits)}] n_train={len(tr)} n_test={len(te)}  elapsed={_fmt_elapsed(run_start)}",
            flush=True,
        )
        X_tr, X_te = X_all[tr], X_all[te]
        yt_tr, yt_te = y_total_all[tr], y_total_all[te]
        ym_tr, ym_te = y_margin_all[tr], y_margin_all[te]

        # Evaluate fixed models
        for m in base_models:
            print(f"[fold {fold_i}/{len(outer_splits)}] eval {m.name}", flush=True)
            mf = m.__class__(feature_version=m.feature_version)  # type: ignore[call-arg]
            met = _fit_eval_model(
                mf,
                X_tr=X_tr,
                ytot_tr=yt_tr,
                ymar_tr=ym_tr,
                X_te=X_te,
                ytot_te=yt_te,
                ymar_te=ym_te,
                feature_names=feats,
            )
            rows.append(
                {
                    "fold": fold_i,
                    "model": mf.name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "tuned": False,
                    "tune_score": None,
                    "params": None,
                    **{k: float(v) for k, v in met.items()},
                }
            )

        # Tune + evaluate XGB
        if include_xgb:
            print(f"[fold {fold_i}/{len(outer_splits)}] tuning XGB...", flush=True)
            from src.modeling.xgb_models import XGBoostTwoHeadModel

            params, tune_score = _tune_xgb(
                X=X_tr,
                ytot=yt_tr,
                ymar=ym_tr,
                feature_names=feats,
                inner_folds=nested.inner_folds,
                trials=nested.trials,
                rng=rng,
                log_prefix=f"[fold {fold_i}/{len(outer_splits)}] ",
                start_ts=fold_start,
            )
            m = XGBoostTwoHeadModel(feature_version="v1", **params)
            met = _fit_eval_model(
                m,
                X_tr=X_tr,
                ytot_tr=yt_tr,
                ymar_tr=ym_tr,
                X_te=X_te,
                ytot_te=yt_te,
                ymar_te=ym_te,
                feature_names=feats,
            )
            rows.append(
                {
                    "fold": fold_i,
                    "model": m.name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "tuned": True,
                    "tune_score": float(tune_score),
                    "params": json.dumps(params, sort_keys=True),
                    **{k: float(v) for k, v in met.items()},
                }
            )

        # Tune + evaluate Cat
        if include_cat:
            print(f"[fold {fold_i}/{len(outer_splits)}] tuning CAT...", flush=True)
            from src.modeling.cat_models import CatBoostTwoHeadModel

            params, tune_score = _tune_cat(
                X=X_tr,
                ytot=yt_tr,
                ymar=ym_tr,
                feature_names=feats,
                inner_folds=nested.inner_folds,
                trials=nested.trials,
                rng=rng,
                log_prefix=f"[fold {fold_i}/{len(outer_splits)}] ",
                start_ts=fold_start,
            )
            m = CatBoostTwoHeadModel(feature_version="v1", **params)
            met = _fit_eval_model(
                m,
                X_tr=X_tr,
                ytot_tr=yt_tr,
                ymar_tr=ym_tr,
                X_te=X_te,
                ytot_te=yt_te,
                ymar_te=ym_te,
                feature_names=feats,
            )
            rows.append(
                {
                    "fold": fold_i,
                    "model": m.name,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "tuned": True,
                    "tune_score": float(tune_score),
                    "params": json.dumps(params, sort_keys=True),
                    **{k: float(v) for k, v in met.items()},
                }
            )

    res = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)

    print("\n=== Nested walk-forward backtest (fold-averaged) ===")
    g = res.groupby("model")
    summary = g[[
        "rmse_total",
        "rmse_margin",
        "mae_total",
        "mae_margin",
        "pi80_cov_total",
        "pi80_cov_margin",
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
    ap.add_argument("--out", type=Path, default=Path("data/processed/nested_walkforward_backtest.csv"))

    ap.add_argument("--train-min", type=int, default=1000)
    ap.add_argument("--test-size", type=int, default=200)
    ap.add_argument("--step-size", type=int, default=200)

    ap.add_argument("--inner-folds", type=int, default=3)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--include-xgb", action="store_true")
    ap.add_argument("--include-cat", action="store_true")

    args = ap.parse_args()

    run_nested_backtest(
        parquet_path=args.data,
        box_dir=args.box_dir,
        out_csv=args.out,
        outer=FoldSpec(train_min=args.train_min, test_size=args.test_size, step_size=args.step_size),
        nested=NestedSpec(inner_folds=args.inner_folds, trials=args.trials, seed=args.seed),
        include_xgb=args.include_xgb,
        include_cat=args.include_cat,
    )


if __name__ == "__main__":
    main()
