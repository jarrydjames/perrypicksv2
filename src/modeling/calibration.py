from __future__ import annotations

"""Calibration helpers (dev/offline tooling).

We keep this separate from Streamlit runtime code.

- Numeric calibration is already computed via ECE/Brier in backtests.
- This module provides reliability curve bins + optional matplotlib plots.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalibrationCurve:
    prob_pred: np.ndarray
    prob_true: np.ndarray
    count: np.ndarray


def calibration_curve_bins(*, y_true: Iterable[float], p_pred: Iterable[float], n_bins: int = 10) -> CalibrationCurve:
    """Compute a reliability curve using equal-width bins over [0,1]."""

    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(p_pred), dtype=float)

    if y.shape != p.shape:
        raise ValueError("y_true and p_pred must have same shape")

    p = np.clip(p, 0.0, 1.0)

    # Bin edges including 0 and 1
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    # digitize returns 1..n_bins
    idx = np.digitize(p, edges, right=True)
    idx = np.clip(idx, 1, int(n_bins))

    prob_pred = np.zeros(int(n_bins), dtype=float)
    prob_true = np.zeros(int(n_bins), dtype=float)
    count = np.zeros(int(n_bins), dtype=int)

    for b in range(1, int(n_bins) + 1):
        mask = idx == b
        c = int(np.sum(mask))
        count[b - 1] = c
        if c <= 0:
            prob_pred[b - 1] = float("nan")
            prob_true[b - 1] = float("nan")
            continue
        prob_pred[b - 1] = float(np.mean(p[mask]))
        prob_true[b - 1] = float(np.mean(y[mask]))

    return CalibrationCurve(prob_pred=prob_pred, prob_true=prob_true, count=count)


def calibration_curve_df(curve: CalibrationCurve) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prob_pred": curve.prob_pred,
            "prob_true": curve.prob_true,
            "count": curve.count,
        }
    )


def save_reliability_plot(*, curve: CalibrationCurve, out_path: Path, title: str) -> None:
    """Save a reliability plot. Requires matplotlib (dev dependency)."""

    import matplotlib.pyplot as plt  # dev-only

    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = calibration_curve_df(curve).dropna().copy()

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Perfect")
    ax.plot(df["prob_pred"], df["prob_true"], "o-", label="Model")

    for x, y, c in zip(df["prob_pred"], df["prob_true"], df["count"], strict=False):
        ax.annotate(str(int(c)), (float(x), float(y)), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
