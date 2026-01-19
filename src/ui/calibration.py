from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_BACKTEST_CSV = Path("data/processed/nested_walkforward_backtest.csv")


def render_calibration_report(*, backtest_csv: Path = DEFAULT_BACKTEST_CSV) -> None:
    """Render a lightweight model health / calibration report.

    This reads existing offline backtest artifacts (no training, no API calls).
    """
    st.subheader("Model calibration (offline backtest)")

    p = Path(backtest_csv)
    if not p.exists():
        st.info(
            "No backtest report found. Run the offline backtest to generate one: "
            "`data/processed/nested_walkforward_backtest.csv`."
        )
        return

    df = pd.read_csv(p)
    if df.empty:
        st.warning("Backtest report CSV is empty.")
        return

    # Lower is better for Brier; coverage closer to 0.80 is better for PI80.
    df = df.copy()

    def cov_err(x: float) -> float:
        return abs(float(x) - 0.80)

    for c in ("pi80_cov_total", "pi80_cov_margin"):
        if c in df.columns:
            df[c + "_err"] = df[c].map(cov_err)

    # Pick a simple summary score: prioritize Brier (classification-ish), then coverage error.
    score_cols = [c for c in ["brier_win", "pi80_cov_total_err", "pi80_cov_margin_err"] if c in df.columns]
    if score_cols:
        df["health_score"] = df[score_cols].sum(axis=1)
        best = df.sort_values("health_score", ascending=True).head(1)
    else:
        best = None

    st.caption(
        "Brier is for win-prob calibration from margin distribution (lower is better). "
        "PI80 coverage is how often the true value landed inside the model’s 80% interval (target ≈ 0.80)."
    )

    show_cols = [
        c
        for c in [
            "fold",
            "model",
            "n_train",
            "n_test",
            "mae_total",
            "rmse_total",
            "mae_margin",
            "rmse_margin",
            "pi80_cov_total",
            "pi80_cov_margin",
            "pi80_width_total",
            "pi80_width_margin",
            "brier_win",
            "sigma_total",
            "sigma_margin",
        ]
        if c in df.columns
    ]

    st.dataframe(df[show_cols], width="stretch", height=260)

    if best is not None and not best.empty:
        r = best.iloc[0].to_dict()
        st.success(
            "Best overall (simple score): "
            f"**{r.get('model')}** on fold {int(r.get('fold'))} "
            f"(brier={float(r.get('brier_win')):.3f}, "
            f"cov_total={float(r.get('pi80_cov_total')):.3f}, cov_margin={float(r.get('pi80_cov_margin')):.3f})."
        )
