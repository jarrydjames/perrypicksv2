from __future__ import annotations

from typing import List, Set

import pandas as pd


DEFAULT_IGNORE: Set[str] = {
    "game_id",
    "season_end_yy",
    "h2_total",
    "h2_margin",
    "final_total",
    "final_margin",
}


def feature_columns(df: pd.DataFrame, *, ignore: Set[str] | None = None) -> List[str]:
    """Deterministic feature selection.

    Keep this shared across training/backtests to avoid train/serve skew.
    """

    ig = set(DEFAULT_IGNORE)
    if ignore:
        ig |= set(ignore)

    cols = [c for c in df.columns if c not in ig]
    cols.sort()
    return cols
