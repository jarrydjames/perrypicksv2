from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_TRAINING_PARQUET = Path("data/processed/halftime_training_23_24_enriched.parquet")


@dataclass(frozen=True)
class TrainingDataSpec:
    path: Path
    min_rows: int = 2000


def load_training_df(spec: TrainingDataSpec) -> pd.DataFrame:
    """Load training data with guardrails.

    Why so strict? Because training on the wrong (tiny) dataset silently produces
    garbage models that look "fine" but are absolutely not.
    """
    path = Path(spec.path)
    if not path.exists():
        raise FileNotFoundError(
            f"Training parquet not found: {path}. "
            f"Expected default at {DEFAULT_TRAINING_PARQUET}."
        )

    df = pd.read_parquet(path)

    if len(df) < int(spec.min_rows):
        raise ValueError(
            f"Training data too small: {path} has {len(df)} rows, expected >= {spec.min_rows}. "
            "You are probably pointing at data_processed/* artifacts instead of data/processed/* training data."
        )

    return df
