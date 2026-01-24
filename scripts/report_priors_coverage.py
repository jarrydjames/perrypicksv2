"""Report market-priors coverage for a training parquet.

This answers:
- how many rows in the parquet?
- how many have market priors present?

Usage:
  .venv/bin/python scripts/report_priors_coverage.py \
    --parquet data/processed/halftime_training_23_24_enriched.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)

    cols = [
        "market_total_line",
        "market_home_spread_line",
        "market_home_team_total_line",
        "market_away_team_total_line",
    ]

    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing expected market priors columns: {missing_cols}")

    n = len(df)
    has_any = df[cols].notna().any(axis=1).sum()
    has_all = df[cols].notna().all(axis=1).sum()

    print(f"Parquet: {args.parquet}")
    print(f"Rows: {n}")
    print(f"Rows with ANY market priors: {has_any} ({has_any/max(n,1):.1%})")
    print(f"Rows with ALL market priors: {has_all} ({has_all/max(n,1):.1%})")


if __name__ == "__main__":
    main()
