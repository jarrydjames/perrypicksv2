from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.features.pbp_possessions import game_possessions_first_half


def _load_pbp_actions(pbp_path: Path) -> list[dict]:
    data = json.loads(pbp_path.read_text())
    return (data.get("game") or {}).get("actions") or []


def _load_box_game(box_path: Path) -> dict:
    data = json.loads(box_path.read_text())
    # our box cache uses {"game": { ... }}
    return data.get("game") or {}


def enrich_parquet(
    *,
    in_parquet: Path,
    out_parquet: Path,
    box_dir: Path = Path("data/raw/box"),
    pbp_dir: Path = Path("data/raw/pbp"),
) -> None:
    """Post-compile enrichment step.

    Adds pace/possessions/PPP first-half features derived from cached PBP.

    This is intentionally separate from the compiler so it won’t interfere with long-running
    data collection jobs.
    """

    df = pd.read_parquet(in_parquet)

    add_rows: list[Dict[str, Any]] = []

    game_ids = df["game_id"].astype(str).tolist()
    total_n = len(game_ids)
    print(f"Enriching {total_n} games...")

    for i, gid in enumerate(game_ids, start=1):
        pbp_path = pbp_dir / f"{gid}.json"
        box_path = box_dir / f"{gid}.json"

        if not pbp_path.exists() or not box_path.exists():
            add_rows.append({"game_id": gid})
            continue

        actions = _load_pbp_actions(pbp_path)
        game = _load_box_game(box_path)

        home_tri = ((game.get("homeTeam") or {}).get("teamTricode") or "HOME")
        away_tri = ((game.get("awayTeam") or {}).get("teamTricode") or "AWAY")

        feats = game_possessions_first_half(actions, home_tri=home_tri, away_tri=away_tri)
        feats["game_id"] = gid
        add_rows.append(feats)

        if i % 100 == 0:
            print(f"Progress: {i}/{total_n}")

    add_df = pd.DataFrame(add_rows)

    # IMPORTANT: prevent leakage.
    # The base parquet currently contains `home_efg/home_ftr/...` computed from FULL-GAME box stats.
    # We drop them here and replace with 1H PBP-derived values from `add_df`.
    leaky_cols = [
        "home_efg",
        "home_ftr",
        "home_tpar",
        "home_tor",
        "home_orbp",
        "away_efg",
        "away_ftr",
        "away_tpar",
        "away_tor",
        "away_orbp",
    ]
    df = df.drop(columns=[c for c in leaky_cols if c in df.columns])

    out = df.merge(add_df, on="game_id", how="left")

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_parquet, index=False)
    print(f"Wrote enriched parquet: {out_parquet}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="outp", type=Path, required=True)
    ap.add_argument("--box-dir", type=Path, default=Path("data/raw/box"))
    ap.add_argument("--pbp-dir", type=Path, default=Path("data/raw/pbp"))
    args = ap.parse_args()

    enrich_parquet(in_parquet=args.inp, out_parquet=args.outp, box_dir=args.box_dir, pbp_dir=args.pbp_dir)


if __name__ == "__main__":
    main()
