"""Build a game_ids JSON file from an existing NBA box cache directory.

Why:
- The schedule API can return empty depending on availability.
- You already have a local cache from the old repo.

This script makes a compiler-friendly game list from `data/raw/box/*.json`.

Usage:
  .venv/bin/python scripts/build_game_ids_from_box_cache.py \
    --box-dir /Users/jarrydhawley/Desktop/Predictor/perrypicks/data/raw/box \
    --out data/processed/game_ids_from_box_cache.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path


def _infer_game_id_from_filename(p: Path) -> str | None:
    # cache filename is usually {game_id}.json
    stem = p.stem.strip()
    if stem and stem.isdigit():
        return stem
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    box_dir: Path = args.box_dir.expanduser().resolve()
    out: Path = args.out.expanduser().resolve()

    game_ids: list[str] = []
    for p in sorted(box_dir.glob("*.json")):
        gid = _infer_game_id_from_filename(p)
        if gid:
            game_ids.append(gid)

    # Compiler expects schedule-like objects when using save/load_game_ids.
    # We'll emit minimal objects with gameId + completed status.
    payload = [{"gameId": gid, "gameStatus": 3} for gid in game_ids]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))

    print(f"Box dir: {box_dir}")
    print(f"Found games: {len(game_ids)}")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
