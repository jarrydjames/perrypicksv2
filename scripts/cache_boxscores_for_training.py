from __future__ import annotations

"""Cache boxscore JSON for game_ids in the training parquet.

Why:
- Proper walk-forward backtests require a true chronological sort.
- We derive that from boxscore.gameTimeUTC.

Usage:
  .venv/bin/python scripts/cache_boxscores_for_training.py \
    --data data/processed/halftime_training_23_24_enriched.parquet \
    --out-dir data/raw/box \
    --max-games 50

Notes:
- Safe to re-run; skips already-cached game ids.
- Uses nba.com liveData boxscore endpoint via existing fetch_box().
"""

import argparse
import json
import time
from pathlib import Path

import sys

import pandas as pd

# Allow running as a plain script without needing PYTHONPATH tweaks.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_from_gameid_v2 import fetch_box


def cache_one(*, gid: str, out_dir: Path, sleep_s: float) -> bool:
    out_path = out_dir / f"{gid}.json"
    if out_path.exists() and out_path.stat().st_size > 0:
        return False

    game = fetch_box(str(gid))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(game))

    if sleep_s > 0:
        time.sleep(float(sleep_s))

    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="Training parquet containing game_id column")
    ap.add_argument("--out-dir", type=Path, default=Path("data/raw/box"))
    ap.add_argument("--max-games", type=int, default=0, help="0 = no limit")
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between requests")

    args = ap.parse_args()

    df = pd.read_parquet(args.data, columns=["game_id"])
    gids = [str(x) for x in df["game_id"].astype(str).unique().tolist()]

    n_new = 0
    n_skip = 0
    limit = int(args.max_games)

    for gid in gids:
        if limit and (n_new + n_skip) >= limit:
            break

        try:
            wrote = cache_one(gid=gid, out_dir=args.out_dir, sleep_s=float(args.sleep))
            if wrote:
                n_new += 1
                print(f"cached {gid}")
            else:
                n_skip += 1
        except Exception as e:
            print(f"FAILED {gid}: {e!r}")

    print(f"Done. new={n_new} skipped={n_skip} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
