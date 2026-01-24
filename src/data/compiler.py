from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.data.schedule import fetch_game_ids_for_seasons, load_game_ids, save_game_ids
from src.oddsportal.priors_store import load_priors_by_game_id, priors_to_features
from src.predict_from_gameid_v2 import (
    fetch_box,
    fetch_pbp_df,
    first_half_score,
    behavior_counts_1h,
    team_totals_from_box_team,
    add_rate_features,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _load_or_fetch_box(gid: str, cache_dir: Path) -> dict:
    fp = cache_dir / f"{gid}.json"
    if fp.exists():
        return _read_json(fp)["game"]
    game = fetch_box(gid)
    # store full CDN response shape for future-proofing
    _write_json(fp, {"game": game})
    return game


def _load_or_fetch_pbp_df(gid: str, cache_dir: Path) -> pd.DataFrame:
    fp = cache_dir / f"{gid}.json"
    if fp.exists():
        data = _read_json(fp)
        # expected shape: {"game": {"actions": [...]}}
        actions = (data.get("game") or {}).get("actions") or []
        return pd.DataFrame(actions)

    df = fetch_pbp_df(gid)
    _write_json(fp, {"game": {"actions": df.to_dict(orient="records")}})
    return df


def _final_score_from_box(game: dict) -> Optional[tuple[int, int]]:
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    hs = int((home.get("statistics", {}) or {}).get("points") or 0)
    a = int((away.get("statistics", {}) or {}).get("points") or 0)
    if hs == 0 and a == 0:
        return None
    return (hs, a)


@dataclass(frozen=True)
class CompileConfig:
    season_end_yy_a: int
    season_end_yy_b: int
    out_dir: str = "data/processed"
    out_name: str = "halftime_training_2seasons.parquet"
    game_ids_path: str | None = None

    # Raw caching (resume-safe)
    cache_box_dir: str = "data/raw/box"
    cache_pbp_dir: str = "data/raw/pbp"

    sleep_min_s: float = 0.12
    sleep_max_s: float = 0.30


def compile_two_seasons(cfg: CompileConfig) -> Path:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.game_ids_path:
        games = load_game_ids(cfg.game_ids_path)
    else:
        games = fetch_game_ids_for_seasons(season_end_yy=[cfg.season_end_yy_a, cfg.season_end_yy_b])
        ids_path = out_dir / f"game_ids_20{cfg.season_end_yy_a:02d}_20{cfg.season_end_yy_b:02d}.json"
        save_game_ids(str(ids_path), games)

    # Prefer completed games if gameStatus is available.
    # If gameStatus is missing (e.g., nba_api export), fall back to all games and let
    # the boxscore final-score check decide what is usable.
    completed = [g.gameId for g in games if int(g.gameStatus or 0) == 3]
    game_ids = completed if completed else [g.gameId for g in games]

    box_cache = Path(cfg.cache_box_dir)
    pbp_cache = Path(cfg.cache_pbp_dir)
    box_cache.mkdir(parents=True, exist_ok=True)
    pbp_cache.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    skipped = 0

    priors_map = load_priors_by_game_id()
    print(f"Loaded market priors: {len(priors_map)} games", flush=True)

    total_n = len(game_ids)
    print(f"Compiling {total_n} games...", flush=True)

    for i, gid in enumerate(game_ids, start=1):
        try:
            game = _load_or_fetch_box(gid, box_cache)
            pbp = _load_or_fetch_pbp_df(gid, pbp_cache)

            h1_home, h1_away = first_half_score(game)
            fin = _final_score_from_box(game)
            if fin is None:
                skipped += 1
                continue
            fin_home, fin_away = fin

            # Targets
            h2_home = fin_home - h1_home
            h2_away = fin_away - h1_away

            row = {
                "game_id": gid,
                "season_end_yy": int(gid[3:5]) if len(gid) >= 5 else None,
                "h1_home": int(h1_home),
                "h1_away": int(h1_away),
                "h1_total": int(h1_home + h1_away),
                "h1_margin": int(h1_home - h1_away),
            }

            # Market priors (pregame total/spread/team totals)
            row.update(priors_to_features(priors_map.get(gid)))

            # Behavior counts
            row.update(behavior_counts_1h(pbp))

            # Rate features from box totals (note: these are full game totals in CDN box endpoint)
            home = game.get("homeTeam", {}) or {}
            away = game.get("awayTeam", {}) or {}
            ht = team_totals_from_box_team(home)
            at = team_totals_from_box_team(away)
            row.update(add_rate_features("home", ht, at))
            row.update(add_rate_features("away", at, ht))

            # Labels
            row.update(
                {
                    "h2_total": float(h2_home + h2_away),
                    "h2_margin": float(h2_home - h2_away),
                    "final_total": float(fin_home + fin_away),
                    "final_margin": float(fin_home - fin_away),
                }
            )

            rows.append(row)

        except Exception:
            skipped += 1

        if i % 50 == 0:
            print(f"Progress: {i}/{total_n}  kept={len(rows)}  skipped={skipped}", flush=True)

        time.sleep(random.uniform(cfg.sleep_min_s, cfg.sleep_max_s))

    df = pd.DataFrame(rows)
    out_path = out_dir / cfg.out_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    # Atomic write to avoid corrupt parquet on interruption.
    df.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    print(f"Saved: {out_path}", flush=True)
    print(f"Rows kept: {len(df)}", flush=True)
    print(f"Rows skipped: {skipped}", flush=True)

    return out_path


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--season-a", type=int, required=True, help="Season end YY, e.g. 24 for 2024-25")
    ap.add_argument("--season-b", type=int, required=True, help="Season end YY")
    ap.add_argument("--game-ids", type=str, default=None, help="Optional path to precomputed game_ids JSON")
    ap.add_argument("--cache-box", type=str, default="data/raw/box")
    ap.add_argument("--cache-pbp", type=str, default="data/raw/pbp")
    ap.add_argument("--out", type=str, default="data/processed/halftime_training_2seasons.parquet")
    ap.add_argument("--sleep-min", type=float, default=0.12)
    ap.add_argument("--sleep-max", type=float, default=0.30)
    args = ap.parse_args()

    cfg = CompileConfig(
        season_end_yy_a=args.season_a,
        season_end_yy_b=args.season_b,
        out_name=Path(args.out).name,
        game_ids_path=args.game_ids,
        cache_box_dir=args.cache_box,
        cache_pbp_dir=args.cache_pbp,
        sleep_min_s=float(args.sleep_min),
        sleep_max_s=float(args.sleep_max),
    )
    compile_two_seasons(cfg)


if __name__ == "__main__":
    main()
