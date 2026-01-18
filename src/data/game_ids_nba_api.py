from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class GameMeta:
    gameId: str
    season: str
    gameDate: str | None
    homeTeam: str | None
    awayTeam: str | None


def fetch_game_ids_for_season(season_str: str) -> List[GameMeta]:
    """Fetch NBA game IDs for an NBA season using `nba_api`.

    season_str examples:
      - "2023-24"
      - "2024-25"

    OFFLINE dataset building only.
    """
    from nba_api.stats.endpoints import leaguegamefinder

    lgf = leaguegamefinder.LeagueGameFinder(season_nullable=season_str, league_id_nullable="00")
    df = lgf.get_data_frames()[0]

    # Each game appears twice (one per team). Collapse to unique game_id.
    out = {}
    for _, r in df.iterrows():
        gid = str(r.get("GAME_ID"))
        matchup = str(r.get("MATCHUP") or "")
        team = str(r.get("TEAM_ABBREVIATION") or "")
        game_date = str(r.get("GAME_DATE") or "")

        home = None
        away = None
        if " vs " in matchup:
            home = team
            away = matchup.split(" vs ")[-1]
        elif " @ " in matchup:
            away = team
            home = matchup.split(" @ ")[-1]

        rec = out.get(gid) or {
            "gameId": gid,
            "season": season_str,
            "gameDate": game_date,
            "homeTeam": None,
            "awayTeam": None,
        }

        if home and not rec["homeTeam"]:
            rec["homeTeam"] = home
        if away and not rec["awayTeam"]:
            rec["awayTeam"] = away

        out[gid] = rec

    games = [GameMeta(**v) for v in out.values()]
    games.sort(key=lambda g: (g.gameDate or "", g.gameId))
    return games


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", action="append", required=True, help="Season string like 2023-24. Repeatable.")
    ap.add_argument("--out", type=Path, default=Path("data/processed/game_ids_23_24.json"))
    ap.add_argument("--sleep", type=float, default=0.7, help="Seconds to sleep between season calls")
    args = ap.parse_args()

    all_games: list[dict] = []

    for i, season in enumerate(args.season):
        games = fetch_game_ids_for_season(season)
        all_games.extend([g.__dict__ for g in games])
        if i < len(args.season) - 1:
            time.sleep(float(args.sleep))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_games, indent=2), encoding="utf-8")
    print(f"Saved {len(all_games)} rows -> {args.out}")


if __name__ == "__main__":
    main()
