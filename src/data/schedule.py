from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

import requests

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"


def _season_from_game_id(game_id: str) -> int:
    """Extract season YY from NBA game_id: 002YYxxxxx (YY=season start year%100)."""
    m = re.match(r"^002(\d{2})", str(game_id))
    return int(m.group(1)) if m else -1


@dataclass(frozen=True)
class GameMeta:
    gameId: str
    gameDate: str | None
    homeTeam: str | None
    awayTeam: str | None
    gameStatus: int | None


def fetch_game_ids_for_seasons(*, season_end_yy: Sequence[int]) -> List[GameMeta]:
    """Fetch schedule and filter to desired seasons.

    season_end_yy are the YY embedded in gameId after 002.
    Example: 2025-26 season => 25.

    Returns all games (including not-final). Caller can filter.
    """
    wanted = set(int(x) for x in season_end_yy)
    sched = requests.get(SCHEDULE_URL, timeout=30).json()
    games = sched.get("leagueSchedule", {}).get("gameDates", [])

    out: List[GameMeta] = []
    for gd in games:
        for g in gd.get("games", []):
            gid = g.get("gameId")
            if not gid:
                continue
            if _season_from_game_id(gid) not in wanted:
                continue

            out.append(
                GameMeta(
                    gameId=gid,
                    gameDate=g.get("gameDateTimeUTC") or g.get("gameDateTime") or g.get("gameDate"),
                    homeTeam=(g.get("homeTeam", {}) or {}).get("teamTricode"),
                    awayTeam=(g.get("awayTeam", {}) or {}).get("teamTricode"),
                    gameStatus=g.get("gameStatus"),
                )
            )

    # De-dupe while preserving order
    seen = set()
    deduped: List[GameMeta] = []
    for g in out:
        if g.gameId in seen:
            continue
        seen.add(g.gameId)
        deduped.append(g)

    return deduped


def save_game_ids(path: str, games: List[GameMeta]) -> None:
    data = [g.__dict__ for g in games]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_game_ids(path: str) -> List[GameMeta]:
    """Load game IDs from JSON.

    Be tolerant: allow extra keys (e.g., from nba_api exports) and missing keys.
    """
    raw = json.loads(open(path, "r").read())

    out: List[GameMeta] = []
    for g in raw:
        if not isinstance(g, dict):
            continue
        out.append(
            GameMeta(
                gameId=str(g.get("gameId") or g.get("game_id") or g.get("GAME_ID") or ""),
                gameDate=g.get("gameDate") or g.get("gameDateTimeUTC") or g.get("GAME_DATE"),
                homeTeam=g.get("homeTeam"),
                awayTeam=g.get("awayTeam"),
                gameStatus=g.get("gameStatus"),
            )
        )

    # Filter invalid
    out = [g for g in out if g.gameId]
    return out
