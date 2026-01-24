from __future__ import annotations

"""NBA scoreboard fetcher (CDN).

Used for Streamlit UX:
- pick a date
- show a dropdown of games with live status (Q + clock)

Endpoint:
  https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_YYYYMMDD.json

We keep this module dependency-light and resilient to missing fields.
"""

import datetime as dt
import json
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import requests


CDN_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_{yyyymmdd}.json"


@dataclass(frozen=True)
class ScoreboardGame:
    game_id: str
    away: str
    home: str
    status_text: str
    period: Optional[int]
    clock: Optional[str]
    away_score: Optional[int] = None
    home_score: Optional[int] = None


def _safe_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _team_name(team_block: dict) -> str:
    for k in ("teamName", "teamCity", "teamTricode", "name"):
        v = team_block.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "TEAM"


def fetch_scoreboard(date: dt.date, *, timeout_s: int = 10) -> List[ScoreboardGame]:
    yyyymmdd = date.strftime("%Y%m%d")
    url = CDN_SCOREBOARD.format(yyyymmdd=yyyymmdd)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nba.com/",
    }

    r = requests.get(url, headers=headers, timeout=int(timeout_s))
    r.raise_for_status()
    data = r.json()

    games = (((data or {}).get("scoreboard") or {}).get("games")) or []
    out: List[ScoreboardGame] = []

    for g in games:
        gid = _safe_str(g.get("gameId"))
        if not gid:
            continue

        away_team = (g.get("awayTeam") or {}) if isinstance(g.get("awayTeam"), dict) else {}
        home_team = (g.get("homeTeam") or {}) if isinstance(g.get("homeTeam"), dict) else {}

        away = _team_name(away_team)
        home = _team_name(home_team)

        status = _safe_str(g.get("gameStatusText") or g.get("gameStatus"))
        if not status:
            status = ""

        period = None
        clock = None
        try:
            period = int(g.get("period")) if g.get("period") is not None else None
        except Exception:
            period = None

        c = g.get("gameClock")
        if isinstance(c, str) and c.strip():
            clock = c.strip()

        away_score = None
        home_score = None
        try:
            away_score = int(away_team.get("score")) if away_team.get("score") is not None else None
        except Exception:
            away_score = None

        try:
            home_score = int(home_team.get("score")) if home_team.get("score") is not None else None
        except Exception:
            home_score = None

        out.append(
            ScoreboardGame(
                game_id=gid,
                away=away,
                home=home,
                status_text=status,
                period=period,
                clock=clock,
                away_score=away_score,
                home_score=home_score,
            )
        )

    return out


def format_game_label(g: ScoreboardGame) -> str:
    bits = []

    # Score (if live)
    if g.away_score is not None and g.home_score is not None and (g.away_score + g.home_score) > 0:
        bits.append(f"{g.home_score}-{g.away_score}")

    # Status / clock
    status = (g.status_text or "").strip()
    if g.period is not None and g.clock:
        bits.append(f"Q{g.period} {g.clock}")
    elif g.period is not None:
        bits.append(f"Q{g.period}")
    elif status:
        bits.append(status)

    tail = " · ".join([b for b in bits if b])
    tail = ("— " + tail) if tail else ""

    return f"{g.away} @ {g.home} {tail}  ({g.game_id})"
