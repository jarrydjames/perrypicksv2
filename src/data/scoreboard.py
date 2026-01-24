from __future__ import annotations

"""Game picker backend for Streamlit.

Goal:
- let user pick a date
- show games for that date
- include live status (Q/clock/score) when available

Why not use nba.com "todaysScoreboard"?
- It frequently returns 403 on Streamlit Cloud.
Why not use data.nba.net?
- SSL/cert hostname issues in some environments.

So we use a reliable combo:
1) scheduleLeagueV2.json (public, works from cloud)
2) per-game live boxscore_{gameId}.json for status + score

This module stays streamlit-free (so it can be reused in scripts/tests).
"""

import datetime as dt
from dataclasses import dataclass
from typing import Any, List, Optional

import requests


SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"


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


def _get_json(url: str, *, timeout_s: int) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.nba.com/",
    }
    r = requests.get(url, headers=headers, timeout=int(timeout_s))
    r.raise_for_status()
    return r.json()


def _extract_team_tricode(team: dict, fallback: str) -> str:
    for k in ("teamTricode", "triCode", "abbreviation"):
        v = team.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
    return fallback


def _fetch_live_status(gid: str, *, timeout_s: int) -> tuple[str, Optional[int], Optional[str], Optional[int], Optional[int]]:
    """Return (status_text, period, clock, home_score, away_score)."""

    try:
        data = _get_json(BOX_URL.format(gid=gid), timeout_s=timeout_s)
        game = (data or {}).get("game") or {}

        status_text = _safe_str(game.get("gameStatusText") or game.get("gameStatus") or "")
        period = None
        try:
            period = int(game.get("period")) if game.get("period") is not None else None
        except Exception:
            period = None

        clock = None
        c = game.get("gameClock")
        if isinstance(c, str) and c.strip():
            clock = c.strip()

        home = game.get("homeTeam") or {}
        away = game.get("awayTeam") or {}
        hs = None
        a_s = None
        try:
            hs = int(((home.get("statistics") or {}).get("points")) or 0)
        except Exception:
            hs = None
        try:
            a_s = int(((away.get("statistics") or {}).get("points")) or 0)
        except Exception:
            a_s = None

        return status_text, period, clock, hs, a_s
    except Exception:
        return "", None, None, None, None


def fetch_scoreboard(date: dt.date, *, timeout_s: int = 10, include_live: bool = True) -> List[ScoreboardGame]:
    """Fetch games for a given date.

    Uses scheduleLeagueV2.json for IDs + teams.
    Optionally enriches with live status/score by calling boxscore_{gid}.json.
    """

    # scheduleLeagueV2 uses "MM/DD/YYYY 00:00:00" strings
    sched_key = date.strftime("%m/%d/%Y")
    payload = _get_json(SCHEDULE_URL, timeout_s=timeout_s)

    league = (payload or {}).get("leagueSchedule") or {}
    game_dates = league.get("gameDates") or []

    games: list[dict] = []
    for gd in game_dates:
        gd_str = _safe_str(gd.get("gameDate"))
        if gd_str.startswith(sched_key):
            games = gd.get("games") or []
            break

    out: List[ScoreboardGame] = []

    for g in games:
        gid = _safe_str(g.get("gameId"))
        if not gid:
            continue

        away_team = g.get("awayTeam") or {}
        home_team = g.get("homeTeam") or {}
        away = _extract_team_tricode(away_team, "AWAY")
        home = _extract_team_tricode(home_team, "HOME")

        status_text = _safe_str(g.get("gameStatusText") or "")
        period = None
        clock = None
        away_score = None
        home_score = None

        if include_live:
            status_text, period, clock, home_score, away_score = _fetch_live_status(gid, timeout_s=min(8, timeout_s))

        out.append(
            ScoreboardGame(
                game_id=gid,
                away=away,
                home=home,
                status_text=status_text,
                period=period,
                clock=clock,
                away_score=away_score,
                home_score=home_score,
            )
        )

    return out


def format_game_label(g: ScoreboardGame) -> str:
    bits = []

    if g.away_score is not None and g.home_score is not None and (g.away_score + g.home_score) > 0:
        bits.append(f"{g.home_score}-{g.away_score}")

    if g.period is not None and g.clock:
        bits.append(f"Q{g.period} {g.clock}")
    elif g.period is not None:
        bits.append(f"Q{g.period}")
    elif g.status_text:
        bits.append(str(g.status_text).strip())

    tail = " · ".join([b for b in bits if b])
    tail = ("— " + tail) if tail else ""

    return f"{g.away} @ {g.home} {tail}  ({g.game_id})"
