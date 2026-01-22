from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4"


@dataclass(frozen=True)
class OddsAPIMarketSnapshot:
    # Main markets (full game)
    total_points: Optional[float]
    total_over_odds: Optional[int]
    total_under_odds: Optional[int]

    spread_home: Optional[float]  # sportsbook convention: home line (e.g. -3.5)
    spread_home_odds: Optional[int]
    spread_away_odds: Optional[int]

    moneyline_home: Optional[int]
    moneyline_away: Optional[int]

    # Team totals (if supported by book/plan)
    team_total_home: Optional[float]
    team_total_home_over_odds: Optional[int]
    team_total_home_under_odds: Optional[int]

    team_total_away: Optional[float]
    team_total_away_over_odds: Optional[int]
    team_total_away_under_odds: Optional[int]

    bookmaker: Optional[str] = None
    last_update: Optional[str] = None


class OddsAPIError(RuntimeError):
    pass


def get_api_key() -> str:
    # Streamlit Cloud: use secrets.
    # Locally: allow env var.
    key = os.getenv("ODDS_API_KEY")
    if key:
        return key

    # Avoid importing streamlit at module import time (cloud safety)
    try:
        import streamlit as st  # type: ignore

        if "ODDS_API_KEY" in st.secrets:
            return str(st.secrets["ODDS_API_KEY"]).strip()
    except Exception:
        pass

    raise OddsAPIError(
        "Missing ODDS_API_KEY. Add it to Streamlit Secrets (ODDS_API_KEY) or set env var ODDS_API_KEY."
    )


def _american_from_price(price: Any) -> Optional[int]:
    if price is None:
        return None
    try:
        return int(price)
    except Exception:
        return None


def fetch_nba_odds_snapshot(
    *,
    home_name: str,
    away_name: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals,team_totals",
    odds_format: str = "american",
    date_format: str = "iso",
    preferred_book: Optional[str] = None,
    timeout_s: int = 10,
) -> OddsAPIMarketSnapshot:
    """Fetch a *single* consolidated odds snapshot for an NBA matchup.

    We deliberately keep this narrow:
    - One endpoint call.
    - We pick ONE bookmaker (either preferred_book or the first available) to avoid mixing books.

    Team totals:
    - If available, we parse market key `team_totals` where each outcome usually has:
      - name: Over/Under
      - description: team name
      - point: team total line
    """

    key = get_api_key()

    url = f"{ODDS_API_BASE}/sports/basketball_nba/odds"
    params = {
        "apiKey": key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    def _do_request(p: Dict[str, Any]) -> requests.Response:
        return requests.get(url, params=p, timeout=timeout_s)

    r = _do_request(params)

    if r.status_code != 200:
        # Fail-soft: if team_totals market isn't supported on this endpoint/plan,
        # retry once without it so we can still autofill totals/spreads/moneylines.
        try:
            err = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        except Exception:
            err = {}

        msg = str(err.get("message") or r.text or "")
        code = str(err.get("error_code") or "")

        if r.status_code == 422 and code == "INVALID_MARKET" and "team_totals" in msg:
            params_no_tt = dict(params)
            params_no_tt["markets"] = ",".join(
                [m for m in str(params.get("markets") or "").split(",") if m.strip() and m.strip() != "team_totals"]
            )
            r = _do_request(params_no_tt)

        if r.status_code != 200:
            raise OddsAPIError(f"Odds API error: HTTP {r.status_code}: {r.text[:300]}")

    events = r.json()
    if not isinstance(events, list):
        raise OddsAPIError("Odds API response not a list")

    # Match by team names (case-insensitive)
    hn = home_name.strip().lower()
    an = away_name.strip().lower()

    match: Optional[Dict[str, Any]] = None
    for ev in events:
        try:
            h = str(ev.get("home_team", "")).strip().lower()
            a = str(ev.get("away_team", "")).strip().lower()
            if h == hn and a == an:
                match = ev
                break
        except Exception:
            continue

    if match is None:
        # Fallback: sometimes Odds API flips labels; handle either orientation
        for ev in events:
            try:
                h = str(ev.get("home_team", "")).strip().lower()
                a = str(ev.get("away_team", "")).strip().lower()
                if h == an and a == hn:
                    match = ev
                    # swap later when parsing
                    break
            except Exception:
                continue

    if match is None:
        raise OddsAPIError(f"No odds match found for {away_name} @ {home_name}")

    home_team_api = str(match.get("home_team", ""))
    away_team_api = str(match.get("away_team", ""))
    swapped = home_team_api.strip().lower() != hn

    bookmakers = match.get("bookmakers") or []
    if not bookmakers:
        raise OddsAPIError("No bookmakers in odds response")

    chosen = None
    if preferred_book:
        for b in bookmakers:
            if str(b.get("key", "")).strip().lower() == preferred_book.strip().lower():
                chosen = b
                break

    if chosen is None:
        chosen = bookmakers[0]

    book_key = str(chosen.get("key") or "") or None
    last_update = str(chosen.get("last_update") or "") or None

    total_points = None
    total_over_odds = None
    total_under_odds = None

    spread_home = None
    spread_home_odds = None
    spread_away_odds = None

    ml_home = None
    ml_away = None

    team_total_home = None
    team_total_home_over_odds = None
    team_total_home_under_odds = None

    team_total_away = None
    team_total_away_over_odds = None
    team_total_away_under_odds = None

    for m in chosen.get("markets") or []:
        mk = str(m.get("key") or "")

        if mk == "totals":
            # outcomes: Over/Under with point
            for o in m.get("outcomes") or []:
                name = str(o.get("name") or "")
                point = o.get("point")
                price = _american_from_price(o.get("price"))
                if point is not None:
                    total_points = float(point)
                if name.lower() == "over":
                    total_over_odds = price
                elif name.lower() == "under":
                    total_under_odds = price

        elif mk == "spreads":
            # outcomes by team name with point (spread)
            for o in m.get("outcomes") or []:
                name = str(o.get("name") or "")
                point = o.get("point")
                price = _american_from_price(o.get("price"))
                if point is None:
                    continue

                if swapped:
                    # API home/away is reversed vs our home_name/away_name
                    if name.strip().lower() == away_name.strip().lower():
                        # our home team
                        spread_home = float(point)
                        spread_home_odds = price
                    elif name.strip().lower() == home_name.strip().lower():
                        spread_away_odds = price
                else:
                    if name.strip().lower() == home_name.strip().lower():
                        spread_home = float(point)
                        spread_home_odds = price
                    elif name.strip().lower() == away_name.strip().lower():
                        spread_away_odds = price

        elif mk == "team_totals":
            # outcomes: Over/Under, but team is in `description`
            for o in m.get("outcomes") or []:
                side = str(o.get("name") or "").strip().lower()  # over/under
                team = str(o.get("description") or "").strip()
                point = o.get("point")
                price = _american_from_price(o.get("price"))
                if point is None or not team:
                    continue

                def is_home_team(t: str) -> bool:
                    return t.strip().lower() == home_name.strip().lower()

                def is_away_team(t: str) -> bool:
                    return t.strip().lower() == away_name.strip().lower()

                # If API home/away swapped relative to our names, that doesn't matter here,
                # because we're matching by actual team names.
                if is_home_team(team):
                    team_total_home = float(point)
                    if side == "over":
                        team_total_home_over_odds = price
                    elif side == "under":
                        team_total_home_under_odds = price

                elif is_away_team(team):
                    team_total_away = float(point)
                    if side == "over":
                        team_total_away_over_odds = price
                    elif side == "under":
                        team_total_away_under_odds = price

        elif mk == "h2h":
            for o in m.get("outcomes") or []:
                name = str(o.get("name") or "")
                price = _american_from_price(o.get("price"))
                if swapped:
                    if name.strip().lower() == away_name.strip().lower():
                        ml_home = price
                    elif name.strip().lower() == home_name.strip().lower():
                        ml_away = price
                else:
                    if name.strip().lower() == home_name.strip().lower():
                        ml_home = price
                    elif name.strip().lower() == away_name.strip().lower():
                        ml_away = price

    return OddsAPIMarketSnapshot(
        total_points=total_points,
        total_over_odds=total_over_odds,
        total_under_odds=total_under_odds,
        spread_home=spread_home,
        spread_home_odds=spread_home_odds,
        spread_away_odds=spread_away_odds,
        moneyline_home=ml_home,
        moneyline_away=ml_away,
        team_total_home=team_total_home,
        team_total_home_over_odds=team_total_home_over_odds,
        team_total_home_under_odds=team_total_home_under_odds,
        team_total_away=team_total_away,
        team_total_away_over_odds=team_total_away_over_odds,
        team_total_away_under_odds=team_total_away_under_odds,
        bookmaker=book_key,
        last_update=last_update,
    )
