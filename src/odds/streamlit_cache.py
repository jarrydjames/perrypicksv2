from __future__ import annotations

from typing import Optional

from src.odds.odds_api import OddsAPIMarketSnapshot, fetch_nba_odds_snapshot


def get_cached_nba_odds(
    *,
    home_name: str,
    away_name: str,
    preferred_book: Optional[str] = None,
    ttl_seconds: int = 120,
) -> OddsAPIMarketSnapshot:
    """Streamlit-cached wrapper around the Odds API call.

    This is intentionally a thin wrapper so the core client stays framework-agnostic.
    """

    import streamlit as st  # local import (cloud-safe)

    @st.cache_data(ttl=ttl_seconds, show_spinner=False)
    def _cached(home: str, away: str, book: Optional[str]) -> OddsAPIMarketSnapshot:
        return fetch_nba_odds_snapshot(home_name=home, away_name=away, preferred_book=book)

    return _cached(home_name, away_name, preferred_book)
