"""OddsPortal market prior extraction helpers.

These functions take OddsHarvester JSON output and extract the market priors
needed by Enhancements.txt:
- pregame total line
- pregame spread line
- (implied) team totals

We intentionally keep this module small + testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PregamePriors:
    total_line: float | None
    home_spread_line: float | None
    home_team_total_line_implied: float | None
    away_team_total_line_implied: float | None


def dec_odds_to_prob(odds: float) -> float:
    # Decimal odds implied probability (no vig normalization)
    if odds <= 0:
        return 0.0
    return 1.0 / odds


_RX_NUM = re.compile(r"([+-]?\d+(?:\.\d+)?)")


def _extract_float(text: str) -> float | None:
    m = _RX_NUM.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _pick_fairest_two_way(rows: list[dict[str, Any]], a_key: str, b_key: str) -> dict[str, Any] | None:
    """Pick the row with odds closest to 50/50.

    This is a pragmatic way to choose the 'main' line when multiple submarkets exist.
    """

    best = None
    best_score = 1e9

    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            a = float(row.get(a_key))
            b = float(row.get(b_key))
        except Exception:
            continue

        score = abs(dec_odds_to_prob(a) - 0.5) + abs(dec_odds_to_prob(b) - 0.5)
        if score < best_score:
            best_score = score
            best = row

    return best


def extract_priors_from_odds_harvester_match(match: dict[str, Any]) -> PregamePriors:
    """Extract priors from a single OddsHarvester match dict.

    Supports two styles of output:
    1) Active mode (specific market): keys like `over_under_games_220_5_market`
    2) Preview mode (passive submarkets): market list entries containing `submarket_name`

    This function is intentionally conservative: it returns None fields if it can't infer.
    """

    total_line: float | None = None
    home_spread_line: float | None = None

    # 1) Preview-mode rows: look for Over/Under and Asian Handicap lists with submarket_name
    for k, v in (match or {}).items():
        if not (isinstance(k, str) and k.endswith("_market") and isinstance(v, list) and v):
            continue

        first = v[0]
        if isinstance(first, dict) and "submarket_name" in first:
            # Over/Under preview rows
            if any(isinstance(r, dict) and r.get("market_type") == "Over/Under" for r in v):
                best = _pick_fairest_two_way(v, "odds_over", "odds_under")
                if best and isinstance(best.get("submarket_name"), str):
                    total_line = _extract_float(best["submarket_name"]) or total_line

            # Asian Handicap preview rows
            if any(isinstance(r, dict) and r.get("market_type") == "Asian Handicap" for r in v):
                best = _pick_fairest_two_way(v, "handicap_team_1", "handicap_team_2")
                if best and isinstance(best.get("submarket_name"), str):
                    home_spread_line = _extract_float(best["submarket_name"]) or home_spread_line

    # 2) Active-mode keys: infer line value from key name
    if total_line is None or home_spread_line is None:
        for k, v in (match or {}).items():
            if not (isinstance(k, str) and k.endswith("_market") and isinstance(v, list) and v):
                continue
            row = v[0] if isinstance(v[0], dict) else None
            if not row:
                continue

            if total_line is None and k.startswith("over_under_games_"):
                # key includes the line number
                try:
                    s = k.replace("over_under_games_", "").replace("_market", "")
                    total_line = float(s.replace("_", "."))
                except Exception:
                    pass

            if home_spread_line is None and k.startswith("asian_handicap_games_"):
                try:
                    s = k.replace("asian_handicap_games_", "")
                    s = s.replace("_games_market", "")
                    s = s.replace("_games", "")
                    home_spread_line = float(s.replace("_", "."))
                except Exception:
                    pass

    # Team totals: OddsHarvester doesn't support them directly; we provide implied fallback.
    home_tt = None
    away_tt = None
    if total_line is not None and home_spread_line is not None:
        implied_margin = -home_spread_line
        home_tt = (total_line + implied_margin) / 2.0
        away_tt = (total_line - implied_margin) / 2.0

    return PregamePriors(
        total_line=total_line,
        home_spread_line=home_spread_line,
        home_team_total_line_implied=home_tt,
        away_team_total_line_implied=away_tt,
    )
