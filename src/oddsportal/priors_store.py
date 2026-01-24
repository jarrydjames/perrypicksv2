"""OddsPortal priors store.

Single responsibility: load priors keyed by NBA `game_id`.

We keep this small so both training + runtime can share it.

Data source:
  data/oddsportal/priors_by_game_id_ALL.jsonl

Row format (JSONL):
  {
    "game_id": "00224...",
    "total_line": 226.0,
    "home_spread_line": -4.5,
    "home_team_total_line_implied": 110.75,
    "away_team_total_line_implied": 115.25,
    ...
  }

Zen puppy rule: tolerate missing files/rows; return None/NaN-friendly output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PRIORS_PATH = Path("data/oddsportal/priors_by_game_id_ALL.jsonl")


@dataclass(frozen=True)
class MarketPriors:
    total_line: float | None
    home_spread_line: float | None
    home_team_total_line: float | None
    away_team_total_line: float | None


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_priors_by_game_id(path: Path = DEFAULT_PRIORS_PATH) -> dict[str, MarketPriors]:
    """Load priors into memory.

    This is fine for our scale (a few thousand lines). If it grows large later,
    we can switch to sqlite/duckdb. YAGNI.
    """

    if not path.exists():
        return {}

    out: dict[str, MarketPriors] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gid = str(obj.get("game_id") or "").strip()
            if not gid:
                continue

            out[gid] = MarketPriors(
                total_line=_to_float(obj.get("total_line")),
                home_spread_line=_to_float(obj.get("home_spread_line")),
                home_team_total_line=_to_float(obj.get("home_team_total_line_implied")),
                away_team_total_line=_to_float(obj.get("away_team_total_line_implied")),
            )

    return out


def priors_to_features(priors: MarketPriors | None) -> dict[str, float]:
    """Convert to model-friendly feature dict.

    Prefix with `market_` to avoid collisions with user-input betting lines.
    """

    if priors is None:
        return {
            "market_total_line": float("nan"),
            "market_home_spread_line": float("nan"),
            "market_home_team_total_line": float("nan"),
            "market_away_team_total_line": float("nan"),
        }

    def f(x: float | None) -> float:
        return float("nan") if x is None else float(x)

    return {
        "market_total_line": f(priors.total_line),
        "market_home_spread_line": f(priors.home_spread_line),
        "market_home_team_total_line": f(priors.home_team_total_line),
        "market_away_team_total_line": f(priors.away_team_total_line),
    }
