from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.betting import (
    prob_moneyline_win_from_mean_sd,
    prob_over_under_from_mean_sd,
    prob_spread_cover_from_mean_sd,
)


@dataclass(frozen=True)
class BetSpec:
    bet_type: str
    side: str
    line: Optional[float]
    odds: Optional[int]


def _derived_from_pred(pred: Dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    derived = pred.get("_derived", {}) or {}
    sd_total = derived.get("sd_final_total")
    sd_margin = derived.get("sd_final_margin")
    try:
        return (float(sd_total) if sd_total is not None else None, float(sd_margin) if sd_margin is not None else None)
    except Exception:
        return (None, None)


def _final_mus_from_pred(pred: Dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Extract final means from a prediction snapshot.

    Preference order (best -> fallback):
    1) pred['_derived']['mu_final_*'] if app computed live-conditioned means
    2) pred['pred']['pred_final_home/away'] (runtime means)
    3) midpoint of pred['bands80'][final_*]
    """

    derived = pred.get("_derived", {}) or {}
    try:
        mu_total = derived.get("mu_final_total")
        mu_margin = derived.get("mu_final_margin")
        mu_home = derived.get("mu_final_home")
        mu_away = derived.get("mu_final_away")
        if all(v is not None for v in (mu_total, mu_margin, mu_home, mu_away)):
            return (float(mu_total), float(mu_margin), float(mu_home), float(mu_away))
    except Exception:
        pass

    p = pred.get("pred", {}) or {}
    try:
        h = p.get("pred_final_home")
        a = p.get("pred_final_away")
        if h is not None and a is not None:
            h = float(h)
            a = float(a)
            return (h + a, h - a, h, a)
    except Exception:
        pass

    bands = pred.get("bands80", {}) or {}

    def mid(key: str) -> Optional[float]:
        lohi = bands.get(key)
        if not lohi or len(lohi) != 2:
            return None
        lo, hi = lohi
        if lo is None or hi is None:
            return None
        return (float(lo) + float(hi)) / 2.0

    return (mid("final_total"), mid("final_margin"), mid("final_home"), mid("final_away"))


def prob_hit_for_bet(
    *,
    pred: Dict[str, Any],
    bet_type: str,
    side: str,
    line: Optional[float],
    home_name: str,
    away_name: str,
) -> Optional[float]:
    """Compute P(hit) for a given bet spec using a prediction snapshot.

    Assumptions:
    - Uses Normal-based probabilities (same as V1).
    - Uses snapshot-derived SDs (clock shrink may already be applied in `pred['_derived']`).

    Returns None if required fields are missing.
    """

    sd_total, sd_margin = _derived_from_pred(pred)
    mu_total, mu_margin, mu_home, mu_away = _final_mus_from_pred(pred)

    bt = (bet_type or "").strip().lower()
    s = (side or "").strip().lower()

    if bt == "total":
        if mu_total is None or sd_total is None or line is None:
            return None
        p_over = prob_over_under_from_mean_sd(mu_total, sd_total, float(line))
        if s.startswith("over"):
            return p_over
        if s.startswith("under"):
            return 1.0 - p_over
        return None

    if bt == "spread":
        if mu_margin is None or sd_margin is None or line is None:
            return None
        # Normalise to "home spread" convention.
        # If user specifies an away line (e.g. Away +12.5), the equivalent home line is -12.5.
        if home_name.lower() in s:
            spread_home = float(line)
            return prob_spread_cover_from_mean_sd(mu_margin, sd_margin, spread_home)

        if away_name.lower() in s:
            spread_home = -float(line)
            p_home_covers = prob_spread_cover_from_mean_sd(mu_margin, sd_margin, spread_home)
            return 1.0 - p_home_covers

        # Fallback: if side includes '+'/'-' only, assume it's home.
        return prob_spread_cover_from_mean_sd(mu_margin, sd_margin, float(line))

    if bt == "moneyline":
        if mu_margin is None or sd_margin is None:
            return None
        p_home_win = prob_moneyline_win_from_mean_sd(mu_margin, sd_margin)
        if home_name.lower() in s:
            return p_home_win
        if away_name.lower() in s:
            return 1.0 - p_home_win
        return None

    if bt == "team total":
        # sd_team is approximate: half of total SD.
        if sd_total is None or line is None:
            return None
        sd_team = max(0.01, float(sd_total) / 2.0)

        if home_name.lower() in s:
            if mu_home is None:
                return None
            p_over = prob_over_under_from_mean_sd(mu_home, sd_team, float(line))
        elif away_name.lower() in s:
            if mu_away is None:
                return None
            p_over = prob_over_under_from_mean_sd(mu_away, sd_team, float(line))
        else:
            return None

        if "over" in s:
            return p_over
        if "under" in s:
            return 1.0 - p_over
        return None

    return None
