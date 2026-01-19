from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.domain.bet_policy import BetPolicy, apply_policy

from src.betting import (
    breakeven_prob_from_american,
    kelly_fraction,
    prob_moneyline_win_from_mean_sd,
    prob_over_under_from_mean_sd,
    prob_spread_cover_from_mean_sd,
)


@dataclass(frozen=True)
class MarketInputs:
    total_line: float
    odds_over: int
    odds_under: int

    spread_home: float
    odds_home: int
    odds_away: int

    moneyline_home: Optional[int] = None
    moneyline_away: Optional[int] = None

    team_total_home: float = 0.0
    team_total_away: float = 0.0
    odds_team_over_home: Optional[int] = None
    odds_team_under_home: Optional[int] = None
    odds_team_over_away: Optional[int] = None
    odds_team_under_away: Optional[int] = None

    bankroll: float = 1000.0
    kelly_mult: float = 0.5


def _add_rec(
    recs: List[Dict[str, Any]],
    *,
    bet_type: str,
    side: str,
    line: Optional[float],
    odds: int,
    p: float,
    kelly_mult: float,
) -> None:
    be = breakeven_prob_from_american(odds)
    recs.append(
        {
            "type": bet_type,
            "side": side,
            "line": line,
            "odds": odds,
            "p": float(p),
            "breakeven": float(be),
            "edge": float(p - be),
            "kelly": float(kelly_fraction(p, odds) * float(kelly_mult)),
        }
    )


def evaluate_markets(
    *,
    pred: Dict[str, Any],
    home_name: str,
    away_name: str,
    final_total_mu: Optional[float],
    final_margin_mu: Optional[float],
    final_home_mu: Optional[float],
    final_away_mu: Optional[float],
    sd_total: float,
    sd_margin: float,
    sd_team: Optional[float],
    inputs: MarketInputs,
    policy: Optional[BetPolicy] = None,
) -> List[Dict[str, Any]]:
    """Return a ranked list of bet recommendations.

    This is deliberately UI-agnostic.
    """
    recs: List[Dict[str, Any]] = []

    # Game total
    if final_total_mu is not None and float(inputs.total_line) > 0:
        p_over = prob_over_under_from_mean_sd(final_total_mu, sd_total, float(inputs.total_line))
        _add_rec(
            recs,
            bet_type="Total",
            side=f"Over {float(inputs.total_line):.1f}",
            line=float(inputs.total_line),
            odds=int(inputs.odds_over),
            p=p_over,
            kelly_mult=float(inputs.kelly_mult),
        )
        _add_rec(
            recs,
            bet_type="Total",
            side=f"Under {float(inputs.total_line):.1f}",
            line=float(inputs.total_line),
            odds=int(inputs.odds_under),
            p=(1.0 - p_over),
            kelly_mult=float(inputs.kelly_mult),
        )

    # Spread
    if final_margin_mu is not None and float(inputs.spread_home) != 0.0:
        p_home_cover = prob_spread_cover_from_mean_sd(final_margin_mu, sd_margin, float(inputs.spread_home))
        _add_rec(
            recs,
            bet_type="Spread",
            side=f"{home_name} {float(inputs.spread_home):+.1f}",
            line=float(inputs.spread_home),
            odds=int(inputs.odds_home),
            p=p_home_cover,
            kelly_mult=float(inputs.kelly_mult),
        )
        _add_rec(
            recs,
            bet_type="Spread",
            side=f"{away_name} {-float(inputs.spread_home):+.1f}",
            line=-float(inputs.spread_home),
            odds=int(inputs.odds_away),
            p=(1.0 - p_home_cover),
            kelly_mult=float(inputs.kelly_mult),
        )

    # Moneyline (derived from margin distribution)
    if final_margin_mu is not None and inputs.moneyline_home is not None and inputs.moneyline_away is not None:
        p_home_win = prob_moneyline_win_from_mean_sd(final_margin_mu, sd_margin)
        _add_rec(
            recs,
            bet_type="Moneyline",
            side=f"{home_name} ML",
            line=None,
            odds=int(inputs.moneyline_home),
            p=p_home_win,
            kelly_mult=float(inputs.kelly_mult),
        )
        _add_rec(
            recs,
            bet_type="Moneyline",
            side=f"{away_name} ML",
            line=None,
            odds=int(inputs.moneyline_away),
            p=(1.0 - p_home_win),
            kelly_mult=float(inputs.kelly_mult),
        )

    # Team totals
    # Important: don't underestimate uncertainty.
    # If total ~ N(mu_T, sd_T) and margin ~ N(mu_M, sd_M) and we assume independence,
    # then home = (T + M)/2 => Var(home) = (Var(T)+Var(M))/4.
    # Away is analogous.
    if sd_team is None:
        sd_team = max(0.01, ((float(sd_total) ** 2 + float(sd_margin) ** 2) ** 0.5) / 2.0)

    if final_home_mu is not None and inputs.team_total_home and inputs.team_total_home > 0:
        if inputs.odds_team_over_home is not None and inputs.odds_team_under_home is not None:
            p_over = prob_over_under_from_mean_sd(final_home_mu, float(sd_team), float(inputs.team_total_home))
            _add_rec(
                recs,
                bet_type="Team total",
                side=f"{home_name} Over {float(inputs.team_total_home):.1f}",
                line=float(inputs.team_total_home),
                odds=int(inputs.odds_team_over_home),
                p=p_over,
                kelly_mult=float(inputs.kelly_mult),
            )
            _add_rec(
                recs,
                bet_type="Team total",
                side=f"{home_name} Under {float(inputs.team_total_home):.1f}",
                line=float(inputs.team_total_home),
                odds=int(inputs.odds_team_under_home),
                p=(1.0 - p_over),
                kelly_mult=float(inputs.kelly_mult),
            )

    if final_away_mu is not None and inputs.team_total_away and inputs.team_total_away > 0:
        if inputs.odds_team_over_away is not None and inputs.odds_team_under_away is not None:
            p_over = prob_over_under_from_mean_sd(final_away_mu, float(sd_team), float(inputs.team_total_away))
            _add_rec(
                recs,
                bet_type="Team total",
                side=f"{away_name} Over {float(inputs.team_total_away):.1f}",
                line=float(inputs.team_total_away),
                odds=int(inputs.odds_team_over_away),
                p=p_over,
                kelly_mult=float(inputs.kelly_mult),
            )
            _add_rec(
                recs,
                bet_type="Team total",
                side=f"{away_name} Under {float(inputs.team_total_away):.1f}",
                line=float(inputs.team_total_away),
                odds=int(inputs.odds_team_under_away),
                p=(1.0 - p_over),
                kelly_mult=float(inputs.kelly_mult),
            )

    recs.sort(key=lambda r: r["edge"], reverse=True)

    if policy is not None:
        return apply_policy(recs, policy)

    return recs
