from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.domain.bet_policy import BetPolicy, apply_policy

from src.betting import (
    american_to_decimal,
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


def _ev_profit(*, p: float, stake: float, odds: int) -> float:
    """Expected profit (not return) given win prob and American odds."""
    D = american_to_decimal(int(odds))
    profit_if_win = float(stake) * (D - 1.0)
    return float(p) * profit_if_win - (1.0 - float(p)) * float(stake)


def _volatility_from_band(*, mu: Optional[float], lo: Optional[float], hi: Optional[float]) -> float:
    """Volatility proxy used to shrink Kelly.

    Defined in Phase 2 plan as:
      (upper80 - lower80) / prediction_mean

    We clamp denominator to avoid blowing up near 0.
    """

    if mu is None or lo is None or hi is None:
        return 0.0

    width = float(hi) - float(lo)
    denom = max(1.0, abs(float(mu)))
    return max(0.0, width / denom)


def _add_rec(
    recs: List[Dict[str, Any]],
    *,
    bet_type: str,
    side: str,
    line: Optional[float],
    odds: int,
    p: float,
    kelly_mult: float,
    volatility: float | None = None,
) -> None:
    be = breakeven_prob_from_american(odds)
    ev100 = _ev_profit(p=float(p), stake=100.0, odds=int(odds))
    # Volatility-aware Kelly shrink (Enhancements.txt)
    # If volatility is high, we scale down aggressiveness.
    vol = max(0.0, float(volatility)) if volatility is not None else 0.0
    kelly_raw = float(kelly_fraction(p, odds)) * float(kelly_mult)
    kelly_adj = kelly_raw / (1.0 + vol) if vol > 0 else kelly_raw

    recs.append(
        {
            "type": bet_type,
            "side": side,
            "line": line,
            "odds": odds,
            "p": float(p),
            "breakeven": float(be),
            "edge": float(p - be),
            "ev_per_100": float(ev100),
            "kelly": float(kelly_adj),
            "kelly_raw": float(kelly_raw),
            "volatility": float(vol),
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
    volatility: float | None = None,
) -> List[Dict[str, Any]]:
    """Return a ranked list of bet recommendations.

    This is deliberately UI-agnostic.
    """
    recs: List[Dict[str, Any]] = []

    bands = pred.get("bands80", {}) or {}

    # Volatility proxy per target (used for Kelly shrink)
    total_vol = float(volatility) if volatility is not None else _volatility_from_band(
        mu=final_total_mu,
        lo=(bands.get("final_total") or [None, None])[0],
        hi=(bands.get("final_total") or [None, None])[1],
    )
    margin_vol = float(volatility) if volatility is not None else _volatility_from_band(
        mu=final_margin_mu,
        lo=(bands.get("final_margin") or [None, None])[0],
        hi=(bands.get("final_margin") or [None, None])[1],
    )
    home_vol = float(volatility) if volatility is not None else _volatility_from_band(
        mu=final_home_mu,
        lo=(bands.get("final_home") or [None, None])[0],
        hi=(bands.get("final_home") or [None, None])[1],
    )
    away_vol = float(volatility) if volatility is not None else _volatility_from_band(
        mu=final_away_mu,
        lo=(bands.get("final_away") or [None, None])[0],
        hi=(bands.get("final_away") or [None, None])[1],
    )

    # Fallback: if team bands absent, reuse total/margin volatility.
    if home_vol <= 0:
        home_vol = max(total_vol, margin_vol)
    if away_vol <= 0:
        away_vol = max(total_vol, margin_vol)

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
            volatility=total_vol,
        )
        _add_rec(
            recs,
            bet_type="Total",
            side=f"Under {float(inputs.total_line):.1f}",
            line=float(inputs.total_line),
            odds=int(inputs.odds_under),
            p=(1.0 - p_over),
            kelly_mult=float(inputs.kelly_mult),
            volatility=total_vol,
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
            volatility=margin_vol,
        )
        _add_rec(
            recs,
            bet_type="Spread",
            side=f"{away_name} {-float(inputs.spread_home):+.1f}",
            line=-float(inputs.spread_home),
            odds=int(inputs.odds_away),
            p=(1.0 - p_home_cover),
            kelly_mult=float(inputs.kelly_mult),
            volatility=margin_vol,
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
            volatility=margin_vol,
        )
        _add_rec(
            recs,
            bet_type="Moneyline",
            side=f"{away_name} ML",
            line=None,
            odds=int(inputs.moneyline_away),
            p=(1.0 - p_home_win),
            kelly_mult=float(inputs.kelly_mult),
            volatility=margin_vol,
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
                volatility=home_vol,
            )
            _add_rec(
                recs,
                bet_type="Team total",
                side=f"{home_name} Under {float(inputs.team_total_home):.1f}",
                line=float(inputs.team_total_home),
                odds=int(inputs.odds_team_under_home),
                p=(1.0 - p_over),
                kelly_mult=float(inputs.kelly_mult),
                volatility=home_vol,
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
                volatility=away_vol,
            )
            _add_rec(
                recs,
                bet_type="Team total",
                side=f"{away_name} Under {float(inputs.team_total_away):.1f}",
                line=float(inputs.team_total_away),
                odds=int(inputs.odds_team_under_away),
                p=(1.0 - p_over),
                kelly_mult=float(inputs.kelly_mult),
                volatility=away_vol,
            )

    recs.sort(key=lambda r: r["edge"], reverse=True)

    if policy is not None:
        return apply_policy(recs, policy)

    return recs
