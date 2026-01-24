"""ROI simulation helpers for backtesting.

Implements the Enhancements.txt suggestion:
- if edge > threshold -> place bet
- compute P/L under American odds
- track ROI, hit rate, drawdown

Notes:
- We do NOT have historic bookmaker prices in the training data yet.
  So by default we simulate at -110 (vig baked in) unless odds are provided.
- This module is intentionally small + dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def american_to_decimal(odds: int) -> float:
    odds = int(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def breakeven_prob(odds: int) -> float:
    odds = int(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def kelly_fraction(p: float, odds: int) -> float:
    p = float(p)
    d = american_to_decimal(int(odds))
    b = d - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - (1.0 - p)) / b
    return float(max(0.0, f))


@dataclass(frozen=True)
class SimConfig:
    stake: float = 100.0
    edge_threshold: float = 0.06
    odds: int = -110


@dataclass
class SimState:
    bankroll: float
    peak: float
    max_drawdown: float
    total_staked: float
    n_bets: int
    n_wins: int
    n_losses: int
    n_push: int


def _settle_bet(*, outcome: float, line: float, is_over: bool, odds: int, stake: float) -> float:
    """Return profit (not return) for a 1-unit bet.

    outcome: realized total/margin (same space as line)
    is_over: True means bet Over, False means Under

    Push returns 0.
    """

    outcome = float(outcome)
    line = float(line)

    if outcome == line:
        return 0.0

    win = outcome > line if is_over else outcome < line
    if win:
        return float(stake) * (american_to_decimal(int(odds)) - 1.0)
    return -float(stake)


def simulate_threshold_strategy(
    *,
    p: np.ndarray,
    y: np.ndarray,
    line: np.ndarray,
    cfg: SimConfig,
    bet_over: bool,
) -> SimState:
    """Simulate betting each row where edge exceeds threshold.

    p: model probability for the bet side
    y: realized outcome
    line: market line

    bet_over: if True, bet Over; else bet Under.

    Returns bankroll statistics starting from 0 (profit units).
    """

    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    line = np.asarray(line, dtype=float)

    be = float(breakeven_prob(cfg.odds))

    state = SimState(
        bankroll=0.0,
        peak=0.0,
        max_drawdown=0.0,
        total_staked=0.0,
        n_bets=0,
        n_wins=0,
        n_losses=0,
        n_push=0,
    )

    for pi, yi, li in zip(p, y, line):
        if not np.isfinite(pi) or not np.isfinite(yi) or not np.isfinite(li):
            continue

        edge = float(pi) - be
        if edge <= float(cfg.edge_threshold):
            continue

        profit = _settle_bet(outcome=float(yi), line=float(li), is_over=bet_over, odds=cfg.odds, stake=cfg.stake)

        state.n_bets += 1
        state.total_staked += float(cfg.stake)

        if profit > 0:
            state.n_wins += 1
        elif profit < 0:
            state.n_losses += 1
        else:
            state.n_push += 1

        state.bankroll += profit
        state.peak = max(state.peak, state.bankroll)
        dd = state.peak - state.bankroll
        state.max_drawdown = max(state.max_drawdown, dd)

    return state


def roi(state: SimState) -> float:
    if state.total_staked <= 0:
        return 0.0
    return float(state.bankroll) / float(state.total_staked)
