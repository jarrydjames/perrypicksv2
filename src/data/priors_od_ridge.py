from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ODPriors:
    league_ppp: float
    home_adv_ppp: float
    off: Dict[str, float]  # team -> offense strength
    deff: Dict[str, float]  # team -> defense strength


def fit_off_def_ridge(
    games: pd.DataFrame,
    *,
    team_col_home: str = "homeTeam",
    team_col_away: str = "awayTeam",
    pts_home_col: str = "final_home_pts",
    pts_away_col: str = "final_away_pts",
    poss_col: str = "game_poss",
    alpha: float = 25.0,
) -> ODPriors:
    """Fit ridge offense/defense decomposition on full-game PPP.

    Model:
      PPP_home = mu + O_home - D_away + H
      PPP_away = mu + O_away - D_home

    We solve via ridge regression on a stacked system.

    NOTE: This is intended to be used in a walk-forward manner (as-of-date).
    This function is just the core estimator.
    """

    df = games.copy()

    # Clean
    df = df.dropna(subset=[team_col_home, team_col_away, pts_home_col, pts_away_col, poss_col])
    if df.empty:
        raise ValueError("No rows to fit priors")

    teams = sorted(set(df[team_col_home]).union(set(df[team_col_away])))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Parameters: [mu, H, O_0..O_{n-1}, D_0..D_{n-1}]
    p = 2 + 2 * n

    X_rows = []
    y = []

    for _, r in df.iterrows():
        home = r[team_col_home]
        away = r[team_col_away]
        poss = float(r[poss_col])
        if poss <= 0:
            continue

        ppp_home = float(r[pts_home_col]) / poss
        ppp_away = float(r[pts_away_col]) / poss

        # Home equation
        xh = np.zeros(p)
        xh[0] = 1.0  # mu
        xh[1] = 1.0  # H
        xh[2 + idx[home]] = 1.0  # O_home
        xh[2 + n + idx[away]] = -1.0  # -D_away
        X_rows.append(xh)
        y.append(ppp_home)

        # Away equation
        xa = np.zeros(p)
        xa[0] = 1.0  # mu
        xa[1] = 0.0  # H
        xa[2 + idx[away]] = 1.0  # O_away
        xa[2 + n + idx[home]] = -1.0  # -D_home
        X_rows.append(xa)
        y.append(ppp_away)

    X = np.vstack(X_rows)
    yv = np.asarray(y, dtype=float)

    # Ridge closed-form: (X'X + alpha*I)^-1 X'y
    # Don’t regularize mu and H as strongly: set their penalty to 0.
    I = np.eye(p)
    I[0, 0] = 0.0
    I[1, 1] = 0.0

    beta = np.linalg.solve(X.T @ X + float(alpha) * I, X.T @ yv)

    mu = float(beta[0])
    H = float(beta[1])
    O = beta[2 : 2 + n]
    D = beta[2 + n : 2 + 2 * n]

    off = {t: float(O[idx[t]]) for t in teams}
    deff = {t: float(D[idx[t]]) for t in teams}

    return ODPriors(league_ppp=mu, home_adv_ppp=H, off=off, deff=deff)
