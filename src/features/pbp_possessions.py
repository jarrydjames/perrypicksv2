from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class TeamHalfStats:
    # Shot counts
    fga: int
    fgm: int
    tpa: int
    tpm: int

    # FT counts
    fta: int
    ftm: int

    # Possession components
    oreb: int
    dreb: int
    tov: int

    # Points
    pts: int

    @property
    def poss(self) -> float:
        # Possession estimate used in the framework document
        return float(self.fga) + 0.44 * float(self.fta) - float(self.oreb) + float(self.tov)

    @property
    def ppp(self) -> float:
        p = max(1.0, self.poss)
        return float(self.pts) / p

    @property
    def efg(self) -> float:
        # eFG% = (FGM + 0.5*3PM) / FGA
        denom = max(1, int(self.fga))
        return (float(self.fgm) + 0.5 * float(self.tpm)) / float(denom)

    @property
    def ftr(self) -> float:
        # FT rate = FTA / FGA
        denom = max(1, int(self.fga))
        return float(self.fta) / float(denom)

    @property
    def tpar(self) -> float:
        # 3PA rate = 3PA / FGA
        denom = max(1, int(self.fga))
        return float(self.tpa) / float(denom)

    @property
    def tor(self) -> float:
        # Turnover rate as turnovers / possessions
        denom = max(1.0, float(self.poss))
        return float(self.tov) / denom

    def orbp_vs(self, opp_dreb: int) -> float:
        denom = max(1, int(self.oreb) + int(opp_dreb))
        return float(self.oreb) / float(denom)


def _as_int(x, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))
    except Exception:
        return default


def _is_made(action: Mapping[str, Any]) -> bool:
    return str(action.get("shotResult") or "").lower() == "made"


def team_stats_from_pbp(
    actions: Iterable[Mapping[str, Any]], *, team_tricode: str, max_period: int
) -> TeamHalfStats:
    """Compute team stats from PBP actions up through max_period.

    This is intentionally simple + robust and used for both:
    - 1H (max_period=2)
    - 2H-to-now (max_period=4, but caller can filter further)

    Counting rules (robust + simple):
    - FGA: actionType in {'2pt','3pt'}
    - FTA: actionType == 'freethrow'
    - TOV: actionType == 'turnover'
    - OREB: actionType == 'rebound' AND subType == 'offensive'
    - PTS: sum of made 2pt/3pt + made freethrows

    Returns a TeamHalfStats with possessions and PPP computed via properties.
    """

    tri = str(team_tricode or "").upper().strip()
    if not tri:
        raise ValueError("team_tricode required")

    fga = fgm = tpa = tpm = 0
    fta = ftm = 0
    oreb = dreb = tov = 0
    pts = 0

    for a in actions:
        try:
            if int(a.get("period", 0)) > int(max_period):
                continue
        except Exception:
            continue

        if str(a.get("teamTricode") or "").upper().strip() != tri:
            continue

        at = str(a.get("actionType") or "").lower().strip()

        if at in {"2pt", "3pt"}:
            fga += 1
            if at == "3pt":
                tpa += 1
            if _is_made(a):
                fgm += 1
                pts += 2 if at == "2pt" else 3
                if at == "3pt":
                    tpm += 1

        elif at == "freethrow":
            fta += 1
            if _is_made(a):
                ftm += 1
                pts += 1

        elif at == "turnover":
            tov += 1

        elif at == "rebound":
            st = str(a.get("subType") or "").lower().strip()
            if st == "offensive":
                oreb += 1
            elif st == "defensive":
                dreb += 1

    return TeamHalfStats(
        fga=fga,
        fgm=fgm,
        tpa=tpa,
        tpm=tpm,
        fta=fta,
        ftm=ftm,
        oreb=oreb,
        dreb=dreb,
        tov=tov,
        pts=pts,
    )


def team_stats_from_pbp_first_half(actions: Iterable[Mapping[str, Any]], *, team_tricode: str) -> TeamHalfStats:
    """Backwards-compatible wrapper (1H only)."""

    return team_stats_from_pbp(actions, team_tricode=team_tricode, max_period=2)


def _filter_actions_by_period(
    actions: Iterable[Mapping[str, Any]], *, min_period: int, max_period: int
) -> Iterable[Mapping[str, Any]]:
    for a in actions:
        try:
            p = int(a.get("period", 0))
        except Exception:
            continue
        if p < int(min_period) or p > int(max_period):
            continue
        yield a


def game_possessions_first_half(
    actions: Iterable[Mapping[str, Any]], *, home_tri: str, away_tri: str
) -> Dict[str, float]:
    """Return 1H possessions/PPP plus leakage-safe rate features.

    We intentionally output keys that match the existing dataset columns
    (`home_efg`, `home_ftr`, ...) but computed from 1H only.
    """

    h = team_stats_from_pbp(actions, team_tricode=home_tri, max_period=2)
    a = team_stats_from_pbp(actions, team_tricode=away_tri, max_period=2)

    game_poss = 0.5 * (h.poss + a.poss)

    return {
        # core
        "home_poss_1h": float(h.poss),
        "away_poss_1h": float(a.poss),
        "game_poss_1h": float(game_poss),
        "home_ppp_1h": float(h.ppp),
        "away_ppp_1h": float(a.ppp),

        # overwrite leaky full-game box-derived rate features with 1H PBP-derived versions
        "home_efg": float(h.efg),
        "home_ftr": float(h.ftr),
        "home_tpar": float(h.tpar),
        "home_tor": float(h.tor),
        "home_orbp": float(h.orbp_vs(a.dreb)),

        "away_efg": float(a.efg),
        "away_ftr": float(a.ftr),
        "away_tpar": float(a.tpar),
        "away_tor": float(a.tor),
        "away_orbp": float(a.orbp_vs(h.dreb)),
    }


def live_2h_pace_from_pbp(
    actions: Iterable[Mapping[str, Any]], *, home_tri: str, away_tri: str
) -> Dict[str, float]:
    """Compute 2H-to-now possessions + PPP from PBP actions.

    This is for *live conditioning* (tracking), not training.

    Returns keys:
    - game_poss_2h
    - home_poss_2h, away_poss_2h
    - home_ppp_2h, away_ppp_2h

    If there are no 2H actions yet, values will be zeros.
    """

    a2 = list(_filter_actions_by_period(actions, min_period=3, max_period=4))
    if not a2:
        return {
            "home_poss_2h": 0.0,
            "away_poss_2h": 0.0,
            "game_poss_2h": 0.0,
            "home_ppp_2h": 0.0,
            "away_ppp_2h": 0.0,
        }

    h = team_stats_from_pbp(a2, team_tricode=home_tri, max_period=4)
    a = team_stats_from_pbp(a2, team_tricode=away_tri, max_period=4)

    game_poss = 0.5 * (h.poss + a.poss)

    return {
        "home_poss_2h": float(h.poss),
        "away_poss_2h": float(a.poss),
        "game_poss_2h": float(game_poss),
        "home_ppp_2h": float(h.ppp),
        "away_ppp_2h": float(a.ppp),
    }
