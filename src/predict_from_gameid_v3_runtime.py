from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple
import os

from src.betting import normal_from_q10q90

import joblib
import numpy as np
import pandas as pd

from src.oddsportal.priors_store import load_priors_by_game_id, priors_to_features
from src.predict_from_gameid_v2 import (
    behavior_counts_1h,
    extract_game_id,
    fetch_box,
    fetch_pbp_df,
    first_half_score,
    team_totals_from_box_team,
)
from src.features.pbp_possessions import live_2h_pace_from_pbp


Z80 = 1.2815515655446004

# Module-level cache: we don't want to re-read JSONL every prediction.
_PRIORS_MAP: dict[str, Any] | None = None

# Module-level model cache: joblib load is expensive (and Streamlit reruns a lot).
_MODEL_CACHE: dict[str, "TwoHeadRuntime"] = {}


def _get_priors_map() -> dict[str, Any]:
    global _PRIORS_MAP
    if _PRIORS_MAP is None:
        _PRIORS_MAP = load_priors_by_game_id()
    return _PRIORS_MAP


@dataclass(frozen=True)
class TwoHeadRuntime:
    model_name: str
    model_version: str
    feature_version: str
    features: list[str]

    total_model: Any
    margin_model: Any

    sigma_total: float
    sigma_margin: float

    # Optional quantile models for 80% PI (q10/q90)
    total_q10_model: Any | None = None
    total_q90_model: Any | None = None
    margin_q10_model: Any | None = None
    margin_q90_model: Any | None = None

    # Optional joint model + residual covariance between (total, margin)
    joint_model: Any | None = None
    residual_cov: list[list[float]] | None = None


def _safe_team_name(team_block: dict, fallback: str) -> str:
    for k in ("teamName", "teamCity", "teamTricode", "name"):
        v = team_block.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return fallback


def _extract_status(game: dict) -> dict:
    status: dict[str, Any] = {}
    for k in ("gameStatus", "gameStatusText", "period", "gameClock", "gameTimeUTC"):
        if k in game:
            status[k] = game.get(k)
    if not status:
        st = game.get("status") or {}
        if isinstance(st, dict):
            for k in ("gameStatus", "gameStatusText", "period", "gameClock", "gameTimeUTC"):
                if k in st:
                    status[k] = st.get(k)
    return status


def _poss_1h(team_stats: dict, opp_stats: dict) -> float:
    # Standard approximation
    fga = float(team_stats.get("fga", 0))
    fta = float(team_stats.get("fta", 0))
    to = float(team_stats.get("to", 0))
    oreb = float(team_stats.get("oreb", 0))
    poss = fga + 0.44 * fta + to - oreb
    return float(max(1.0, poss))


def _rate_features(prefix: str, t: dict, opp: dict) -> dict:
    poss = _poss_1h(t, opp)
    efg = (t["fgm"] + 0.5 * t["tpm"]) / max(t["fga"], 1)
    ftr = t["fta"] / max(t["fga"], 1)
    tpar = t["tpa"] / max(t["fga"], 1)
    tor = t["to"] / poss
    orbp = t["oreb"] / max(t["oreb"] + opp["dreb"], 1)
    ppp = t["pts"] / poss

    return {
        f"{prefix}_efg": float(efg),
        f"{prefix}_ftr": float(ftr),
        f"{prefix}_tpar": float(tpar),
        f"{prefix}_tor": float(tor),
        f"{prefix}_orbp": float(orbp),
        f"{prefix}_poss_1h": float(poss),
        f"{prefix}_ppp_1h": float(ppp),
    }


def _load_twohead(path: str) -> TwoHeadRuntime:
    obj = joblib.load(path)
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected model format for {path}")

    total = obj.get("total") or {}
    margin = obj.get("margin") or {}

    total_model = total.get("model")
    margin_model = margin.get("model")

    total_q10 = total.get("q10_model")
    total_q90 = total.get("q90_model")
    margin_q10 = margin.get("q10_model")
    margin_q90 = margin.get("q90_model")

    joint = obj.get("joint") or {}
    joint_model = joint.get("model")
    residual_cov = joint.get("residual_cov")

    if total_model is None or margin_model is None:
        raise ValueError(f"Missing heads in {path}")

    return TwoHeadRuntime(
        model_name=str(obj.get("model_name") or "twohead"),
        model_version=str(obj.get("model_version") or ""),
        feature_version=str(obj.get("feature_version") or "v1"),
        features=list(obj.get("features") or []),
        total_model=total_model,
        margin_model=margin_model,
        sigma_total=float(total.get("residual_sigma") or 12.0),
        sigma_margin=float(margin.get("residual_sigma") or 8.0),
        total_q10_model=total_q10,
        total_q90_model=total_q90,
        margin_q10_model=margin_q10,
        margin_q90_model=margin_q90,
        joint_model=joint_model,
        residual_cov=residual_cov,
    )


def _norm_pi80(mu: float, sigma: float) -> Tuple[float, float]:
    return (float(mu - Z80 * sigma), float(mu + Z80 * sigma))


def _align_features(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Align runtime features to the model's expected feature list.

    Prevents train/serve skew crashes when new features are added.
    Missing columns are filled with NaN.
    """

    out = X.copy()
    for c in features:
        if c not in out.columns:
            out[c] = np.nan
    return out[features]


def _get_model(path: str) -> TwoHeadRuntime:
    cached = _MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    m = _load_twohead(path)
    _MODEL_CACHE[path] = m
    return m


def _model_path(model_name: str) -> str:
    """Map friendly model names to artifact paths.

    Keep this tiny + explicit (Zen puppy hates magic).
    """

    name = str(model_name or "").strip().lower()
    if name in {"gbt", "hgb", "histgb"}:
        return "models_v2/gbt_twohead.joblib"
    if name in {"ridge"}:
        return "models_v2/ridge_twohead.joblib"
    if name in {"rf", "random_forest", "randomforest"}:
        return "models_v2/random_forest_twohead.joblib"

    raise ValueError(f"Unknown model name: {model_name!r} (expected gbt|ridge|random_forest)")


def predict_from_game_id(gid_or_url: str) -> Dict[str, Any]:
    """Production runtime predictor.

    Final model use-cases:
    - margin (spread/ML): Ridge twohead (calibration-first)
    - total (game total): Ridge twohead (best chrono walkforward RMSE)
    - team totals: derived from (total, margin) with conservative variance propagation

    Returns a dict shaped like what app.py expects.
    """

    gid = extract_game_id(gid_or_url)
    game = fetch_box(gid)
    pbp = fetch_pbp_df(gid)

    h1_home, h1_away = first_half_score(game)
    beh = behavior_counts_1h(pbp)

    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}

    home_name = _safe_team_name(home, "Home")
    away_name = _safe_team_name(away, "Away")

    home_tri = str(home.get("teamTricode") or "HOME").upper().strip() or "HOME"
    away_tri = str(away.get("teamTricode") or "AWAY").upper().strip() or "AWAY"

    pace2h = {}
    try:
        pace2h = live_2h_pace_from_pbp(pbp.to_dict("records"), home_tri=home_tri, away_tri=away_tri)
    except Exception:
        pace2h = {}
    status = _extract_status(game)

    ht = team_totals_from_box_team(home)
    at = team_totals_from_box_team(away)

    # Live score (current points) from boxscore team stats.
    # Note: during halftime this should match 1H total; during 2H it updates live.
    live_home_pts = int(ht.get("pts") or 0)
    live_away_pts = int(at.get("pts") or 0)

    row: Dict[str, Any] = {
        "h1_home": float(h1_home),
        "h1_away": float(h1_away),
        "h1_total": float(h1_home + h1_away),
        "h1_margin": float(h1_home - h1_away),
        **{k: float(v) for k, v in beh.items()},
    }

    # Market priors (pregame lines)
    row.update(priors_to_features(_get_priors_map().get(gid)))

    row.update(_rate_features("home", ht, at))
    row.update(_rate_features("away", at, ht))

    # Game possessions = average of team possessions.
    row["game_poss_1h"] = 0.5 * (float(row["home_poss_1h"]) + float(row["away_poss_1h"]))

    X = pd.DataFrame([row])

    # Load final models (configurable via env vars)
    # Chronological walkforward results: Ridge is best for BOTH total + margin.
    total_choice = os.getenv("PERRYPICKS_TOTAL_MODEL", "ridge")
    margin_choice = os.getenv("PERRYPICKS_MARGIN_MODEL", "ridge")

    total_model = _get_model(_model_path(total_choice))
    margin_model = _get_model(_model_path(margin_choice))

    # Feature alignment
    X_total = _align_features(X, total_model.features) if total_model.features else X
    X_margin = _align_features(X, margin_model.features) if margin_model.features else X

    # Pass numpy arrays to avoid sklearn warning: "X has feature names..."
    mu_total_2h = float(total_model.total_model.predict(X_total.to_numpy(dtype=float))[0])
    mu_margin_2h = float(margin_model.margin_model.predict(X_margin.to_numpy(dtype=float))[0])

    # 2H team points
    mu_h2_home = 0.5 * (mu_total_2h + mu_margin_2h)
    mu_h2_away = 0.5 * (mu_total_2h - mu_margin_2h)

    # Final means
    final_home_mu = float(h1_home) + float(mu_h2_home)
    final_away_mu = float(h1_away) + float(mu_h2_away)
    final_total_mu = final_home_mu + final_away_mu
    final_margin_mu = final_home_mu - final_away_mu

    # Uncertainty
    # Prefer quantile intervals when present; fall back to Normal(mu, sigma).
    h1_total = float(h1_home + h1_away)
    h1_margin = float(h1_home - h1_away)

    # Default sd from residual sigmas
    sd_total = float(total_model.sigma_total)
    sd_margin = float(margin_model.sigma_margin)

    # 2H intervals
    if total_model.total_q10_model is not None and total_model.total_q90_model is not None:
        q10_t = float(total_model.total_q10_model.predict(X_total.to_numpy(dtype=float))[0])
        q90_t = float(total_model.total_q90_model.predict(X_total.to_numpy(dtype=float))[0])
        h2_total_lo, h2_total_hi = q10_t, q90_t
        mu_total_2h, sd_total = normal_from_q10q90(q10_t, q90_t, default_sd=sd_total)
    else:
        h2_total_lo, h2_total_hi = _norm_pi80(mu_total_2h, sd_total)

    if margin_model.margin_q10_model is not None and margin_model.margin_q90_model is not None:
        q10_m = float(margin_model.margin_q10_model.predict(X_margin.to_numpy(dtype=float))[0])
        q90_m = float(margin_model.margin_q90_model.predict(X_margin.to_numpy(dtype=float))[0])
        h2_margin_lo, h2_margin_hi = q10_m, q90_m
        mu_margin_2h, sd_margin = normal_from_q10q90(q10_m, q90_m, default_sd=sd_margin)
    else:
        h2_margin_lo, h2_margin_hi = _norm_pi80(mu_margin_2h, sd_margin)

    # Recompute means in case quantile models adjusted mu_total_2h/mu_margin_2h
    mu_h2_home = 0.5 * (mu_total_2h + mu_margin_2h)
    mu_h2_away = 0.5 * (mu_total_2h - mu_margin_2h)

    final_home_mu = float(h1_home) + float(mu_h2_home)
    final_away_mu = float(h1_away) + float(mu_h2_away)
    final_total_mu = final_home_mu + final_away_mu
    final_margin_mu = final_home_mu - final_away_mu

    # Final intervals are halftime-shifted 2H intervals
    final_total_lo, final_total_hi = float(h2_total_lo + h1_total), float(h2_total_hi + h1_total)
    final_margin_lo, final_margin_hi = float(h2_margin_lo + h1_margin), float(h2_margin_hi + h1_margin)

    # Derived team sigma using (optional) covariance between total and margin.
    # Enhancements.txt: avoid independence assumption.
    var_t = float(sd_total) ** 2
    var_m = float(sd_margin) ** 2

    cov_tm = 0.0
    if total_model.residual_cov and isinstance(total_model.residual_cov, list):
        try:
            cov_tm = float(total_model.residual_cov[0][1])
        except Exception:
            cov_tm = 0.0

    var_home = (var_t + var_m + 2.0 * cov_tm) / 4.0
    var_away = (var_t + var_m - 2.0 * cov_tm) / 4.0

    sd_home = float(np.sqrt(max(0.01, var_home)))
    sd_away = float(np.sqrt(max(0.01, var_away)))

    final_home_lo, final_home_hi = _norm_pi80(final_home_mu, sd_home)
    final_away_lo, final_away_hi = _norm_pi80(final_away_mu, sd_away)

    out: Dict[str, Any] = {
        "game_id": gid,
        "home_name": home_name,
        "away_name": away_name,
        "h1_home": int(h1_home),
        "h1_away": int(h1_away),
        "live_home": int(live_home_pts),
        "live_away": int(live_away_pts),
        "live_total": int(live_home_pts + live_away_pts),
        "live_margin": int(live_home_pts - live_away_pts),
        "status": status,
        "_live": pace2h,
        "pred": {
            "pred_2h_total": mu_total_2h,
            "pred_2h_margin": mu_margin_2h,
            "pred_final_home": final_home_mu,
            "pred_final_away": final_away_mu,
        },
        # Compatibility with app and bet probability helper
        "bands80": {
            "h2_total": [h2_total_lo, h2_total_hi],
            "h2_margin": [h2_margin_lo, h2_margin_hi],
            "final_total": [final_total_lo, final_total_hi],
            "final_margin": [final_margin_lo, final_margin_hi],
            "final_home": [final_home_lo, final_home_hi],
            "final_away": [final_away_lo, final_away_hi],
        },
        "normal": {
            "h2_total": [h2_total_lo, h2_total_hi],
            "h2_margin": [h2_margin_lo, h2_margin_hi],
            "final_total": [final_total_lo, final_total_hi],
            "final_margin": [final_margin_lo, final_margin_hi],
        },
        "_derived": {
            "sd_final_total": float(sd_total),
            "sd_final_margin": float(sd_margin),
            "cov_total_margin": float(cov_tm),
        },
        "labels": {
            "total": (
                f"{total_model.model_name}(q80)"
                if total_model.total_q10_model is not None
                else f"{total_model.model_name}(sd={sd_total:.2f})"
            ),
            "margin": (
                f"{margin_model.model_name}(q80)"
                if margin_model.margin_q10_model is not None
                else f"{margin_model.model_name}(sd={sd_margin:.2f})"
            ),
        },
        "text": (
            f"GAME_ID: {gid}\n"
            f"Matchup: {away_name} @ {home_name}\n"
            f"Halftime: {home_name} {h1_home} – {h1_away} {away_name}\n\n"
            f"2H Projection:\n"
            f"  Total:  {mu_total_2h:.2f}  (80% CI: {h2_total_lo:.1f} – {h2_total_hi:.1f})\n"
            f"  Margin: {mu_margin_2h:.2f} (80% CI: {h2_margin_lo:.1f} – {h2_margin_hi:.1f})  [home={home_name}]\n\n"
            f"Final Projection:\n"
            f"  {home_name}: {final_home_mu:.1f}  (80% CI: {final_home_lo:.1f} – {final_home_hi:.1f})\n"
            f"  {away_name}: {final_away_mu:.1f}  (80% CI: {final_away_lo:.1f} – {final_away_hi:.1f})\n"
            f"  Total: {final_total_mu:.2f}  (80% CI: {final_total_lo:.1f} – {final_total_hi:.1f})\n"
            f"  Margin ({home_name}): {final_margin_mu:.2f}  (80% CI: {final_margin_lo:.1f} – {final_margin_hi:.1f})"
        ),
    }

    return out
