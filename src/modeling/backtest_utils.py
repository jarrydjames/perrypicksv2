from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------
# metrics
# -----------------

def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return float(np.mean((y >= lo) & (y <= hi)))


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y_true) ** 2))


def ece(y_true: np.ndarray, p: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected calibration error for binary probabilities.

    Returns a value in [0,1], lower is better.
    """

    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(p)
    y_true = y_true[mask]
    p = p[mask]

    if len(p) == 0:
        return 0.0

    n_bins = int(max(2, n_bins))
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        n = int(np.sum(in_bin))
        if n == 0:
            continue
        acc = float(np.mean(y_true[in_bin]))
        conf = float(np.mean(p[in_bin]))
        ece_val += (n / len(p)) * abs(acc - conf)

    return float(ece_val)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Normal CDF via erf (portable)."""

    from math import erf

    z = np.asarray(z, dtype=float)
    v_erf = np.vectorize(erf)
    return 0.5 * (1.0 + v_erf(z / np.sqrt(2.0)))


def p_home_win(mu_margin: np.ndarray, sigma_margin: float) -> np.ndarray:
    # P(margin > 0) = 1 - Phi((0 - mu)/sigma)
    s = float(max(1e-9, sigma_margin))
    z = (0.0 - np.asarray(mu_margin, dtype=float)) / s
    return 1.0 - normal_cdf(z)


# -----------------
# data
# -----------------

def _box_game_time_utc(box_path: Path) -> Optional[str]:
    """Extract gameTimeUTC from a cached boxscore json.

    We support two cache formats:
    1) Full payload: {"game": {"gameTimeUTC": "...", ...}}
    2) Slim payload (what our cache script writes): {"gameTimeUTC": "...", ...}
    """

    try:
        data = json.loads(box_path.read_text())

        # Format (2): slim
        ts = data.get("gameTimeUTC")
        if ts:
            return str(ts)

        # Format (1): full
        game = data.get("game") or {}
        ts = game.get("gameTimeUTC")
        if ts:
            return str(ts)

    except Exception:
        return None

    return None


def attach_game_time_utc(df: pd.DataFrame, *, box_dir: Path) -> pd.DataFrame:
    """Attach gameTimeUTC from cached box json.

    If the local box cache is missing (common on fresh clones), we fall back to a
    deterministic ordering key so walk-forward splits still work.

    NOTE: This fallback does *not* represent true chronological ordering, but it
    is good enough for quick local experiments.
    """

    times: List[Optional[str]] = []
    for gid in df["game_id"].astype(str).tolist():
        ts = _box_game_time_utc(box_dir / f"{gid}.json")
        times.append(ts)

    out = df.copy()
    out["gameTimeUTC"] = times

    # If we have at least some timestamps, use them.
    if out["gameTimeUTC"].notna().any():
        out = out.dropna(subset=["gameTimeUTC"]).copy()
        out["gameTimeUTC"] = pd.to_datetime(out["gameTimeUTC"], utc=True, errors="coerce")
        out = out.dropna(subset=["gameTimeUTC"]).copy()
        return out

    # Fallback: create a pseudo-timestamp for stable sorting
    # Prefer season + game_id if available.
    if "season_end_yy" in out.columns:
        key = out["season_end_yy"].astype(str).str.zfill(4) + "_" + out["game_id"].astype(str)
    else:
        key = out["game_id"].astype(str)

    out["gameTimeUTC"] = pd.to_datetime(key, errors="coerce", utc=True)
    # If coercion failed (likely), just set a monotonic range.
    if out["gameTimeUTC"].isna().all():
        out["gameTimeUTC"] = pd.to_datetime(np.arange(len(out)), unit="s", utc=True)

    return out


# -----------------
# folds
# -----------------


@dataclass(frozen=True)
class FoldSpec:
    train_min: int
    test_size: int
    step_size: int


def iter_walkforward_indices(n: int, *, spec: FoldSpec) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) in a walk-forward fashion."""

    start = int(spec.train_min)
    while start + spec.test_size <= n:
        train_idx = np.arange(0, start)
        test_idx = np.arange(start, start + spec.test_size)
        yield train_idx, test_idx
        start += int(spec.step_size)
