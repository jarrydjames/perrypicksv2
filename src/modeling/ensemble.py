from __future__ import annotations

import math
from typing import Iterable, List, Tuple

import numpy as np

from src.modeling.uncertainty import clamp_sigma


def simplex_project(v: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex: w>=0, sum w = 1.

    Deterministic O(d log d) algorithm.
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("v must be 1D")

    n = v.shape[0]
    if n == 0:
        raise ValueError("empty vector")

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        # fallback: uniform
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    w = np.maximum(v - theta, 0.0)

    s = w.sum()
    if s <= 0:
        return np.ones(n) / n
    return w / s


def fit_ensemble_weights(
    base_mu: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.2,
    steps: int = 250,
) -> np.ndarray:
    """Fit non-negative weights that sum to 1.

    base_mu: shape (n_samples, n_models)
    y: shape (n_samples,)

    Uses projected gradient descent on MSE.
    """
    X = np.asarray(base_mu, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("base_mu must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("mismatched rows")

    m = X.shape[1]
    w = np.ones(m) / m

    for _ in range(int(steps)):
        pred = X @ w
        grad = (2.0 / X.shape[0]) * (X.T @ (pred - y))
        w = simplex_project(w - float(lr) * grad)

    return w


def ensemble_mu_sigma(
    base_mus: Iterable[float],
    base_sigmas: Iterable[float],
    weights: Iterable[float],
) -> Tuple[float, float]:
    """Combine mus and sigmas.

    Framework formula:
      mu* = Σ w_i mu_i
      sigma*^2 = Σ w_i^2 sigma_i^2 + Var(mu_i)

    The Var(mu_i) term is the (weighted) model disagreement.
    """
    mu = np.asarray(list(base_mus), dtype=float)
    sig = np.asarray([clamp_sigma(s) for s in base_sigmas], dtype=float)
    w = simplex_project(np.asarray(list(weights), dtype=float))

    mu_star = float(mu @ w)

    # Weighted disagreement term
    mu_mean = mu_star
    var_mu = float(((mu - mu_mean) ** 2 @ w))

    var_noise = float(((sig**2) * (w**2)).sum())
    sigma_star = math.sqrt(max(1e-8, var_noise + var_mu))
    return (mu_star, sigma_star)
