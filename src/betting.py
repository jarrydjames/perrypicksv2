from __future__ import annotations

import math

# -----------------------------
# Odds parsing + conversions
# -----------------------------

def parse_american_odds(x) -> int:
    """
    Accepts:
      - int/float like -110, 120
      - strings like "-110", "+120", "120", " -105 "
    Returns:
      int (american odds)
    """
    if x is None:
        raise ValueError("Odds cannot be None")
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(round(x))
    s = str(x).strip()
    if s.startswith("+"):
        s = s[1:].strip()
    if s == "":
        raise ValueError(f"Invalid odds: {x!r}")
    try:
        return int(float(s))
    except Exception as e:
        raise ValueError(f"Invalid odds: {x!r}") from e


def american_to_decimal(odds: int) -> float:
    """
    Converts American odds to decimal odds (includes stake).
    Example:
      -110 -> 1.9091
      +120 -> 2.20
    """
    odds = int(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def implied_prob_from_american(odds: int) -> float:
    """
    Implied probability from American odds INCLUDING vig.
    +120 -> 0.4545...
    -110 -> 0.5238...
    """
    odds = int(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def breakeven_prob_from_american(odds: int) -> float:
    """
    Break-even probability for a single bet at given odds
    (same as implied probability for that single price).
    """
    return implied_prob_from_american(int(odds))


# -----------------------------
# Edge + Kelly
# -----------------------------

def edge(p: float, breakeven: float) -> float:
    """
    Simple edge metric:
      edge = p - breakeven
    Positive means value (in probability terms).
    """
    return float(p) - float(breakeven)


def kelly_fraction(p: float, american_odds: int) -> float:
    """
    Kelly fraction f* for a single bet with win probability p and payout odds.

    Using decimal odds D:
      net odds b = D - 1
      f* = (b*p - (1-p)) / b
    Clamp to 0 if negative.
    """
    p = float(p)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    D = american_to_decimal(int(american_odds))
    b = D - 1.0
    if b <= 0:
        return 0.0

    f = (b * p - (1.0 - p)) / b
    return max(0.0, f)


# -----------------------------
# Normal distribution helpers
# -----------------------------

def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """
    CDF of Normal(mu, sigma). Uses erf via math.erf.
    """
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    z = (float(x) - float(mu)) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def prob_over_under_from_mean_sd(mu: float, sd: float, line: float) -> float:
    """
    Returns P(Over line) given total ~ Normal(mu, sd)
    """
    mu = float(mu)
    sd = float(sd)
    line = float(line)
    return 1.0 - normal_cdf(line, mu=mu, sigma=sd)


def prob_spread_cover_from_mean_sd(mu_margin: float, sd_margin: float, spread_line_home: float) -> float:
    """Returns P(home covers) using sportsbook spread convention.

    Definitions:
      margin = home - away ~ Normal(mu_margin, sd_margin)
      spread_line_home is the number shown next to the **home** team.
        Examples:
          home -12.5  -> spread_line_home = -12.5
          home +4.5   -> spread_line_home = +4.5

    A bet on the home spread wins when:
      margin + spread_line_home > 0
    Therefore:
      P(home covers) = P(margin > -spread_line_home)

    (The previous implementation incorrectly used `margin > spread_line_home`, which flips meaning for negative spreads.)
    """
    mu = float(mu_margin)
    sd = float(sd_margin)
    s = float(spread_line_home)
    threshold = -s
    return 1.0 - normal_cdf(threshold, mu=mu, sigma=sd)


def prob_moneyline_win_from_mean_sd(mu_margin: float, sd_margin: float) -> float:
    """Returns P(home wins) from the margin distribution.

    Assumes margin = home - away ~ Normal(mu_margin, sd_margin).
    Home wins if margin > 0.
    """
    return 1.0 - normal_cdf(0.0, mu=float(mu_margin), sigma=float(sd_margin))


# -----------------------------
# Compatibility helper (IMPORTANT)
# -----------------------------
def normal_from_q10q90(q10: float, q90: float, default_sd: float | None = None):
    """
    Backwards/forwards compatible helper used by src.predict_api.

    Supports:
      - normal_from_q10q90(q10, q90)
      - normal_from_q10q90(q10, q90, default_sd)

    Assumes q10/q90 are the 10th and 90th percentiles of a Normal distribution.

    For Normal:
      q90 - q10 = 2 * z * sd, where z = 1.281551565545
      sd = (q90 - q10) / (2*z)
      mu = (q10 + q90) / 2  (symmetric percentiles)

    Returns:
      (mu, sd)
    """
    z = 1.281551565545
    try:
        q10 = float(q10)
        q90 = float(q90)
        width = q90 - q10
        if width <= 0:
            raise ValueError("Non-positive interval width")
        sd = width / (2.0 * z)
        mu = 0.5 * (q10 + q90)
        if default_sd is not None:
            sd = max(float(default_sd), sd)
        return (mu, sd)
    except Exception:
        # If anything goes wrong, fall back to midpoint + default_sd (or a safe value)
        mu = 0.5 * (float(q10) + float(q90)) if q10 is not None and q90 is not None else 0.0
        sd = float(default_sd) if default_sd is not None else 1.0
        return (mu, max(0.01, sd))


# -----------------------------
# Optional: formatting helpers
# -----------------------------
def fmt_pct(x: float, digits: int = 1) -> str:
    """Format a probability as a percent string.

    Note: we intentionally avoid displaying hard 0.0% / 100.0% unless the value is
    truly extreme. This is *display-only*; we do not clamp the underlying math.
    """
    p = float(x)
    if p >= 0.999:
        return ">99.9%"
    if p <= 0.001:
        return "<0.1%"
    return f"{100.0*p:.{digits}f}%"
