from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


PolicyName = Literal["Conservative", "Standard", "Aggressive"]


@dataclass(frozen=True)
class BetPolicy:
    """Rule set for turning model outputs into actionable recommendations.

    Keep this separate from market math (SRP).
    """

    name: PolicyName

    # Minimum probability edge vs break-even required to recommend a bet.
    # Example: 0.06 means model must beat break-even by 6 percentage points.
    min_edge: float

    # Fractional Kelly multiplier. 0.0 = flat sizing (stake decided elsewhere).
    kelly_mult: float

    # Absolute max fraction of bankroll per bet (even after Kelly). Example 0.01 = 1%.
    max_bankroll_frac_per_bet: float

    # Maximum number of bets to recommend at once.
    max_bets: int

    # Minimum model probability allowed to bet (avoids betting huge plus-money longshots off noise).
    min_model_p: float = 0.0

    # Optional: require a minimum expected value edge in raw Kelly terms.
    min_kelly: float = 0.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def apply_policy(recs: List[Dict[str, Any]], policy: BetPolicy) -> List[Dict[str, Any]]:
    """Annotate + filter market recs.

    Adds:
      - action: "BET" | "PASS"
      - stake_frac: recommended bankroll fraction (capped)

    Returns filtered list (BETs first), respecting policy.max_bets.
    """

    out: List[Dict[str, Any]] = []

    for r in recs:
        edge = float(r.get("edge", 0.0))
        p = float(r.get("p", 0.0))
        kelly = float(r.get("kelly", 0.0))  # already includes multiplier in current pipeline

        action = "BET"
        reasons: list[str] = []

        if edge < float(policy.min_edge):
            action = "PASS"
            reasons.append(f"edge<{policy.min_edge:.3f}")

        if p < float(policy.min_model_p):
            action = "PASS"
            reasons.append(f"p<{policy.min_model_p:.3f}")

        if kelly < float(policy.min_kelly):
            action = "PASS"
            reasons.append(f"kelly<{policy.min_kelly:.3f}")

        # Cap stake
        stake_frac = clamp01(kelly)
        stake_frac = min(stake_frac, float(policy.max_bankroll_frac_per_bet))

        rr = dict(r)
        rr["action"] = action
        rr["policy"] = policy.name
        rr["policy_reasons"] = reasons
        rr["stake_frac"] = float(stake_frac) if action == "BET" else 0.0

        out.append(rr)

    # Prefer BETs, then highest edge
    out.sort(key=lambda r: (r.get("action") != "BET", -float(r.get("edge", 0.0))))

    bets = [r for r in out if r.get("action") == "BET"]
    passes = [r for r in out if r.get("action") != "BET"]

    # IMPORTANT: do not drop evaluations.
    # Policy should annotate what to do, not hide information.
    max_bets = max(0, int(policy.max_bets))
    if len(bets) > max_bets:
        keep = bets[:max_bets]
        overflow = bets[max_bets:]
        for r in overflow:
            r["action"] = "PASS"
            r["stake_frac"] = 0.0
            reasons = list(r.get("policy_reasons") or [])
            reasons.append("max_bets")
            r["policy_reasons"] = reasons
        bets = keep
        passes = overflow + passes

    return bets + passes


def preset(name: PolicyName) -> BetPolicy:
    # These defaults are intentionally selective. The vig is real.
    if name == "Conservative":
        return BetPolicy(
            name=name,
            min_edge=0.08,
            kelly_mult=0.25,
            max_bankroll_frac_per_bet=0.005,  # 0.5%
            max_bets=1,
            min_model_p=0.55,
        )

    if name == "Aggressive":
        return BetPolicy(
            name=name,
            min_edge=0.04,
            kelly_mult=0.50,
            max_bankroll_frac_per_bet=0.015,  # 1.5%
            max_bets=3,
            min_model_p=0.52,
        )

    # Standard
    return BetPolicy(
        name="Standard",
        min_edge=0.06,
        kelly_mult=0.35,
        max_bankroll_frac_per_bet=0.01,  # 1%
        max_bets=2,
        min_model_p=0.54,
    )


def list_presets() -> List[PolicyName]:
    return ["Conservative", "Standard", "Aggressive"]
