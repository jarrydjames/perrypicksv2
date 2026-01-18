from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from src.betting import fmt_pct


def kelly_to_text(f: float) -> str:
    if f <= 0:
        return "0% (no bet)"
    return f"{min(0.25, f) * 100:.1f}% of bankroll"


def render_recommendations(recs: List[Dict[str, Any]], *, kelly_mult: float) -> None:
    """Mobile-friendly rec renderer.

    Avoid wide tables; use a compact list.
    """
    if not recs:
        st.info("Add market lines above to see bet evaluation.")
        return

    top = recs[0]
    if top["edge"] <= 0.0:
        st.markdown("**Recommendation:** No clear value bet from the lines entered (all edges are ≤ 0).")
    else:
        st.markdown(
            f"**Recommendation:** {top['side']} at **{top['odds']}** looks best "
            f"({fmt_pct(top['p'])} to hit, edge **{top['edge']*100:.1f} pts** vs break-even). "
            f"Suggested size (Kelly×{kelly_mult:.2f}): **{kelly_to_text(top['kelly'])}**."
        )

    with st.expander("Show top bets", expanded=False):
        for r in recs[:10]:
            st.markdown(
                f"**{r['type']}** — {r['side']} @ `{r['odds']}`\n\n"
                f"P(hit): **{fmt_pct(r['p'])}**  ·  Break-even: {fmt_pct(r['breakeven'])}  ·  Edge: **{r['edge']*100:.1f} pts**\n\n"
                f"Kelly: {kelly_to_text(r['kelly'])}"
            )
            st.divider()
