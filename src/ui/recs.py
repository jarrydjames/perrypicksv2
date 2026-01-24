from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from src.betting import fmt_pct


def kelly_to_text(f: float) -> str:
    if f <= 0:
        return "0% (no bet)"
    return f"{min(0.25, f) * 100:.1f}% of bankroll"


def render_recommendations(
    recs: List[Dict[str, Any]],
    *,
    kelly_mult: float,
    rank_by: str = "edge",
    on_track: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    """Mobile-friendly rec renderer.

    Avoid wide tables; use a compact list.
    """
    if not recs:
        st.info("Add market lines above to see bet evaluation.")
        return

    rank = str(rank_by or "edge").strip().lower()

    def fnum(v: Any) -> float:
        try:
            return float(v)
        except Exception:
            return float("-inf")

    # Sort once, use everywhere.
    if rank in {"p", "prob", "probability", "probability_to_hit"}:
        # Primary: probability. Tie-break: edge.
        sorted_recs = sorted(recs, key=lambda r: (fnum(r.get("p")), fnum(r.get("edge"))), reverse=True)
        rank_label = "probability to hit"
    else:
        # Primary: edge. Tie-break: probability.
        sorted_recs = sorted(recs, key=lambda r: (fnum(r.get("edge")), fnum(r.get("p"))), reverse=True)
        rank_label = "edge"

    top = sorted_recs[0]
    if top["edge"] <= 0.0:
        st.markdown("**Recommendation:** No clear value bet from the lines entered (all edges are ≤ 0).")
    else:
        ev100 = top.get("ev_per_100")
        ev_txt = f"EV ${float(ev100):+.2f} per $100" if ev100 is not None else "EV n/a"
        st.markdown(
            f"**Recommendation:** {top['side']} at **{top['odds']}** looks best "
            f"({fmt_pct(top['p'])} to hit, edge **{top['edge']*100:.1f} pts** vs break-even, {ev_txt}). "
            f"Suggested size (Kelly×{kelly_mult:.2f}): **{kelly_to_text(top['kelly'])}**."
        )

    with st.expander("Show top bets", expanded=False):
        st.caption(f"Evaluated {len(sorted_recs)} bets. Showing top 10 by {rank_label}.")
        show_all = st.checkbox("Show all evaluated bets", value=False, key="pp_show_all_bets")
        shown = sorted_recs if show_all else sorted_recs[:10]

        for i, r in enumerate(shown):
            ev100 = r.get("ev_per_100")
            ev_line = f"EV per $100: ${float(ev100):+.2f}" if ev100 is not None else "EV per $100: n/a"

            c1, c2 = st.columns([0.78, 0.22], vertical_alignment="center")
            with c1:
                action = r.get("action")
                stake_frac = float(r.get("stake_frac", 0.0) or 0.0)
                action_txt = f"**{action}**" if action else ""
                stake_txt = f"Stake: **{stake_frac*100:.2f}%**" if action == "BET" else "Stake: 0%"

                st.markdown(
                    f"{action_txt}  **{r['type']}** — {r['side']} @ `{r['odds']}`\n\n"
                    f"P(hit): **{fmt_pct(r['p'])}**  ·  Break-even: {fmt_pct(r['breakeven'])}  ·  Edge: **{r['edge']*100:.1f} pts**\n\n"
                    f"{ev_line}  ·  Kelly: {kelly_to_text(r['kelly'])}  ·  {stake_txt}"
                )

            with c2:
                if on_track is not None and r.get("action") == "BET":
                    if st.button("📌 Track", key=f"pp_track_rec_{i}", width="stretch"):
                        on_track(r)

            st.divider()
