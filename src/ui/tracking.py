from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from src.betting import american_to_decimal, breakeven_prob_from_american
from src.domain.bet_probability import prob_hit_for_bet
from src.storage.sqlite_store import SQLiteStore


@dataclass(frozen=True)
class SnapshotThrottle:
    every_n_minutes: int

    def should_record(self, last_ts_epoch: Optional[float]) -> bool:
        if not self.every_n_minutes or self.every_n_minutes <= 0:
            return False
        if last_ts_epoch is None:
            return True
        return (time.time() - float(last_ts_epoch)) >= (self.every_n_minutes * 60)


def get_store() -> SQLiteStore:
    # Cache the store per Streamlit session.
    if "_pp_store" not in st.session_state:
        st.session_state["_pp_store"] = SQLiteStore(db_path="data/perrypicks.sqlite")
    return st.session_state["_pp_store"]


def maybe_record_snapshot(
    *,
    game_id: str,
    pred: Dict[str, Any],
    throttle: SnapshotThrottle,
    last_record_key: str,
) -> None:
    """Record a snapshot (throttled) into SQLite.

    `last_record_key` is a session_state key used to store last-record epoch.
    """
    last_epoch = st.session_state.get(last_record_key)
    if not throttle.should_record(last_epoch):
        return

    store = get_store()
    store.add_snapshot(game_id, pred)
    st.session_state[last_record_key] = time.time()


def render_export_import(*, store: SQLiteStore, game_id: str) -> None:
    with st.expander("Export / Import", expanded=False):
        export = store.export_game(game_id)
        st.download_button(
            "⬇️ Export this game (JSON)",
            data=json.dumps(export, indent=2),
            file_name=f"perrypicks_{game_id}.json",
            mime="application/json",
            width="stretch",
        )

        up = st.file_uploader("Import a prior export (JSON)", type=["json"], accept_multiple_files=False)
        if up is not None:
            try:
                payload = json.loads(up.getvalue().decode("utf-8"))
                imported_gid = store.import_game(payload)
                st.success(f"Imported: {imported_gid}")
            except Exception as e:
                st.error(f"Import failed: {e!r}")


def _ev_hold(*, p: float, stake: float, american_odds: int) -> float:
    """Expected profit (not return) if you hold the bet."""
    D = american_to_decimal(int(american_odds))
    profit_if_win = float(stake) * (D - 1.0)
    return float(p) * profit_if_win - (1.0 - float(p)) * float(stake)


def _fair_cashout_return(*, p: float, stake: float, american_odds: int) -> float:
    """Fair cashout return (including stake) under your model."""
    D = american_to_decimal(int(american_odds))
    return float(p) * (float(stake) * D)


def render_tracking_panel(
    *,
    game_id: str,
    home_name: str,
    away_name: str,
    recs: list[dict],
    snapshot_every_min: int,
    show_export_import: bool = True,
    show_snapshot_history: bool = True,
) -> None:
    """Tracking UX (multi-bet)."""

    store = get_store()

    st.subheader("Tracking")
    st.caption("Track bets at halftime, then monitor probability drift through the 2nd half.")

    # Track a selected recommendation
    if not recs:
        st.info("Enter market lines above to generate recommendations, then track one or more bets.")
    else:
        labels = [f"{r['type']} — {r['side']} @ {r['odds']} (edge {r['edge']*100:.1f} pts)" for r in recs[:12]]
        idx = st.selectbox("Select a bet to track", list(range(len(labels))), format_func=lambda i: labels[i])
        chosen = recs[int(idx)]

        if st.button("📌 Track selected bet", width="stretch"):
            bet_id = f"{game_id}:{int(time.time())}:{chosen['type']}:{chosen['side']}"
            store.add_bet(
                bet_id=bet_id,
                game_id=game_id,
                bet_type=str(chosen["type"]),
                side=str(chosen["side"]),
                line=chosen.get("line"),
                odds=int(chosen.get("odds")),
                payload={"reco": chosen},
            )
            # Nice UX: ask the main app loop to capture an immediate snapshot.
            st.session_state[f"_pp_force_snapshot:{game_id}"] = True
            st.success("Bet tracked. (Next refresh will capture a snapshot)")

    # View tracked bets
    bets = store.list_bets(game_id)
    if not bets:
        if show_export_import:
            render_export_import(store=store, game_id=game_id)

        if show_snapshot_history:
            st.markdown("### Snapshot history")
            st.caption("Snapshots are auto-recorded in the background while this page stays open.")
            st.write(f"Snapshot interval: **every {snapshot_every_min} min**")
            st.write(f"Saved snapshots: **{len(store.list_snapshots(game_id))}**")

        return

    bet_labels = [f"{b.bet_type} — {b.side} @ {b.odds} ({b.created_ts_utc})" for b in bets]
    bet_idx = st.selectbox("Tracked bet to view", list(range(len(bet_labels))), format_func=lambda i: bet_labels[i])
    selected_bet = bets[int(bet_idx)]

    with st.expander(f"Tracked bets ({len(bets)})", expanded=False):
        for b in bets:
            st.markdown(f"**{b.bet_type}** — {b.side} @ `{b.odds}`  ·  {b.created_ts_utc}")

    if show_export_import:
        render_export_import(store=store, game_id=game_id)

    if not show_snapshot_history:
        return

    # Snapshot view + drift chart
    st.markdown("### Snapshot history")
    st.caption("Snapshots are auto-recorded while this page stays open. Export if you want to keep them forever.")
    st.write(f"Snapshot interval: **every {snapshot_every_min} min**")

    snaps = store.list_snapshots(game_id)
    st.write(f"Saved snapshots: **{len(snaps)}**")

    if not snaps:
        st.info("No snapshots saved yet. Leave the page open during the 2nd half (or hit refresh) to accumulate snapshots.")
        return

    rows = []
    for s in snaps:
        p = prob_hit_for_bet(
            pred=s.payload,
            bet_type=selected_bet.bet_type,
            side=selected_bet.side,
            line=selected_bet.line,
            home_name=home_name,
            away_name=away_name,
        )
        rows.append({"ts_utc": s.ts_utc, "p_hit": p})

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["p_hit"]).copy()

    if df.empty:
        st.warning("Couldn't compute probabilities for this bet from snapshots (missing fields).")
        return

    # Opening vs current
    p_open = float(df.iloc[0]["p_hit"])
    p_now = float(df.iloc[-1]["p_hit"])

    be = None
    edge_open = None
    edge_now = None
    if selected_bet.odds is not None:
        be = float(breakeven_prob_from_american(int(selected_bet.odds)))
        edge_open = p_open - be
        edge_now = p_now - be

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("P(hit) now", f"{p_now*100:.1f}%", delta=f"{(p_now-p_open)*100:+.1f} pts")
    with k2:
        st.metric("Break-even", "" if be is None else f"{be*100:.1f}%")
    with k3:
        if edge_now is None:
            st.metric("Edge now", "")
        else:
            st.metric("Edge now", f"{edge_now*100:+.1f} pts", delta=f"{(edge_now-edge_open)*100:+.1f} pts")

    if edge_now is not None:
        st.success("Still +EV based on current snapshot.") if edge_now > 0 else st.warning(
            "No longer +EV based on current snapshot (edge <= 0)."
        )

    st.line_chart(df.set_index("ts_utc")["p_hit"], height=220)

    with st.expander("Cashout helper", expanded=False):
        st.caption("Enter your stake and the sportsbook cashout return to compare fair value vs cashout.")

        if selected_bet.odds is None:
            st.info("Cashout math requires odds on the tracked bet.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                stake = st.number_input("Stake", min_value=0.0, value=100.0, step=10.0)
            with c2:
                cashout_return = st.number_input(
                    "Cashout offer (return)",
                    min_value=0.0,
                    value=0.0,
                    step=10.0,
                    help="Total amount you get back if you accept cashout now (includes stake).",
                )

            fair_return = _fair_cashout_return(p=p_now, stake=float(stake), american_odds=int(selected_bet.odds))
            ev_hold = _ev_hold(p=p_now, stake=float(stake), american_odds=int(selected_bet.odds))

            st.write(f"Fair cashout return (model): **{fair_return:,.2f}**")
            st.write(f"EV of holding (profit): **{ev_hold:,.2f}**")

            if cashout_return and cashout_return > 0:
                if float(cashout_return) >= fair_return:
                    st.success("Cashout is at or above fair value (per model). Taking it is reasonable.")
                else:
                    st.info("Cashout is below fair value (per model). Holding is +EV vs cashout.")

    with st.expander("Raw values", expanded=False):
        st.dataframe(df.tail(50), width="stretch")
