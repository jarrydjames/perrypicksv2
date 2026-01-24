import re
from datetime import datetime, timezone

import streamlit as st

# Optional autorefresh (recommended)
try:
    from streamlit_autorefresh import st_autorefresh

    HAS_AUTOREFRESH = True
except Exception:
    HAS_AUTOREFRESH = False

from src.predict_api import predict_game
from src.betting import parse_american_odds
from src.domain.bet_policy import list_presets, preset
from src.domain.markets import MarketInputs, evaluate_markets
from src.ui.recs import render_recommendations
from src.ui.calibration import render_calibration_report
from src.ui.styles import apply_base_styles
from src.ui.tracking import SnapshotThrottle, get_store, maybe_record_snapshot, render_tracking_panel
from src.odds.streamlit_cache import get_cached_nba_odds
from src.odds.odds_api import OddsAPIError
from src.ui.log_monitor import render_log_monitor

# -----------------------------
# Page + Theme UX
# -----------------------------
apply_base_styles()

# UI flags
SHOW_DEV_TOOLS = False

# -----------------------------
# Helpers
# -----------------------------
GAMEID_RE = re.compile(r"(002\d{7})")


def extract_gid_safe(s: str) -> str | None:
    """
    Returns GAME_ID (e.g., 0022500551) if present, else None.
    Never raises.
    """
    if not s:
        return None
    m = GAMEID_RE.search(str(s))
    return m.group(1) if m else None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sd_from_q10_q90(lo: float, hi: float) -> float:
    # For Normal: q90 - q10 ~= 2*1.2816*sd
    denom = 2.0 * 1.281551565545
    return max(0.01, (hi - lo) / denom)


def parse_pt_clock(clock_str: str | None) -> str | None:
    """
    Converts NBA clock strings like 'PT11M47.00S' to '11:47'.
    """
    if not clock_str:
        return None
    m = re.search(r"PT(\d+)M(\d+)(?:\.\d+)?S", clock_str)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    return f"{mm}:{ss:02d}"


def minutes_remaining(period: int | None, clock_str: str | None) -> float | None:
    """
    period: 1-4, clock_str: 'PT##M##.##S'
    Returns minutes remaining in regulation game (max 48).
    OT not handled (returns None).
    """
    if not period or not clock_str:
        return None
    m = re.search(r"PT(\d+)M(\d+)(?:\.\d+)?S", clock_str)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    mins_left_in_period = mm + ss / 60.0
    if period < 1 or period > 4:
        return None
    periods_left_after_this = 4 - period
    return periods_left_after_this * 12.0 + mins_left_in_period


def shrink_sd_with_clock(
    sd: float,
    min_rem: float | None,
    *,
    min_total: float = 24.0,
    min_sd: float = 1.0,
) -> float:
    """Shrink SD as time elapses in 2H.

    We use a sqrt-time heuristic, but **never** allow SD to fall below a realistic floor.
    Otherwise you'll get absurd 0%/100% probabilities.
    """
    if min_rem is None:
        return max(float(min_sd), float(sd))

    rem_2h = min(24.0, max(0.0, float(min_rem)))
    scale = (rem_2h / float(min_total)) ** 0.5 if float(min_total) > 0 else 1.0
    return max(float(min_sd), float(sd) * float(scale))


def kelly_to_text(f: float) -> str:
    if f <= 0:
        return "0% (no bet)"
    return f"{min(0.25, f) * 100:.1f}% of bankroll"


def init_state():
    st.session_state.setdefault("last_pred", None)
    st.session_state.setdefault("pred_history", [])      # list of {ts, pred}
    st.session_state.setdefault("tracked_bets", [])      # list of bet dicts
    st.session_state.setdefault("tracked_parlays", [])   # list of parlay dicts
    st.session_state.setdefault("track_history", {})     # key -> list of {ts,p,edge}
    st.session_state.setdefault("auto_refresh", False)
    st.session_state.setdefault("refresh_mins", 3)
    st.session_state.setdefault("use_clock_shrink", True)

    # Betting widget defaults (single source of truth = st.session_state)
    st.session_state.setdefault("total_line", 0.0)
    st.session_state.setdefault("odds_over", "-110")
    st.session_state.setdefault("odds_under", "-110")

    st.session_state.setdefault("spread_line_home", 0.0)
    st.session_state.setdefault("odds_home", "-110")
    st.session_state.setdefault("odds_away", "-110")

    st.session_state.setdefault("moneyline_home", "")
    st.session_state.setdefault("moneyline_away", "")

    st.session_state.setdefault("team_total_home", 0.0)
    st.session_state.setdefault("odds_team_over_home", "")
    st.session_state.setdefault("odds_team_under_home", "")

    st.session_state.setdefault("team_total_away", 0.0)
    st.session_state.setdefault("odds_team_over_away", "")
    st.session_state.setdefault("odds_team_under_away", "")


def _mark_user_set(key: str) -> None:
    st.session_state[f"_pp_user_set:{key}"] = True


def _maybe_apply_pending_odds_autofill() -> None:
    """Apply any pending odds autofill payload into widget session_state.

    Must run BEFORE widgets are created.

    Rules:
    - Only fill fields that are blank/0.
    - Never overwrite non-empty user-entered values.
    """

    payload = st.session_state.pop("_pp_autofill_odds", None)
    if not payload:
        return

    def set_if_empty(key: str, value) -> None:
        if value is None:
            return

        cur = st.session_state.get(key)
        user_set = bool(st.session_state.get(f"_pp_user_set:{key}", False))

        # Some fields have UX placeholders (e.g. -110). If the user never touched the field,
        # we consider that placeholder "empty" and allow autofill to overwrite it.
        PLACEHOLDER_VALUES = {
            "odds_over": "-110",
            "odds_under": "-110",
            "odds_home": "-110",
            "odds_away": "-110",
        }

        if not user_set and key in PLACEHOLDER_VALUES:
            if str(cur).strip() == str(PLACEHOLDER_VALUES[key]).strip():
                st.session_state[key] = value
                return

        # treat 0.0 as empty for number inputs
        if cur in (None, ""):
            st.session_state[key] = value
        elif isinstance(cur, (int, float)) and float(cur) == 0.0:
            st.session_state[key] = value

    # totals
    set_if_empty("total_line", payload.get("total_line"))
    set_if_empty("odds_over", payload.get("odds_over"))
    set_if_empty("odds_under", payload.get("odds_under"))

    # spread
    set_if_empty("spread_line_home", payload.get("spread_line_home"))
    set_if_empty("odds_home", payload.get("odds_home"))
    set_if_empty("odds_away", payload.get("odds_away"))

    # moneyline
    set_if_empty("moneyline_home", payload.get("moneyline_home"))
    set_if_empty("moneyline_away", payload.get("moneyline_away"))

    # team totals
    set_if_empty("team_total_home", payload.get("team_total_home"))
    set_if_empty("odds_team_over_home", payload.get("odds_team_over_home"))
    set_if_empty("odds_team_under_home", payload.get("odds_team_under_home"))

    set_if_empty("team_total_away", payload.get("team_total_away"))
    set_if_empty("odds_team_over_away", payload.get("odds_team_over_away"))
    set_if_empty("odds_team_under_away", payload.get("odds_team_under_away"))

    st.session_state["_pp_odds_status"] = payload.get("status")


def team_labels_for_ui(current_game_id: str | None) -> tuple[str, str]:
    """Return (home_label, away_label) for UI.

    Uses last prediction if it matches the currently entered game id.
    Falls back to generic labels.

    Important: Streamlit widget labels may change; always use stable `key=`.
    """
    pred = st.session_state.get("last_pred") or {}
    if current_game_id and pred.get("game_id") == current_game_id:
        return (pred.get("home_name") or "Home"), (pred.get("away_name") or "Away")
    return "Home", "Away"


init_state()

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <style>
      .pp-header-text { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Mobile-friendly header: no side logos (they make layout weird on narrow screens).
st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.markdown(
    """
    <div class="pp-header-text">
      <div class="pp-title">PerryPicks</div>
      <div class="pp-sub">Paste an NBA game URL or GAME_ID. Add lines/odds. Get a projection + value bets + tracking.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Inputs (top, no sidebar)
# -----------------------------
DEFAULT_GAME = "https://www.nba.com/game/nyk-vs-por-0022500551"

with st.container():
    # Mobile-first: fewer columns; still works fine on desktop.
    c1, c2 = st.columns([2.0, 1.0], vertical_alignment="bottom")

    with c1:
        # Optional: pick a game without leaving the app.
        with st.expander("Pick game by date (no more nba.com copy/paste)", expanded=False):
            import datetime as _dt

            from src.data.scoreboard import fetch_scoreboard, format_game_label

            st.session_state.setdefault("pp_pick_date", _dt.date.today())
            # Streamlit: don't pass both `value=` and a `key=` with session_state pre-set.
            pick_date = st.date_input("Date", key="pp_pick_date")

            try:
                games = fetch_scoreboard(pick_date)
            except Exception as e:
                games = []
                st.warning(f"Could not load scoreboard for {pick_date}: {e}")

            if not games:
                st.info("No games found for this date (or NBA CDN is being cranky).")
            else:
                labels = [format_game_label(g) for g in games]
                idx = st.selectbox("Games", list(range(len(games))), format_func=lambda i: labels[i])
                chosen = games[int(idx)]

                if st.button("Use selected game", width="stretch"):
                    st.session_state["game_input"] = chosen.game_id
                    st.rerun()

        game_input = st.text_input(
            "Game URL or GAME_ID",
            value=st.session_state.get("game_input", DEFAULT_GAME),
            help="Example: https://www.nba.com/game/nyk-vs-por-0022500551  or  0022500551",
        )
        st.session_state["game_input"] = game_input

    with c2:
        st.session_state.auto_refresh = st.toggle("Auto refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_mins = st.number_input(
                "Refresh every N minutes", min_value=1, max_value=10, value=int(st.session_state.refresh_mins), step=1
            )

        st.session_state.use_clock_shrink = st.toggle(
            "Clock-aware confidence",
            value=st.session_state.use_clock_shrink,
            help="Shrinks uncertainty as time runs off in 2H (but never below Min SD).",
        )

        st.session_state.setdefault("snapshot_every_min", 2)
        st.session_state.snapshot_every_min = st.number_input(
            "Save snapshot every N minutes",
            min_value=1,
            max_value=10,
            value=int(st.session_state.snapshot_every_min),
            step=1,
            help="Stores prediction snapshots for tracking probability drift."
        )

        manual_refresh = st.button("🔄 Refresh now", width="stretch")

    # Auto refresh hook
    if st.session_state.auto_refresh and HAS_AUTOREFRESH:
        st_autorefresh(interval=int(st.session_state.refresh_mins * 60_000), key="pp_autorefresh")
    elif st.session_state.auto_refresh and not HAS_AUTOREFRESH:
        st.info("Auto refresh needs `streamlit-autorefresh` (already in requirements.txt).")

st.write("")

# Grab team labels if we already have a prediction for this game.
_gid_hint = extract_gid_safe(st.session_state.get("game_input", ""))
ui_home, ui_away = team_labels_for_ui(_gid_hint)

# Apply any pending odds autofill BEFORE widgets are created.
_maybe_apply_pending_odds_autofill()

# -----------------------------
# Betting Inputs (directly under URL area)
# -----------------------------
with st.container():
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.subheader("Market lines (optional)")

    odds_status = st.session_state.get("_pp_odds_status")
    if odds_status:
        if str(odds_status).lower().startswith("odds auto-fill failed"):
            st.warning(str(odds_status))
        else:
            st.caption(str(odds_status))

    b1, b2 = st.columns(2)

    with b1:
        st.markdown("**Game total (O/U)**")
        total_line = st.number_input(
            "Total line",
            step=0.5,
            help="Enter 0 to ignore",
            key="total_line",
        )
        odds_over = st.text_input(
            "Over odds",
            key="odds_over",
            on_change=_mark_user_set,
            args=("odds_over",),
        )
        odds_under = st.text_input(
            "Under odds",
            key="odds_under",
            on_change=_mark_user_set,
            args=("odds_under",),
        )

        st.write("")
        st.markdown("**Home spread (home - away)**")
        spread_line_home = st.number_input(
            "Spread line",
            step=0.5,
            help="Example: -3.5 means home is -3.5",
            key="spread_line_home",
        )
        odds_home = st.text_input(
            f"{ui_home} spread odds",
            key="odds_home",
            on_change=_mark_user_set,
            args=("odds_home",),
        )
        odds_away = st.text_input(
            f"{ui_away} spread odds",
            key="odds_away",
            on_change=_mark_user_set,
            args=("odds_away",),
        )

    with b2:
        st.markdown("**Sizing**")

        policy_name = st.selectbox(
            "Bet policy preset",
            options=list_presets(),
            index=list_presets().index(st.session_state.get("policy_name", "Standard")),
            help="Controls selectivity + sizing caps. Most games should be NO BET."
        )
        st.session_state["policy_name"] = policy_name
        policy = preset(policy_name)

        bankroll = st.number_input("Bankroll", value=1000.0, step=50.0)
        st.caption(
            f"Policy: {policy.name}  min_edge={policy.min_edge*100:.1f} pts  "
            f"max_bets={policy.max_bets}  max_stake={policy.max_bankroll_frac_per_bet*100:.2f}%"
        )
        kelly_mult = st.slider(
            "Kelly multiplier",
            0.0,
            1.0,
            float(policy.kelly_mult),
            0.05,
            help="Fractional Kelly. Policy preset sets the default; you can still override."
        )

        with st.expander("Moneyline + Team totals", expanded=False):
            moneyline_home = st.text_input(
                f"{ui_home} moneyline odds",
                key="moneyline_home",
                on_change=_mark_user_set,
                args=("moneyline_home",),
            )
            moneyline_away = st.text_input(
                f"{ui_away} moneyline odds",
                key="moneyline_away",
                on_change=_mark_user_set,
                args=("moneyline_away",),
            )

            st.write("")
            st.markdown("**Team totals**")
            team_total_home = st.number_input(
                f"{ui_home} team total line",
                step=0.5,
                key="team_total_home",
            )
            odds_team_over_home = st.text_input(
                f"{ui_home} TT over odds",
                key="odds_team_over_home",
                on_change=_mark_user_set,
                args=("odds_team_over_home",),
            )
            odds_team_under_home = st.text_input(
                f"{ui_home} TT under odds",
                key="odds_team_under_home",
                on_change=_mark_user_set,
                args=("odds_team_under_home",),
            )

            st.write("")
            team_total_away = st.number_input(
                f"{ui_away} team total line",
                step=0.5,
                key="team_total_away",
            )
            odds_team_over_away = st.text_input(
                f"{ui_away} TT over odds",
                key="odds_team_over_away",
                on_change=_mark_user_set,
                args=("odds_team_over_away",),
            )
            odds_team_under_away = st.text_input(
                f"{ui_away} TT under odds",
                key="odds_team_under_away",
                on_change=_mark_user_set,
                args=("odds_team_under_away",),
            )

    st.markdown('<div class="pp-muted">Tip: Track a bet to record how your probability/edge moves through the 2nd half.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Backtest log monitor (optional)
# -----------------------------
if SHOW_DEV_TOOLS:
    with st.expander("🧾 Monitor overnight backtest log", expanded=False):
        render_log_monitor(logs_dir="logs", max_lines_default=250, key_prefix="logmon")

    st.write("")

# -----------------------------
# Validate input before running
# -----------------------------
gid = extract_gid_safe(game_input)
if gid is None:
    st.warning("Paste a full nba.com game URL or a GAME_ID like **0022500551** to run predictions.")
    st.stop()

# -----------------------------
# Run prediction (manual refresh or initial load)
# -----------------------------
def run_prediction():
    pred = predict_game(game_input)

    status = pred.get("status", {}) or {}
    period = status.get("period")
    clock = status.get("gameClock")

    min_rem = minutes_remaining(period, clock)

    # Derive SD from bands80 (q10/q90-like)
    bands = pred.get("bands80", {}) or {}
    (t_lo, t_hi) = bands.get("final_total", (None, None))
    (m_lo, m_hi) = bands.get("final_margin", (None, None))

    # Fallback to normal if needed
    normal = pred.get("normal", {}) or {}
    if t_lo is None or t_hi is None:
        (t_lo, t_hi) = normal.get("final_total", (None, None))
    if m_lo is None or m_hi is None:
        (m_lo, m_hi) = normal.get("final_margin", (None, None))

    # If still missing, use sane defaults
    base_sd_total = 12.0
    base_sd_margin = 8.0
    if t_lo is not None and t_hi is not None:
        base_sd_total = sd_from_q10_q90(float(t_lo), float(t_hi))
    if m_lo is not None and m_hi is not None:
        base_sd_margin = sd_from_q10_q90(float(m_lo), float(m_hi))

    final_sd_total = base_sd_total
    final_sd_margin = base_sd_margin

    # Only shrink in the 2nd half. (Shrinking in Q1 is... not math.)
    try:
        per_i = int(period) if period is not None else None
    except Exception:
        per_i = None

    min_rem_2h = None
    if per_i is not None and per_i >= 3 and min_rem is not None:
        # Clamp to regulation 2H window
        min_rem_2h = min(24.0, max(0.0, float(min_rem)))

    if st.session_state.use_clock_shrink and min_rem_2h is not None:
        min_total_sd = max(4.0, 0.35 * float(base_sd_total))
        min_margin_sd = max(3.0, 0.35 * float(base_sd_margin))

        # 1) clock-based shrink (sqrt-time)
        final_sd_total = shrink_sd_with_clock(final_sd_total, min_rem_2h, min_sd=min_total_sd)
        final_sd_margin = shrink_sd_with_clock(final_sd_margin, min_rem_2h, min_sd=min_margin_sd)

        # 2) pace-based variance scaling (sqrt(remaining possessions))
        # If the game is a track meet, uncertainty should be higher; if it's a slog, lower.
        # We estimate possessions/min from live PBP and compare to a baseline pace.
        pace = pred.get("_live", {}) or {}
        try:
            game_poss_2h = float(pace.get("game_poss_2h") or 0.0)
        except Exception:
            game_poss_2h = 0.0

        mins_elapsed_2h = max(0.0, 24.0 - float(min_rem_2h))
        poss_per_min = (game_poss_2h / mins_elapsed_2h) if (mins_elapsed_2h > 1e-6 and game_poss_2h > 0) else None

        # NBA baseline: ~100 possessions/game => ~2.08 game possessions/min
        baseline_ppm = 100.0 / 48.0

        if poss_per_min is not None:
            pace_factor = float(poss_per_min) / float(baseline_ppm)
            # keep it sane; early Q3 can be noisy
            pace_factor = max(0.75, min(1.35, pace_factor))
            final_sd_total = float(final_sd_total) * (pace_factor ** 0.5)
            final_sd_margin = float(final_sd_margin) * (pace_factor ** 0.5)

        # re-apply floors post scaling
        final_sd_total = max(float(min_total_sd), float(final_sd_total))
        final_sd_margin = max(float(min_margin_sd), float(final_sd_margin))

    # Live-conditioned means (helps tracking probabilities feel sane as the game evolves)
    mu2h_total = None
    mu2h_margin = None
    try:
        mu2h_total = float((pred.get("pred", {}) or {}).get("pred_2h_total"))
        mu2h_margin = float((pred.get("pred", {}) or {}).get("pred_2h_margin"))
    except Exception:
        mu2h_total = None
        mu2h_margin = None

    live_home = float(pred.get("live_home") or 0)
    live_away = float(pred.get("live_away") or 0)
    live_total = live_home + live_away
    live_margin = live_home - live_away

    mu_final_total_live = None
    mu_final_margin_live = None
    mu_final_home_live = None
    mu_final_away_live = None

    if (
        per_i is not None
        and per_i >= 3
        and min_rem_2h is not None
        and live_total > 0
        and mu2h_total is not None
        and mu2h_margin is not None
    ):
        # Pace-aware live conditioning (recommended): use observed 2H possessions + scoring
        # so totals/spreads drift with what the game is actually doing.
        pace = pred.get("_live", {}) or {}
        try:
            game_poss_2h = float(pace.get("game_poss_2h") or 0.0)
        except Exception:
            game_poss_2h = 0.0

        mins_elapsed_2h = max(0.0, 24.0 - float(min_rem_2h))
        poss_per_min = (game_poss_2h / mins_elapsed_2h) if (mins_elapsed_2h > 1e-6 and game_poss_2h > 0) else 0.0
        exp_rem_poss = poss_per_min * float(min_rem_2h)

        # Pull halftime scores from prediction payload (don’t rely on outer-scope vars)
        h1_home = float(pred.get("h1_home", 0.0) or 0.0)
        h1_away = float(pred.get("h1_away", 0.0) or 0.0)

        h1_total = float(h1_home + h1_away)
        h1_margin = float(h1_home - h1_away)
        obs_2h_total = float(live_total - h1_total)
        obs_2h_margin = float(live_margin - h1_margin)

        # Observed rates per possession (can be noisy early Q3)
        obs_total_ppp = (obs_2h_total / game_poss_2h) if game_poss_2h > 1e-6 else 0.0
        obs_margin_ppp = (obs_2h_margin / game_poss_2h) if game_poss_2h > 1e-6 else 0.0

        # Stability baseline: total points per *game* possession ~ 2.24 in NBA
        baseline_total_ppp = 2.24
        w = min(0.85, max(0.0, game_poss_2h / 40.0))  # ramp in with possessions
        blend_total_ppp = (w * obs_total_ppp) + ((1.0 - w) * baseline_total_ppp)

        frac = max(0.0, min(1.0, float(min_rem_2h) / 24.0))

        if exp_rem_poss > 0:
            mu_final_total_live = float(live_total + exp_rem_poss * blend_total_ppp)
        else:
            # fallback to time scaling if we can't estimate pace
            mu_final_total_live = float(live_total + mu2h_total * frac)

        # For margin, blend observed per-possession with model's time-scaled remaining margin.
        model_rem_margin = float(mu2h_margin) * frac
        obs_rem_margin = float(exp_rem_poss * obs_margin_ppp) if exp_rem_poss > 0 else model_rem_margin
        mu_final_margin_live = float(live_margin + ((1.0 - w) * model_rem_margin + w * obs_rem_margin))

        mu_final_home_live = 0.5 * (mu_final_total_live + mu_final_margin_live)
        mu_final_away_live = 0.5 * (mu_final_total_live - mu_final_margin_live)

    pred["_derived"] = {
        "min_remaining": min_rem,
        "base_sd_total": float(base_sd_total),
        "base_sd_margin": float(base_sd_margin),
        "sd_final_total": float(final_sd_total),
        "sd_final_margin": float(final_sd_margin),
        "clock_mmss": parse_pt_clock(clock),
        "period": period,
        "clock_str": clock,
        # optional live-conditioned means for tracking
        "mu_final_total": mu_final_total_live,
        "mu_final_margin": mu_final_margin_live,
        "mu_final_home": mu_final_home_live,
        "mu_final_away": mu_final_away_live,
    }

    st.session_state.last_pred = pred
    st.session_state.pred_history.append({"ts": now_utc_iso(), "pred": pred})


if manual_refresh or st.session_state.last_pred is None:
    try:
        run_prediction()
    except Exception as e:
        st.error(f"Prediction failed: {repr(e)}")
        st.stop()

# If user explicitly clicked refresh, attempt a cheap odds autofill.
# We fetch AFTER prediction so we know the matchup teams.
if manual_refresh:
    once_key = f"_pp_odds_autofill_rerun:{gid}:{st.session_state.get('last_pred', {}).get('status', {}).get('gameClock')}"
    if not st.session_state.get(once_key):
        try:
            p = st.session_state.last_pred or {}
            home_name = str(p.get("home_name") or "").strip()
            away_name = str(p.get("away_name") or "").strip()

            enable_team_totals = False
            try:
                # Free plan doesn’t support team_totals; keep this off unless you upgrade.
                enable_team_totals = bool(st.secrets.get("ODDS_API_ENABLE_TEAM_TOTALS", False))
            except Exception:
                enable_team_totals = False

            snap = get_cached_nba_odds(
                home_name=home_name,
                away_name=away_name,
                preferred_book="draftkings",
                include_team_totals=enable_team_totals,
                ttl_seconds=120,
            )

            st.session_state["_pp_autofill_odds"] = {
                "total_line": snap.total_points,
                "odds_over": str(snap.total_over_odds) if snap.total_over_odds is not None else None,
                "odds_under": str(snap.total_under_odds) if snap.total_under_odds is not None else None,
                "spread_line_home": snap.spread_home,
                "odds_home": str(snap.spread_home_odds) if snap.spread_home_odds is not None else None,
                "odds_away": str(snap.spread_away_odds) if snap.spread_away_odds is not None else None,
                "moneyline_home": str(snap.moneyline_home) if snap.moneyline_home is not None else None,
                "moneyline_away": str(snap.moneyline_away) if snap.moneyline_away is not None else None,
                "team_total_home": snap.team_total_home,
                "odds_team_over_home": str(snap.team_total_home_over_odds) if snap.team_total_home_over_odds is not None else None,
                "odds_team_under_home": str(snap.team_total_home_under_odds) if snap.team_total_home_under_odds is not None else None,
                "team_total_away": snap.team_total_away,
                "odds_team_over_away": str(snap.team_total_away_over_odds) if snap.team_total_away_over_odds is not None else None,
                "odds_team_under_away": str(snap.team_total_away_under_odds) if snap.team_total_away_under_odds is not None else None,
                "status": (
                    f"Auto-filled odds from Odds API ({snap.bookmaker or 'draftkings'})."
                    + (
                        " Team totals not available via your Odds API endpoint/plan."
                        if (
                            enable_team_totals
                            and (snap.team_total_home is None and snap.team_total_away is None)
                        )
                        else ""
                    )
                ),
            }
        except OddsAPIError as e:
            st.session_state["_pp_autofill_odds"] = {"status": f"Odds auto-fill failed: {e}"}
        except Exception as e:
            st.session_state["_pp_autofill_odds"] = {"status": f"Odds auto-fill failed (unexpected): {repr(e)}"}

        st.session_state[once_key] = True
        st.rerun()

pred = st.session_state.last_pred

# Record snapshots for tracking
force_key = f"_pp_force_snapshot:{gid}"
forced = bool(st.session_state.pop(force_key, False))

if forced:
    # Record immediately (ignores throttle) so the tracked bet has a starting point.
    maybe_record_snapshot(
        game_id=gid,
        pred=pred,
        throttle=SnapshotThrottle(every_n_minutes=1),
        last_record_key=f"_pp_last_snapshot_epoch:{gid}",
    )
else:
    # Throttled, e.g. every N minutes.
    maybe_record_snapshot(
        game_id=gid,
        pred=pred,
        throttle=SnapshotThrottle(every_n_minutes=int(st.session_state.snapshot_every_min)),
        last_record_key=f"_pp_last_snapshot_epoch:{gid}",
    )

# -----------------------------
# Display: game info + projections
# -----------------------------
home_name = pred.get("home_name", "HOME")
away_name = pred.get("away_name", "AWAY")
h1_home = float(pred.get("h1_home", 0))
h1_away = float(pred.get("h1_away", 0))

bands = pred.get("bands80", {}) or {}

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
g1, g2, g3, g4 = st.columns([1.3, 1.0, 1.0, 1.0])

with g1:
    st.markdown(f"**{away_name} @ {home_name}**")
    st.markdown(f"**Halftime:** {home_name} {int(h1_home)} – {int(h1_away)} {away_name}")

    # Live score (if available)
    live_home = int(pred.get("live_home") or 0)
    live_away = int(pred.get("live_away") or 0)
    if (live_home + live_away) > 0:
        live_margin = live_home - live_away
        st.markdown(f"**Score now:** {home_name} {live_home} – {live_away} {away_name}")
        st.caption(f"Margin now ({home_name}): {live_margin:+d}")

    per = pred["_derived"].get("period")
    mmss = pred["_derived"].get("clock_mmss")
    if per and mmss:
        st.markdown(
            f"<div style='font-size:34px;font-weight:900;line-height:1.0'>{mmss}</div>"
            f"<div class='pp-muted'>Q{per}</div>",
            unsafe_allow_html=True,
        )
    elif pred["_derived"]["min_remaining"] is not None:
        st.markdown(f"<span class='pp-muted'>Minutes remaining: {pred['_derived']['min_remaining']:.1f}</span>", unsafe_allow_html=True)

with g2:
    # 2H total is in pred["text"], but we also have normal/bands. Use bands80 h2_total if present.
    h2t = pred.get("normal", {}).get("h2_total")
    b_h2t = bands.get("h2_total")
    if b_h2t:
        h2t_lo, h2t_hi = b_h2t
    elif h2t:
        h2t_lo, h2t_hi = h2t
    else:
        h2t_lo, h2t_hi = (None, None)

    # The model mean isn't explicitly in bands; use midpoint if no mean provided.
    h2_total_mean = pred.get("pred_2h_total", None)
    if h2_total_mean is None and h2t_lo is not None and h2t_hi is not None:
        h2_total_mean = (float(h2t_lo) + float(h2t_hi)) / 2.0

    if h2_total_mean is None:
        st.markdown("<div class='pp-kpi'><div class='pp-muted'>2H Total</div><div style='font-size:20px;font-weight:800'>—</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='pp-kpi'><div class='pp-muted'>2H Total</div>"
            f"<div style='font-size:20px;font-weight:800'>{float(h2_total_mean):.2f}</div>"
            + (f"<div class='pp-muted'>80%: {float(h2t_lo):.1f} – {float(h2t_hi):.1f}</div>" if h2t_lo is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )

with g3:
    h2m = pred.get("normal", {}).get("h2_margin")
    b_h2m = bands.get("h2_margin")
    if b_h2m:
        h2m_lo, h2m_hi = b_h2m
    elif h2m:
        h2m_lo, h2m_hi = h2m
    else:
        h2m_lo, h2m_hi = (None, None)

    h2_margin_mean = pred.get("pred_2h_margin", None)
    if h2_margin_mean is None and h2m_lo is not None and h2m_hi is not None:
        h2_margin_mean = (float(h2m_lo) + float(h2m_hi)) / 2.0

    if h2_margin_mean is None:
        st.markdown(f"<div class='pp-kpi'><div class='pp-muted'>2H Margin ({home_name})</div><div style='font-size:20px;font-weight:800'>—</div></div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='pp-kpi'><div class='pp-muted'>2H Margin ({home_name})</div>"
            f"<div style='font-size:20px;font-weight:800'>{float(h2_margin_mean):.2f}</div>"
            + (f"<div class='pp-muted'>80%: {float(h2m_lo):.1f} – {float(h2m_hi):.1f}</div>" if h2m_lo is not None else "")
            + "</div>",
            unsafe_allow_html=True,
        )

with g4:
    ft_lo, ft_hi = bands.get("final_total", (None, None))
    final_total_mean = None
    if ft_lo is not None and ft_hi is not None:
        final_total_mean = (float(ft_lo) + float(ft_hi)) / 2.0
    st.markdown(
        "<div class='pp-kpi'><div class='pp-muted'>Final Total</div>"
        f"<div style='font-size:20px;font-weight:800'>{float(final_total_mean):.2f}</div>"
        + (f"<div class='pp-muted'>80%: {float(ft_lo):.1f} – {float(ft_hi):.1f}</div>" if ft_lo is not None else "")
        + "</div>",
        unsafe_allow_html=True,
    )

st.write("")
st.markdown("### Predicted final score")
sc1, sc2 = st.columns(2)

fh_lo, fh_hi = bands.get("final_home", (None, None))
fa_lo, fa_hi = bands.get("final_away", (None, None))

# derive means from intervals
final_home_mean = (float(fh_lo) + float(fh_hi)) / 2.0 if fh_lo is not None else 0.0
final_away_mean = (float(fa_lo) + float(fa_hi)) / 2.0 if fa_lo is not None else 0.0

with sc1:
    st.metric(label=home_name, value=f"{final_home_mean:.1f}", delta=None)
    if fh_lo is not None:
        st.caption(f"80% CI: {float(fh_lo):.1f} – {float(fh_hi):.1f}")

with sc2:
    st.metric(label=away_name, value=f"{final_away_mean:.1f}", delta=None)
    if fa_lo is not None:
        st.caption(f"80% CI: {float(fa_lo):.1f} – {float(fa_hi):.1f}")

st.write("")
st.markdown("### Predicted final margin")
fm_lo, fm_hi = bands.get("final_margin", (None, None))
final_margin_mean = (float(fm_lo) + float(fm_hi)) / 2.0 if fm_lo is not None else 0.0
m1, m2 = st.columns(2)
with m1:
    st.metric(label=f"Margin (home: {home_name} - {away_name})", value=f"{final_margin_mean:+.1f}")
with m2:
    if fm_lo is not None:
        st.metric(label="80% CI", value=f"{float(fm_lo):+.1f} to {float(fm_hi):+.1f}")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Betting evaluation + recommendation (Totals/Spread/ML/Team Totals)
# -----------------------------

def parse_optional_american(s: str):
    s = (s or "").strip()
    if not s:
        return None
    return parse_american_odds(s)


derived = pred.get("_derived", {}) or {}

final_total_mu = (float(ft_lo) + float(ft_hi)) / 2.0 if ft_lo is not None else None
final_margin_mu = (float(fm_lo) + float(fm_hi)) / 2.0 if fm_lo is not None else None

# Volatility proxy from model uncertainty (Enhancements.txt)
# Relative 80% PI width for final total.
volatility = None
if ft_lo is not None and ft_hi is not None and final_total_mu is not None:
    width = float(ft_hi) - float(ft_lo)
    denom = max(1.0, abs(float(final_total_mu)))
    volatility = max(0.0, width / denom)

# Team means (from bands)
final_home_mu = (float(fh_lo) + float(fh_hi)) / 2.0 if fh_lo is not None else None
final_away_mu = (float(fa_lo) + float(fa_hi)) / 2.0 if fa_lo is not None else None

sd_total = float(derived.get("sd_final_total", 12.0))
sd_margin = float(derived.get("sd_final_margin", 8.0))

inputs = MarketInputs(
    total_line=float(total_line),
    odds_over=parse_american_odds(odds_over),
    odds_under=parse_american_odds(odds_under),
    spread_home=float(spread_line_home),
    odds_home=parse_american_odds(odds_home),
    odds_away=parse_american_odds(odds_away),
    moneyline_home=parse_optional_american(moneyline_home),
    moneyline_away=parse_optional_american(moneyline_away),
    team_total_home=float(team_total_home),
    team_total_away=float(team_total_away),
    odds_team_over_home=parse_optional_american(odds_team_over_home),
    odds_team_under_home=parse_optional_american(odds_team_under_home),
    odds_team_over_away=parse_optional_american(odds_team_over_away),
    odds_team_under_away=parse_optional_american(odds_team_under_away),
    bankroll=float(bankroll),
    kelly_mult=float(kelly_mult),
)

policy = preset(st.session_state.get("policy_name", "Standard"))

recs = evaluate_markets(
    pred=pred,
    home_name=home_name,
    away_name=away_name,
    final_total_mu=final_total_mu,
    final_margin_mu=final_margin_mu,
    final_home_mu=final_home_mu,
    final_away_mu=final_away_mu,
    sd_total=sd_total,
    sd_margin=sd_margin,
    sd_team=None,
    inputs=inputs,
    policy=policy,
    volatility=volatility,
)

st.markdown('<div class="pp-card">', unsafe_allow_html=True)
st.subheader("Bet evaluation")
st.caption(
    f"Using SD(total)={sd_total:.2f}, SD(margin)={sd_margin:.2f}. "
    "Includes totals, spreads, moneyline, and team totals (if you entered lines)."
)

rank_choice = st.radio(
    "Rank bets by",
    options=["Edge", "Probability to hit"],
    horizontal=True,
    index=0,
    key="pp_rank_bets_by",
)
rank_by = "p" if rank_choice == "Probability to hit" else "edge"

def track_reco(r: dict) -> None:
    store = get_store()
    bet_id = f"{gid}:{now_utc_iso()}:{r.get('type')}:{r.get('side')}"
    store.add_bet(
        bet_id=bet_id,
        game_id=gid,
        bet_type=str(r.get("type")),
        side=str(r.get("side")),
        line=r.get("line"),
        odds=int(r.get("odds")),
        payload={"reco": r},
    )
    st.session_state[f"_pp_force_snapshot:{gid}"] = True
    st.success("Bet tracked. (Next refresh will capture a snapshot)")


render_recommendations(recs, kelly_mult=float(kelly_mult), rank_by=rank_by, on_track=track_reco)

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

# -----------------------------
# Calibration report (offline)
# -----------------------------
if SHOW_DEV_TOOLS:
    with st.expander("📏 Model calibration (offline backtest)", expanded=False):
        render_calibration_report()

    st.write("")

# -----------------------------
# Tracking (SQLite + Export/Import)
# -----------------------------
st.markdown('<div class="pp-card">', unsafe_allow_html=True)

show_hist = st.toggle(
    "Show tracking history",
    value=bool(st.session_state.get("pp_show_tracking_history", True)),
    help="Shows probability drift over time for tracked bets.",
)
st.session_state["pp_show_tracking_history"] = bool(show_hist)

render_tracking_panel(
    game_id=gid,
    home_name=home_name,
    away_name=away_name,
    recs=recs,
    snapshot_every_min=int(st.session_state.snapshot_every_min),
    show_export_import=False,
    show_snapshot_history=bool(show_hist),
)
st.markdown("</div>", unsafe_allow_html=True)
