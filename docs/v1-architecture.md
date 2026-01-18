# PerryPicks V1 — Architecture + Data Flow (Repo: `perrypicks`)

This doc describes **what V1 actually does**, based on the code in this repo.
It’s intentionally un-sexy: clarity > clever.

## High-level

- **UI:** `app.py` (Streamlit, single-page)
- **Prediction entrypoint:** `src/predict_api.py::predict_game()`
- **Prediction implementation:** `src/predict_from_gameid_v2_ci.py::predict_from_game_id()`
- **Model inference (means):** `src/predict_from_gameid.py::predict_from_halftime()`
- **Betting math:** `src/betting.py`
- **Bundled assets:**
  - `models/*.joblib` (sklearn models + feature lists)
  - `data_processed/*` (backtests + parquet/json artifacts; not required for runtime)

## Runtime environment

- Streamlit Cloud
- Python: `runtime.txt` => `python-3.12`
- Dependencies: `requirements.txt`

## User flow (UI)

File: `app.py`

1. User pastes:
   - NBA game URL like `https://www.nba.com/game/nyk-vs-por-0022500551`
   - OR a raw `GAME_ID` like `0022500551`
2. (Optional) User enters betting lines + odds:
   - game total + over/under odds
   - home spread (home - away) + home/away odds
   - bankroll + kelly multiplier
3. App calls `predict_game(game_input)`
4. App displays:
   - matchup + halftime score
   - time/period (if available)
   - 2H total + 2H margin projections
   - final score + final total + final margin
5. App computes probabilities/edges from Normal assumptions and recommends a bet.

### Session state

`app.py` uses `st.session_state` for:

- `last_pred`: last prediction payload (dict)
- `pred_history`: [{ts, pred}, ...]
- `auto_refresh`, `refresh_mins`
- `use_clock_shrink`: shrink uncertainty as game clock runs (2H)

Nothing persists across sessions/users (no DB).

## Prediction flow (end-to-end)

### 1) Public entrypoint

File: `src/predict_api.py`

```py
def predict_game(game_input: str, use_binned_intervals: bool = True) -> Dict[str, Any]:
    from src.predict_from_gameid_v2_ci import predict_from_game_id
    return predict_from_game_id(game_input, use_binned_intervals=use_binned_intervals)
```

### 2) Fetch live game data

File: `src/predict_from_gameid_v2_ci.py`

- Extracts GAME_ID via regex.
- Pulls NBA CDN JSON endpoints:
  - play-by-play: `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gid}.json`
  - boxscore: `https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json`

### 3) Feature engineering (current-game)

- Halftime score derived from boxscore `periods` for periods 1–2.
- 1H “behavior counts” derived from PBP rows where `period <= 2`:
  - counts of actionType prefixes: 2pt/3pt/turnover/rebound/foul/timeout/substitution

### 4) Model inference (means)

File: `src/predict_from_gameid.py::predict_from_halftime()`

Loads models:

- `models/team_2h_total.joblib`
- `models/team_2h_margin.joblib`

Each joblib is expected to contain:

- `{"features": [...], "model": sklearn_model}`

Computes:

- `pred_2h_total`
- `pred_2h_margin` (home - away)

Then derives:

- `pred_final_home`, `pred_final_away`, `pred_final_total`, `pred_final_margin`

### 5) Uncertainty / bands

File: `src/predict_from_gameid_v2_ci.py`

Builds central **80% intervals** using Normal approximations (q10/q90):

- Uses fixed baseline SDs (conservative defaults):
  - total SD ~ 12
  - margin SD ~ 8
- Optionally widens slightly for `use_binned_intervals`.

Returns payload containing:

- `bands80`: intervals for h2_total/h2_margin/final_total/final_margin/final_home/final_away
- `normal`: same structure kept for backwards compatibility
- `status`: period + clock + status text when available
- `text`: human-readable output blob

### 6) Clock-aware confidence (UI-side)

File: `app.py`

- Derives SD from q10/q90 using:
  - `sd = (q90 - q10) / (2 * 1.28155)`
- Optionally shrinks SD based on minutes remaining in regulation:
  - `sd_scaled = sd * sqrt(remaining_2h / 24)`

This affects bet probabilities/edges.

## Betting evaluation

File: `src/betting.py`

- Parses American odds
- Converts to break-even probability
- Assumes outcomes are Normal:
  - totals: `Total ~ Normal(mu_total, sd_total)`
  - margin: `Margin(home-away) ~ Normal(mu_margin, sd_margin)`
- Computes:
  - `P(over)` / `P(under)`
  - `P(home covers)` / `P(away covers)`
- Computes simple edge:
  - `edge = p - breakeven`
- Computes Kelly fraction:
  - `f* = (b*p - (1-p)) / b` clamped to >= 0

## “Data contract” between prediction layer and UI

`app.py` expects prediction payload keys:

- `home_name`, `away_name`
- `h1_home`, `h1_away`
- `status`: dict with `period`, `gameClock` (best-effort)
- `bands80`: dict with keys:
  - `h2_total`, `h2_margin`, `final_total`, `final_margin`, `final_home`, `final_away`
  - each value is `[lo, hi]` (q10/q90-ish)
- `normal`: dict (fallback intervals)

## Known limitations / gotchas (useful for V2)

- **No caching** of NBA CDN calls or model loads (Streamlit re-runs can be chatty).
- **Uncertainty is heuristic**, not truly calibrated from the model (SD defaults).
- **Clock handling** is UI-only and only supports regulation periods 1–4 (no OT).
- **Tracking is session-only**; refresh loses everything.
- **`app.py` mixes UI + domain logic** (hard to test, easy to spaghetti).

## Suggested V2 refactor boundaries

Keep it boring and testable:

- `src/integrations/nba_cdn.py` (fetch + parse box/pbp)
- `src/domain/features.py` (feature builders)
- `src/domain/predict.py` (model loading + inference)
- `src/domain/uncertainty.py` (bands + calibration)
- `src/domain/betting.py` (already mostly there)
- `app/` (Streamlit UI only)

