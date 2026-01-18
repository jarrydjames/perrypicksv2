# Final model selection (production)

This doc captures the **final** model stack used by the Streamlit app.

## Goals
- **Calibration-first** probabilities for betting decisions (edges, cashout eval).
- **Streamlit Cloud friendly**: small artifacts, sklearn-only runtime deps.
- Avoid train/serve skew: live features must match the trained feature list.

## Models in production
### 1) Margin (spread / moneyline)
- **Artifact:** `models_v2/ridge_twohead.joblib`
- **Use cases:**
  - Spread bets (home/away cover)
  - Moneyline (home/away win)
- **Why:** Ridge was the most stable + best calibrated (coverage close to target) and best Brier for win probability.

### 2) Game total
- **Artifact:** `models_v2/gbt_twohead.joblib`
- **Use cases:**
  - Full game total (Over/Under)
  - Drives derived team totals
- **Why:** Small model artifact and strong overall performance. Works in Streamlit Cloud without xgboost/catboost.

### 3) Team totals
- **Artifact:** Derived (not directly modeled)
- **Mean:**
  - `home = (total + margin) / 2`
  - `away = (total - margin) / 2`
- **Uncertainty (conservative variance propagation):**
  - `Var(home) = (Var(total) + Var(margin)) / 4`
  - `sd_team = sqrt((sd_total^2 + sd_margin^2) / 4)`

## Features
Both production artifacts expect the same 27 halftime features:
- 1H score + action counts (`h1_*`)
- home/away efficiency rates + possessions + points-per-possession
- `game_poss_1h`

The runtime builder lives in `src/predict_from_gameid_v3_runtime.py`.

## Intentionally NOT shipped
- `models_v2/random_forest_twohead.joblib` (~84MB)

This is ignored via `.gitignore` to keep repo size and Streamlit build times sane.
