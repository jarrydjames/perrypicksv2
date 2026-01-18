# PerryPicks V2 — Plan (aligned to `nba_model_spec_v4`)

This doc is the **working plan** for V2. It’s based on:
- current V1 repo (`perrypicks/`)
- your contract/testing framework (`nba_model_spec_v4/`)

## Goals you stated

- **Decision point is halftime** (place/skip bet).
- After halftime, you want to **track over time**:
  - how predictions change
  - how probability of the bet hitting changes
  - through the remainder of the game
- V2 includes **both** model/prediction rigor and UI improvements.
- Single-game decision UX is primary; multi-game view is a nice-to-have.

---

## What “persistent tracking” buys you (benefits)

Persistent tracking means predictions + bets are stored outside Streamlit session memory.
In V1, everything is `st.session_state` → it evaporates on refresh/deploy/session.

### 1) You can measure whether you’re actually improving
Without persistence:
- you can’t compute long-term hit rate / ROI
- you can’t compare model versions
- you can’t quantify whether calibration changes helped

With persistence:
- track per-bet outcomes and compute:
  - win/loss
  - closing line value (if you add it)
  - ROI by market type
  - calibration curves (reliability) from real usage

### 2) You can plot probability drift after you place a bet
This is directly tied to your requirement.

Persist snapshots like:
- halftime prediction (time T0)
- Q3 9:00 (T1)
- Q3 6:00 (T2)
- Q4 12:00...

Then you can plot:
- `p_hit(t)` vs time
- `mu_total(t)` vs time
- `sigma_total(t)` vs time

This becomes the “confidence monitor” for the remainder of the game.

### 3) Multi-game dashboards become possible
If everything is stored, you can show:
- today’s tracked games
- active bets and their current p(hit)
- historical performance

Without persistence, multi-game support is mostly a UI toy.

### 4) Reproducibility + audit trail
If you ever ask “why did it recommend that?” you can answer with:
- the exact input snapshot
- the exact model version
- the exact market lines
- the exact probability + edge at the time

This is a massive quality-of-life upgrade.

### 5) Enables contract enforcement in production
Your `nba_model_spec_v4` output schema wants:
- `model_version`, `feature_version`, `calibration_id`

Persisting outputs means you can enforce that every record is “valid per contract” and later debug any violations.

---

## Persistent tracking tradeoffs (real talk)

### Streamlit Cloud + “free only” storage reality
You want this to run on **Streamlit Cloud free** with **no paid hosting**.
That means we should assume:
- no guaranteed durable disk across redeploys
- no paid database
- no paid APIs

So the best-practice move is:
- keep the app fully functional with **zero external services**
- provide **export/import** so you can keep history without hosting anything

Options:
1. **SQLite file (local disk)**
   - ✅ simplest MVP, works great in local dev
   - ✅ can survive reruns within the same container
   - ⚠️ may reset on redeploy / container recycle
2. **Export/Import (CSV/JSON)**
   - ✅ truly free + durable (you download it)
   - ✅ works even if Streamlit resets
   - ⚠️ requires user action (download/upload)
3. **Free-tier hosted DB** (optional later)
   - ✅ durable + multi-device
   - ⚠️ might require accounts/keys; free tiers can change

Recommendation under your constraints:
- **V2 MVP:** SQLite + one-click Export/Import. No external services.
- Build a small storage interface so if you later want a free-tier DB, we plug it in cleanly.

---

## How V2 tracking should work (concept)

### Bet lifecycle
1. At halftime:
   - user enters market lines + odds
   - app creates a `bet` record (or “decision”) with:
     - model output at decision time
     - probability, breakeven, edge, kelly suggestion
2. During 2H:
   - app periodically polls game status + score
   - app re-computes updated probabilities for that *same bet line*
   - app stores time-series `prediction_snapshot` rows
3. At game end:
   - app marks bet as settled (win/loss/push) and stores final score.

This matches your requirement: decision at halftime, but track p(hit) for duration.

### What changes after halftime?
Two possibilities (pick one in V2):
- **A) “Static model + shrinking sigma”** (what V1 effectively does)
  - mu stays anchored to halftime projection
  - sigma shrinks with clock
  - p(hit) changes mainly because uncertainty shrinks

- **B) “Dynamic re-forecast”** (more powerful)
  - incorporate live score in 2H and update mu
  - sigma shrinks too
  - p(hit) changes due to both mu shift and sigma shrink

Your spec v4 input schema includes `period` and `seconds_remaining_in_period`, plus `score`.
So it’s clearly aiming for **(B)** eventually.

---

## Mapping `nba_model_spec_v4` to current `perrypicks`

### Current V1 input reality
V1 effectively uses:
- `game_id`
- halftime score
- PBP behavior counts for 1H
- some team rate features in `predict_from_gameid_v2.py` (not currently used by Streamlit)

What’s missing vs v4 schema:
- `timestamp_utc` (easy)
- explicit `team_a/team_b` objects (easy)
- `seconds_remaining_in_period` (easy from clock)
- `box` aggregated stats (we can compute from boxscore JSON)
- `priors` (needs a prior-store keyed by game date)
- `feature_version` (we should introduce)

### Current V1 output reality
V1 returns:
- means via `pred_*`
- intervals as `bands80` lists

What v4 wants:
- `mu` object (team_a, team_b, total, margin)
- `sigma` object (total, margin)
- `pi80` object with `{low, high}` per metric
- `probabilities` computed vs market
- `model_name`, `model_version`, `calibration_id`, `feature_version`

This is totally doable.

---

## Proposed V2 iterations

### V2.0 (MVP) — ship fast, unlock tracking
- Add persistence layer (SQLite) with tables:
  - `games`
  - `bets`
  - `prediction_snapshots`
- Add “Track this bet” button at halftime
- Add 2H monitoring panel:
  - chart p(hit) over time
  - list snapshots
- Convert prediction output into a v4-ish structure (even if priors are empty at first).

### V2.1 — contract enforcement + nicer boundaries
- Actually validate input/output against JSON schema
- Introduce `feature_version`, `model_version`, `calibration_id`
- Break `app.py` into UI components + domain services

### V2.2 — real calibration + backtest gates
- Use your acceptance criteria:
  - interval coverage in [0.78, 0.82]
  - Brier improvements
  - walk-forward leakage tests
- Replace heuristic SDs with calibrated sigma scaling

### V2.3 — multi-game optional
- Home screen: today’s tracked games
- Drill-in: single-game detail (still the main UX)

---

## Decisions needed (small, but important)

1. Persistence backend for Streamlit Cloud:
   - SQLite for MVP, or jump straight to Postgres?
2. Do we treat “post-halftime updates” as:
   - sigma shrink only, or
   - dynamic re-forecast incorporating 2H score/time?
3. Minimum edge thresholds for recommendations (per SPEC.md):
   - `edge_min` default? (e.g., 0.02)
   - `p_min` default? (e.g., 0.60)

