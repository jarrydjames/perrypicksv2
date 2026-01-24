# OddsPortal halftime odds feasibility spike

This repo can model 2H outcomes and backtest recommendations.

Before we build a full in-play **halftime pricing** backtest, we must confirm:

> Does OddsPortal (via OddsHarvester) expose **odds-history timestamps during the game**?

If the answer is no, we cannot reconstruct a reliable “price at halftime” snapshot after the fact.

## Run the spike

From the repo root:

```bash
.venv/bin/python scripts/oddsportal_halftime_spike.py \
  --odds-harvester /Users/jarrydhawley/Desktop/Predictor/OddsHarvester \
  --season 2023-2024 \
  --max-pages 1 \
  --out data/oddsportal/spike_23_24.json \
  --headless
```

Output:
- `data/oddsportal/spike_23_24.json` (summary + one example match payload)
- `data/oddsportal/spike_23_24.raw.json` (raw OddsHarvester output)

## What success looks like

Minimum success:
- `matches_with_any_history > 0`
- `unique_history_timestamps` is reasonably large (not just 1–2 opening entries)

**Not proven yet** (next phase):
- timestamps that align with an NBA game’s halftime moment
- ability to identify the associated **line** (not just odds)

If the spike is promising, next step is to join scraped OddsPortal matches to NBA `game_id`s
and verify timestamps fall inside a computed halftime window for each game.
