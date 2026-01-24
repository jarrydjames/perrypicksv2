# Training (V2) — two seasons

This repo now includes a **model stack** per the framework docs:
- Ridge (baseline anchor)
- Random Forest (nonlinear)
- Gradient Boosted Trees (HistGradientBoosting)

Each model trains **two heads**:
- `h2_total`
- `h2_margin`

## 1) Compile training data (two seasons)

### Pick your seasons
NBA game IDs embed the season as `002YYxxxxx`.
Use `YY` values for the two seasons you want.

Examples:
- 2023-24 season => `YY = 23`
- 2024-25 season => `YY = 24`
- 2025-26 season => `YY = 25`

### Command
From repo root:

```bash
python -m src.data.compiler \
  --season-a 23 --season-b 24 \
  --cache-box data/raw/box \
  --cache-pbp data/raw/pbp \
  --out data/processed/halftime_training_23_24.parquet
```

Output:
- training parquet: `data/processed/halftime_training_23_24.parquet`
- game id list: `data/processed/game_ids_2023_2024.json` (name varies)

## 2) (Optional) Enrich training data with pace/PPP features

Once compilation is done and you have cached raw data under:
- `data/raw/box/`
- `data/raw/pbp/`

…you can enrich the training parquet with first-half possessions + PPP features:

```bash
.venv/bin/python -m src.data.enrich_training_data \
  --in  data/processed/halftime_training_23_24.parquet \
  --out data/processed/halftime_training_23_24_enriched.parquet
```

## 3) Quick sanity check: priors coverage

```bash
.venv/bin/python scripts/report_priors_coverage.py \
  --parquet data/processed/halftime_training_23_24_enriched.parquet
```

## 4) Train models

```bash
python -m src.modeling.train_models --data data/processed/halftime_training_23_24_enriched.parquet --out-dir models_v2
```

Outputs:
- `models_v2/ridge_twohead.joblib`
- `models_v2/random_forest_twohead.joblib`
- `models_v2/gbt_twohead.joblib`

## Notes / caveats

- The compiler is **resume-safe** via raw JSON caches:
  - `data/raw/box/{gid}.json`
  - `data/raw/pbp/{gid}.json`
  If you rerun, it will reuse cached files and only fetch missing games.

- The schedule endpoint used is `scheduleLeagueV2.json`.
  Depending on NBA availability, it may not expose older seasons.
  If you hit that limitation, run compilation with a precomputed game list:

  ```bash
  python -m src.data.compiler --season-a 23 --season-b 24 --game-ids /path/to/game_ids.json --out ...
  ```
