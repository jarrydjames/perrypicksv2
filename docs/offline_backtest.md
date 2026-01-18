# Offline backtesting (laptop-friendly)

Goal:
- Run **Option 1 (quick walk-forward)** for preliminary results.
- Prepare your laptop to run **Option 2 (nested walk-forward tuning)** overnight **without Wi‑Fi**.

## Prereqs
You need these local files:
- `data/processed/halftime_training_23_24_enriched.parquet`
- `data/raw/box/` cached jsons (for `gameTimeUTC` ordering)

## 0) While online: download wheels for offline install
This downloads pip wheels into `./wheels/` so you can install dependencies later without internet.

```bash
# first time only (if needed)
chmod +x scripts/offline_prep.sh

bash scripts/offline_prep.sh
```

## 1) While online: create / update your venv
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip

# install runtime deps
python -m pip install -r requirements.txt

# install dev deps (includes xgboost + catboost)
python -m pip install -r requirements-dev.txt
```

## 2) Option 1: quick walk-forward backtest (prelim)
This is the fast, fixed-hyperparam run.

```bash
.venv/bin/python -m src.modeling.walkforward_backtest --include-xgb --include-cat \
  --train-min 1000 --test-size 200 --step-size 200
```

Output:
- `data/processed/walkforward_backtest.csv`

## 3) Offline install (later, without Wi‑Fi)
If you need to reinstall packages offline:

```bash
. .venv/bin/activate
python -m pip install --no-index --find-links wheels -r requirements-dev.txt
```

## 4) Option 2: overnight (nested walk-forward tuning)
Use the nested runner (adds inner tuning loops for XGBoost/CatBoost) and save results.

```bash
.venv/bin/python -m src.modeling.nested_walkforward_backtest --include-xgb --include-cat \
  --train-min 1000 --test-size 200 --step-size 200 \
  --inner-folds 3 --trials 15
```

Output:
- `data/processed/nested_walkforward_backtest.csv`
