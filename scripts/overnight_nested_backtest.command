#!/usr/bin/env bash
set -euo pipefail

# Double-clickable Mac script. Keeps the machine awake while running.
# Runs the nested walk-forward backtest (Option 2) and logs output.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate

TS="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nested_backtest_${TS}.log"

DATA_PATH="data/processed/halftime_training_23_24_enriched.parquet"
BOX_DIR="data/raw/box"
OUT_CSV=${OUT_CSV:-"data/processed/nested_walkforward_backtest.csv"}

TRAIN_MIN=${TRAIN_MIN:-1000}
TEST_SIZE=${TEST_SIZE:-200}
STEP_SIZE=${STEP_SIZE:-200}
INNER_FOLDS=${INNER_FOLDS:-3}
TRIALS=${TRIALS:-15}
SEED=${SEED:-0}

echo "Writing log to: $LOG_FILE"
echo "Output CSV: $OUT_CSV" | tee -a "$LOG_FILE"

# --- Preflight checks (fail fast, don't waste a night) ---
if [[ ! -d .venv ]]; then
  echo "[error] .venv missing. Run: uv venv --python 3.12 .venv" | tee -a "$LOG_FILE"
  exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "[error] data file missing: $DATA_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

if [[ ! -d "$BOX_DIR" ]]; then
  echo "[error] box dir missing: $BOX_DIR" | tee -a "$LOG_FILE"
  exit 1
fi

if ! command -v caffeinate >/dev/null 2>&1; then
  echo "[error] caffeinate not found (macOS only)." | tee -a "$LOG_FILE"
  exit 1
fi

if ! .venv/bin/python -c "import pandas, numpy, sklearn" >/dev/null 2>&1; then
  echo "[error] core deps not importable in .venv" | tee -a "$LOG_FILE"
  exit 1
fi

if ! .venv/bin/python -c "import xgboost" >/dev/null 2>&1; then
  echo "[warn] xgboost not importable; removing --include-xgb" | tee -a "$LOG_FILE"
  INCLUDE_FLAGS="--include-cat"
fi

INCLUDE_FLAGS="--include-xgb --include-cat"

# If catboost isn't installed, don't waste the night dying at import time.
if ! .venv/bin/python -c "import catboost" >/dev/null 2>&1; then
  echo "[warn] catboost not importable; running overnight with XGBoost only" | tee -a "$LOG_FILE"
  INCLUDE_FLAGS="--include-xgb"
fi

# Keep Mac awake while command runs
caffeinate -dimsu bash -c "\
  .venv/bin/python -u -m src.modeling.nested_walkforward_backtest \
    --data \"$DATA_PATH\" \
    --box-dir \"$BOX_DIR\" \
    --out \"$OUT_CSV\" \
    --train-min $TRAIN_MIN --test-size $TEST_SIZE --step-size $STEP_SIZE \
    --inner-folds $INNER_FOLDS --trials $TRIALS --seed $SEED \
    $INCLUDE_FLAGS \
  2>&1 | tee -a \"$LOG_FILE\"\
"

echo "\nDone. Results: $OUT_CSV" | tee -a "$LOG_FILE"
