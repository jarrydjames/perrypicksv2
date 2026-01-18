#!/usr/bin/env bash
set -euo pipefail

# One-time-ish setup for local dev/backtests.
# Installs runtime + dev deps into .venv.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY_BIN="${PY_BIN:-python3}"

if [[ ! -d .venv ]]; then
  echo "Creating venv (.venv)..."
  "$PY_BIN" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Some Python builds create venvs without pip. Cool cool cool.
if ! python -m pip --version >/dev/null 2>&1; then
  echo "Bootstrapping pip via ensurepip..."
  python -m ensurepip --upgrade
fi

python -m pip install -U pip wheel

echo "Installing runtime deps..."
python -m pip install -r requirements.txt

echo "Installing dev deps (xgboost/catboost)..."
python -m pip install -r requirements-dev.txt

echo "Sanity check imports..."
python - <<'PY'
import xgboost, catboost
print('xgboost', xgboost.__version__)
print('catboost', catboost.__version__)
PY

echo "Done."
