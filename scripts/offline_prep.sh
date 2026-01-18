#!/usr/bin/env bash
set -euo pipefail

# Prep an offline-friendly environment by downloading wheels while you have internet.
# Then you can install with --no-index offline.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="$ROOT_DIR/wheels"

mkdir -p "$WHEEL_DIR"

echo "Downloading wheels into: $WHEEL_DIR"

PY_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="python3"
fi

echo "\n[1/2] Runtime deps (Streamlit app)"
"$PY_BIN" -m pip download -r "$ROOT_DIR/requirements.txt" -d "$WHEEL_DIR"

echo "\n[2/2] Dev deps (backtest-only models)"
"$PY_BIN" -m pip download -r "$ROOT_DIR/requirements-dev.txt" -d "$WHEEL_DIR"

echo "\nDone. Offline install example:"
cat <<'EOF'
python -m pip install --no-index --find-links wheels -r requirements-dev.txt
EOF
