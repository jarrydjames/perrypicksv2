#!/bin/bash
set -euo pipefail

# One-click launcher for the local one-time OddsPortal historical pull.
#
# Notes:
# - This runs headless.
# - It writes a JSONL priors DB for bet365.us.
# - You can safely rerun with --resume if it gets interrupted.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "Running from: $REPO_ROOT"

# Optional venv activation if you use one.
if [ -f ".venv/bin/activate" ]; then
  echo "Activating venv: .venv"
  source ".venv/bin/activate"
fi

ODDS_HARVESTER_DIR="/Users/jarrydhawley/Desktop/Predictor/OddsHarvester"
SEASON="2023-2024"
PAGES=28
BATCH_SIZE=25
OUT_FILE="data/oddsportal/historic_bet365_${SEASON}.jsonl"

python3 scripts/oddsportal_historic_pull.py \
  --odds-harvester "$ODDS_HARVESTER_DIR" \
  --season "$SEASON" \
  --pages "$PAGES" \
  --batch-size "$BATCH_SIZE" \
  --out "$OUT_FILE" \
  --headless \
  --resume

echo "\nDone. Output: $REPO_ROOT/$OUT_FILE"

echo "\nPress Enter to close..."
read
