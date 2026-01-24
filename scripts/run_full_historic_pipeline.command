#!/bin/bash
set -euo pipefail

# Double-click me. I dare you.
#
# Full one-time pipeline:
# 1) Pull OddsPortal priors (bet365) for multiple seasons
# 2) Join priors -> NBA game_id using your existing box cache
# 3) Merge to one JSONL file

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "Running from: $REPO_ROOT"

# Bootstrap a local venv for the one-time scrape tooling.
# This avoids polluting your global python and ensures playwright is available.
if [ -f ".venv/bin/python" ]; then
  VENV_PY_VER=$(".venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [ "$VENV_PY_VER" != "3.12" ] && [ "$VENV_PY_VER" != "3.13" ] && [ "$VENV_PY_VER" != "3.14" ]; then
    echo "Existing .venv uses Python $VENV_PY_VER (<3.12). Recreating venv..."
    rm -rf .venv
  fi
fi

if [ ! -f ".venv/bin/activate" ]; then
  echo "Creating venv: .venv (requires Python >= 3.12 for OddsHarvester)"
  /usr/local/bin/python3 -m venv .venv
fi

echo "Activating venv: .venv"
source ".venv/bin/activate"

echo "Upgrading pip"
python -m pip install --upgrade pip >/dev/null

echo "Installing historic scrape deps (including OddsHarvester editable install)"
python -m pip install -r requirements-historic.txt >/dev/null

# Install OddsHarvester + its pinned deps into this venv
python -m pip install -e "/Users/jarrydhawley/Desktop/Predictor/OddsHarvester" >/dev/null

echo "Ensuring Playwright Chromium is installed (may take a minute first time)"
python -m playwright install chromium >/dev/null 2>&1 || true

# --- EDIT THESE IF NEEDED ---
ODDS_HARVESTER_DIR="/Users/jarrydhawley/Desktop/Predictor/OddsHarvester"
NBA_BOX_DIR="/Users/jarrydhawley/Desktop/Predictor/perrypicks/data/raw/box"

# Seasons you want priors for (must match OddsPortal URL season slug)
SEASONS="2023-2024,2024-2025"

# Upper bound: script stops early if pages stop returning new links
PAGES=60
# ----------------------------

python3 scripts/run_full_historic_pipeline.py \
  --odds-harvester "$ODDS_HARVESTER_DIR" \
  --box-dir "$NBA_BOX_DIR" \
  --seasons "$SEASONS" \
  --pages "$PAGES" \
  --headless

echo "\nDone. Your final merged file is:"
echo "  $REPO_ROOT/data/oddsportal/priors_by_game_id_ALL.jsonl"

echo "\nPress Enter to close..."
read
