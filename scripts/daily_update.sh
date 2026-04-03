#!/usr/bin/env bash
# Daily data update — run via cron at market close (e.g., 6:00 PM ET)
#
# Crontab entry (Eastern Time):
#   0 18 * * 1-5 cd /path/to/stock-ml-pipeline && bash scripts/daily_update.sh >> logs/daily.log 2>&1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
fi

# Create log directory
mkdir -p logs

echo "=== Daily update $(date -Iseconds) ==="

# ── 1. Price data (last 5 days to catch gaps) ──
echo "[1/3] Updating prices..."
python -m src.collectors.price_collector --mode daily

# ── 2. News + Sentiment ──
echo "[2/3] Collecting news..."
python -m src.collectors.news_collector --source all

# ── 3. GDELT events ──
echo "[3/3] Fetching GDELT events..."
python -m src.collectors.gdelt_collector

echo "=== Daily update complete $(date -Iseconds) ==="
