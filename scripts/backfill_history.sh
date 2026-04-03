#!/usr/bin/env bash
# One-time historical backfill — run after initial setup
# Downloads all data from 2000-01-01 to today
#
# Usage:
#   bash scripts/backfill_history.sh           # full backfill
#   bash scripts/backfill_history.sh --quick   # prices only (for testing)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if not already active
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    source .venv/bin/activate
fi

QUICK_MODE=false
if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=true
fi

echo "========================================="
echo "  Stock Pipeline — Historical Backfill"
echo "  Start date: 2000-01-01"
echo "  Mode: $(if $QUICK_MODE; then echo 'QUICK (prices only)'; else echo 'FULL'; fi)"
echo "========================================="
echo ""

# ── 1. Price data (yfinance — no API key needed) ──
echo "[1/5] Backfilling price data (21 tickers, 2000-today)..."
python -m src.collectors.price_collector --mode backfill
echo ""

if $QUICK_MODE; then
    echo "Quick mode — skipping macro, news, and GDELT."
    echo "Done! Run without --quick for full backfill."
    exit 0
fi

# ── 2. Macro data (FRED — requires FRED_API_KEY) ──
echo "[2/5] Fetching macro indicators from FRED..."
if python -m src.collectors.macro_collector 2>&1; then
    echo "  Macro data complete."
else
    echo "  WARNING: Macro fetch failed (check FRED_API_KEY in .env). Continuing..."
fi
echo ""

# ── 3. Historical news (FNSPID from HuggingFace — large download) ──
echo "[3/5] Downloading historical news dataset (FNSPID, 2000-2023)..."
echo "  This is a large download on first run. It will be cached for future use."
if python -m src.collectors.historical_news_collector 2>&1; then
    echo "  Historical news complete."
else
    echo "  WARNING: Historical news download failed. Continuing..."
fi
echo ""

# ── 4. Recent news (RSS + NewsAPI) ──
echo "[4/5] Collecting recent news headlines..."
python -m src.collectors.news_collector --source all
echo ""

# ── 5. GDELT events ──
echo "[5/5] Fetching GDELT global events..."
python -m src.collectors.gdelt_collector
echo ""

echo "========================================="
echo "  Backfill complete!"
echo ""
echo "  Verify with:"
echo "    make check-health"
echo "    python -c \"from src.utils.db import read_sql; print(read_sql('SELECT COUNT(*) FROM prices'))\""
echo "========================================="
