#!/usr/bin/env bash
# Smart Pipeline — runs the full data→predict cycle
# Designed for cron: runs after market close, handles all markets
#
# Schedule (add to crontab -e):
#   # US market close (4 PM ET = 10 PM UTC, Mon-Fri)
#   0 22 * * 1-5 cd /data/github-repos/stock-ml-pipeline && bash scripts/smart_pipeline.sh >> logs/pipeline.log 2>&1
#
#   # MENA market close (3 PM AST = 12 PM UTC, Sun-Thu)
#   0 12 * * 0-4 cd /data/github-repos/stock-ml-pipeline && bash scripts/smart_pipeline.sh --region mena >> logs/pipeline.log 2>&1
#
#   # Weekly full retrain (Saturday 6 AM UTC)
#   0 6 * * 6 cd /data/github-repos/stock-ml-pipeline && bash scripts/smart_pipeline.sh --retrain >> logs/pipeline.log 2>&1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
source .venv/bin/activate

# Create log directory
mkdir -p logs

REGION="${1:---all}"
RETRAIN=false
FORCE=false

for arg in "$@"; do
    case $arg in
        --retrain) RETRAIN=true ;;
        --force) FORCE=true ;;
        --region) shift; REGION="$1" ;;
    esac
done

echo ""
echo "================================================================"
echo "  Smart Pipeline — $(date -Iseconds)"
echo "  Region: $REGION | Retrain: $RETRAIN"
echo "================================================================"
echo ""

# ── Step 1: Update price data ──
echo "[1/7] Updating prices..."
python -m src.collectors.price_collector --mode daily
echo ""

# ── Step 2: Collect news from all sources ──
echo "[2/7] Collecting news (RSS + GDELT)..."
python -m src.collectors.news_collector --source all 2>&1 || echo "  RSS collection had errors (continuing)"
echo ""

# Wait before GDELT to avoid rate limits
sleep 3
python -m src.collectors.gdelt_collector 2>&1 || echo "  GDELT collection had errors (continuing)"
echo ""

# ── Step 3: Score sentiment ──
echo "[3/7] Scoring sentiment..."
python -m src.features.sentiment --score-new 2>&1 || echo "  Sentiment scoring skipped (model not downloaded)"
echo ""

# ── Step 4: Rebuild features ──
echo "[4/7] Rebuilding feature matrix..."
python -m src.features.pipeline 2>&1
echo ""

# ── Step 5: Check drift ──
echo "[5/7] Checking for drift..."
DRIFT_RESULT=$(python -m src.training.drift 2>&1 | tail -1)
echo "  Drift result: $DRIFT_RESULT"
echo ""

# ── Step 6: Retrain if needed ──
if $RETRAIN || $FORCE; then
    echo "[6/7] Retraining (scheduled/forced)..."
    python -m src.training.retrain --force
elif echo "$DRIFT_RESULT" | grep -qi "RETRAIN"; then
    echo "[6/7] Retraining (drift detected)..."
    python -m src.training.retrain
else
    echo "[6/7] No retraining needed — models are current"
fi
echo ""

# ── Step 7: Log predictions ──
echo "[7/7] Generating and logging predictions..."
python -c "
from src.inference.predictor import get_predictor
from src.inference.logger import log_prediction
from src.utils.config import get_ticker_list

predictor = get_predictor()
logged = 0
for ticker in get_ticker_list():
    try:
        result = predictor.predict_ticker(ticker, horizon=5)
        pred = result.get('best_prediction')
        if pred:
            log_prediction(
                ticker=ticker,
                model_name=result.get('best_model', 'unknown'),
                horizon_days=5,
                predicted_direction=pred['direction'],
                predicted_proba=pred['probabilities'],
                confidence=pred['confidence'],
                regime_state=result.get('regime', {}).get('state'),
            )
            logged += 1
    except Exception as e:
        pass

print(f'Logged {logged} predictions to DB')
" 2>&1
echo ""

# ── Step 8: Backfill actuals for past predictions ──
echo "[8/8] Backfilling actuals for past predictions..."
python -c "from src.inference.logger import backfill_actuals; print(f'Updated {backfill_actuals()} predictions with actual returns')" 2>&1
echo ""

echo "================================================================"
echo "  Pipeline complete — $(date -Iseconds)"
echo "================================================================"
