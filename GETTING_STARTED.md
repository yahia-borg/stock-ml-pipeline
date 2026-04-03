# Getting Started — From Clone to Live Dashboard

This guide takes you from zero to a running dashboard with live predictions. Follow each step in order.

**Time estimates:** Setup ~10 min, Data ~20 min, Training ~15 min, Dashboard ~5 min

---

## Prerequisites

Install these before starting:

| Tool | Version | Check | Install |
|------|---------|-------|---------|
| Python | 3.11+ | `python3 --version` | [python.org](https://www.python.org/downloads/) |
| Docker | 20.10+ | `docker --version` | [docker.com](https://docs.docker.com/get-docker/) |
| Node.js | 18+ | `node --version` | [nodejs.org](https://nodejs.org/) |
| Git | any | `git --version` | [git-scm.com](https://git-scm.com/) |
| Make | any | `make --version` | Pre-installed on Linux/Mac |

**Disk space needed:** ~3 GB (deps + data + models)

---

## Step 1: Clone and Setup (5 min)

```bash
# Clone the repo
git clone <your-repo-url>
cd stock-ml-pipeline

# Run the setup script (creates venv, installs deps, starts DB)
bash scripts/setup.sh

# Activate the virtual environment
source .venv/bin/activate

# Verify everything works
make check-health
```

**What happened:**
- Created Python virtual environment at `.venv/`
- Installed CPU PyTorch + all dependencies
- Copied `.env.example` to `.env`
- Started TimescaleDB in Docker (creates all tables automatically)
- Ran health checks

**If you have an NVIDIA GPU:** Run `bash scripts/setup.sh --gpu` instead for CUDA-enabled PyTorch.

---

## Step 2: Get API Keys (2 min)

Edit `.env` with your keys. Only FRED is required for the full pipeline:

```bash
nano .env    # or use any text editor
```

| Key | How to get it | Required? |
|-----|---------------|-----------|
| `FRED_API_KEY` | Go to [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html), create account, get key | Yes (for macro data) |
| `NEWSAPI_KEY` | Go to [newsapi.org/register](https://newsapi.org/register) | Optional (for live news) |

**Without any API keys:** You can still run price-only mode with `make backfill-quick`.

---

## Step 3: Download Historical Data (15-20 min)

```bash
# Option A: Full backfill (prices + macro + news + GDELT) — recommended
make backfill

# Option B: Quick mode (prices only — faster, no API keys needed)
make backfill-quick
```

**What downloads:**
- Price data: 21 tickers + VIX from 2000-today via yfinance (~2 min)
- Macro data: 10 FRED indicators with as-reported vintage values (~1 min)
- News headlines: RSS feeds + NewsAPI (~1 min)
- GDELT events: Global events with tone scoring (~1 min)

Check what you have:
```bash
make data-status
```

---

## Step 4: Score News Sentiment (3 min)

```bash
# Score all news articles with FinBERT
# (downloads ~500MB model on first run, cached after)
make score-sentiment
```

**First run downloads:** `beethogedeon/Modern-FinBERT-large` to `~/.cache/huggingface/`. This is a one-time download.

Test it works:
```bash
make sentiment-test
# Output: Score: -0.450 (negative, confidence: 0.892)
```

Check status:
```bash
make sentiment-status
```

---

## Step 5: Build Feature Matrix (2-5 min)

```bash
make features
```

This builds ~130+ features per ticker:
- Technical indicators (momentum, volatility, microstructure)
- Cross-sectional features (market breadth, sector momentum)
- Calendar features (OPEX, quarter-end, holidays)
- Macro features (yield curve, inflation, fed rate regime)
- VIX features (variance risk premium)
- Sentiment features (aggregated daily scores)
- Targets (adaptive volatility-based labels for 1d, 5d, 20d)

Verify:
```bash
make feature-status
# Output: Rows: ~130,000+, Columns: ~150+, Tickers: 21
```

---

## Step 6: Train Models (5-10 min)

```bash
# 1. Fit the HMM regime detector (instant)
make train-regime

# 2. Train XGBoost + LightGBM baselines (2-5 min)
make train-baseline

# 3. Check what's trained
make train-status
```

You'll see a comparison table at the end:
```
BASELINE COMPARISON
Model                     Acc   DirAcc   Sharpe       F1
lgbm_1d               0.XXXX   0.XXXX   0.XXXX   0.XXXX
lgbm_5d               0.XXXX   0.XXXX   0.XXXX   0.XXXX
xgboost_5d            0.XXXX   0.XXXX   0.XXXX   0.XXXX
...
```

**Optional: Evaluate Chronos-2 zero-shot (adds ~800MB download first time):**
```bash
make eval-chronos
```

**Optional: Train TFT (slower, GPU recommended):**
```bash
make train-tft-quick    # quick test: 5 epochs, ~5 min on CPU
make train-tft          # full training: 50 epochs, GPU recommended
```

**Optional: Train ensemble over all models:**
```bash
make train-ensemble
```

Compare all models:
```bash
make compare-models
```

---

## Step 7: Start the API Server (instant)

```bash
# In terminal 1: start the API server
make run
```

The API is now running at **http://localhost:8099**.

Test it:
```bash
# In terminal 2:
curl http://localhost:8099/health
# {"status":"ok","timestamp":"...","models_loaded":6}

curl http://localhost:8099/api/predictions/AAPL
# {"ticker":"AAPL","horizon":5,"predictions":{...},"regime":{...}}

curl http://localhost:8099/api/pipeline/status
# {"database_connected":true,"tables":{"prices":130000,...},...}
```

API endpoints:
| Endpoint | Description |
|----------|-------------|
| `GET /health` | API health check |
| `GET /api/predictions/latest` | Predictions for all tickers |
| `GET /api/predictions/{ticker}` | Prediction for one ticker |
| `GET /api/prices/{ticker}?days=365` | OHLCV price data |
| `GET /api/models/results` | Model evaluation metrics |
| `GET /api/sentiment/latest` | Recent sentiment scores |
| `GET /api/pipeline/status` | Pipeline health status |

---

## Step 8: Start the Dashboard (2 min)

```bash
# In terminal 3: install dashboard deps (first time only)
make dashboard-install

# Start the dashboard dev server
make dashboard
```

Open **http://localhost:3000** in your browser.

**Dashboard pages:**
- **Overview** — KPI cards, prediction table for all tickers with direction/confidence/regime
- **Models** — Bar charts comparing accuracy/Sharpe, per-model metric cards
- **Sentiment** — Live news feed with FinBERT scores, bullish/bearish counts
- **Pipeline** — Database status, row counts, last update times, quick commands

---

## Summary — What's Running

After completing all steps, you have 3 services running:

| Service | URL | Terminal |
|---------|-----|----------|
| TimescaleDB | localhost:5432 | Docker (background) |
| FastAPI | localhost:8099 | Terminal 1 |
| Dashboard | localhost:3000 | Terminal 3 |

---

## Daily Operations

Once set up, your daily workflow is:

```bash
# 1. Update data (run after market close, ~2 min)
make daily-update

# 2. Score new headlines (~1 min)
make score-sentiment

# 3. Rebuild features (~2 min)
make features

# 4. (Optional) Retrain models if needed
make train-baseline
```

Or automate it all via cron:
```bash
# Add to crontab: crontab -e
# Run daily at 6 PM ET (after market close)
0 18 * * 1-5 cd /path/to/stock-ml-pipeline && source .venv/bin/activate && bash scripts/daily_update.sh
```

---

## Sharing with Friends

For friends to set up their own copy:

```bash
git clone <your-repo-url>
cd stock-ml-pipeline
bash scripts/setup.sh      # one command — handles everything
source .venv/bin/activate
make backfill-quick         # get price data
make features               # build features
make train-baseline         # train models
make run                    # start API (terminal 1)
make dashboard-install && make dashboard  # start dashboard (terminal 2)
# Open http://localhost:3000
```

Total time: ~20 minutes from clone to live dashboard.

---

## Troubleshooting

**"No models yet" in dashboard:**
```bash
make train-baseline    # trains XGBoost + LightGBM
```

**"No feature data" errors:**
```bash
make backfill-quick    # get price data first
make features          # then build features
```

**Database not connecting:**
```bash
make db               # start/restart TimescaleDB
make check-health     # verify connection
```

**Port already in use:**
```bash
# API on different port:
uvicorn src.inference.api:app --port 8100

# Dashboard on different port:
cd dashboard && npm run dev -- --port 3001
```

**Low disk space (model downloads):**
```bash
# Check HuggingFace cache size:
du -sh ~/.cache/huggingface/

# Move cache to external drive:
echo "HF_HOME=/mnt/external/hf_cache" >> .env
```
