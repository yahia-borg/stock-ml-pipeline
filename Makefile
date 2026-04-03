.PHONY: help setup setup-gpu db db-stop db-reset install install-gpu test lint run api clean check-health \
       backfill backfill-quick daily-update collect-prices collect-macro collect-news collect-gdelt data-status \
       features features-single feature-status \
       score-sentiment score-sentiment-llm sentiment-status \
       train-baseline train-lgbm train-xgboost train-regime train-status \
       eval-chronos eval-moirai compare-models \
       train-tft train-tft-quick train-ensemble \
       dashboard dashboard-install \
       monitor drift-check retrain retrain-force backfill-actuals

VENV       := .venv
PYTHON     := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
SHELL      := /bin/bash

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# First-time setup (CPU — default)
# ──────────────────────────────────────────────
setup: ## Full setup: venv + deps (CPU) + .env + database
	@echo "==> Creating virtual environment..."
	python3.11 -m venv $(VENV) 2>/dev/null || python3 -m venv $(VENV)
	@echo "==> Installing CPU PyTorch..."
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	@echo "==> Installing project + dev deps..."
	$(PIP) install -e ".[dl,dev]"
	@echo "==> Copying .env..."
	@test -f .env || cp .env.example .env
	@echo "==> Starting database..."
	$(MAKE) db
	@echo ""
	@echo "========================================"
	@echo "  Setup complete!"
	@echo "  Activate venv:  source .venv/bin/activate"
	@echo "  Check health:   make check-health"
	@echo "========================================"

# ──────────────────────────────────────────────
# First-time setup (GPU — NVIDIA CUDA)
# ──────────────────────────────────────────────
setup-gpu: ## Full setup with GPU PyTorch (needs NVIDIA GPU + CUDA)
	@echo "==> Creating virtual environment..."
	python3.11 -m venv $(VENV) 2>/dev/null || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	@echo "==> Installing GPU PyTorch (this is a large download)..."
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu124
	@echo "==> Installing project + dev deps..."
	$(PIP) install -e ".[dl,dev]"
	@test -f .env || cp .env.example .env
	$(MAKE) db
	@echo ""
	@echo "========================================"
	@echo "  GPU Setup complete!"
	@echo "  Activate venv:  source .venv/bin/activate"
	@echo "========================================"

# ──────────────────────────────────────────────
# Install deps only (no DB, no .env)
# ──────────────────────────────────────────────
install: ## Install deps into existing venv (CPU)
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -e ".[dl,dev]"

install-gpu: ## Install deps into existing venv (GPU)
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu124
	$(PIP) install -e ".[dl,dev]"

# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────
db: ## Start TimescaleDB (creates tables on first run)
	docker compose up -d timescaledb
	@echo "Waiting for database to be ready..."
	@until docker compose exec timescaledb pg_isready -U postgres -q 2>/dev/null; do sleep 1; done
	@echo "Database is ready on localhost:$${DB_PORT:-5432}"

db-stop: ## Stop TimescaleDB (data is preserved)
	docker compose stop timescaledb

db-reset: ## Destroy database and start fresh (WARNING: deletes all data)
	@echo "This will DELETE all data. Press Ctrl+C to cancel, Enter to continue."
	@read _confirm
	docker compose down -v
	$(MAKE) db

# ──────────────────────────────────────────────
# Development
# ──────────────────────────────────────────────
test: ## Run tests
	$(PYTHON) -m pytest tests/ -v --tb=short

lint: ## Run linter
	$(VENV)/bin/ruff check src/ tests/
	$(VENV)/bin/ruff format --check src/ tests/

lint-fix: ## Auto-fix lint issues
	$(VENV)/bin/ruff check --fix src/ tests/
	$(VENV)/bin/ruff format src/ tests/

# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
run: ## Run FastAPI server locally (for development)
	$(PYTHON) -m uvicorn src.inference.api:app --host 0.0.0.0 --port 8099 --reload

api: ## Start API in Docker (builds image if needed)
	docker compose --profile api up -d --build

# ──────────────────────────────────────────────
# Data collection (Phase 1)
# ──────────────────────────────────────────────
backfill: ## Full historical backfill (2000-today): prices + macro + news + GDELT
	bash scripts/backfill_history.sh

backfill-quick: ## Quick backfill (prices only — for testing)
	bash scripts/backfill_history.sh --quick

daily-update: ## Daily data update (prices + news + GDELT)
	bash scripts/daily_update.sh

collect-prices: ## Download price data only
	$(PYTHON) -m src.collectors.price_collector --mode backfill

collect-macro: ## Download macro data from FRED only
	$(PYTHON) -m src.collectors.macro_collector

collect-news: ## Collect current news headlines (RSS + NewsAPI)
	$(PYTHON) -m src.collectors.news_collector

collect-gdelt: ## Fetch GDELT global events
	$(PYTHON) -m src.collectors.gdelt_collector

collect-fundamentals: ## Fetch company fundamentals from yfinance (quarterly)
	$(PYTHON) -m src.collectors.fundamentals_collector

data-status: ## Show data collection status (row counts)
	@echo "==> Data status:"
	@$(PYTHON) -c "\
from src.utils.db import read_sql, check_connection; \
tables = ['prices', 'macro_indicators', 'news_articles', 'gdelt_events']; \
[print(f'  {t}: {read_sql(f\"SELECT COUNT(*) as n FROM {t}\")[\"n\"].iloc[0]:,} rows') for t in tables] \
if check_connection() else print('  DB not connected')"
	@echo "==> Parquet files:"
	@find data/raw -name "*.parquet" -exec ls -lh {} \; 2>/dev/null || echo "  No parquet files yet"

# ──────────────────────────────────────────────
# Feature engineering (Phase 2)
# ──────────────────────────────────────────────
features: ## Build full feature matrix for all tickers
	$(PYTHON) -m src.features.pipeline

features-single: ## Build features for one ticker (usage: make features-single TICKER=AAPL)
	$(PYTHON) -m src.features.pipeline --ticker $(TICKER) --skip-cross

feature-status: ## Show feature matrix summary
	@$(PYTHON) -c "\
import pandas as pd; from pathlib import Path; \
p = Path('data/processed/feature_matrix.parquet'); \
print(f'Feature matrix: {p}') if p.exists() else print('No feature matrix yet — run: make features'); \
df = pd.read_parquet(p) if p.exists() else None; \
print(f'  Rows: {len(df):,}') if df is not None else None; \
print(f'  Columns: {len(df.columns)}') if df is not None else None; \
print(f'  Tickers: {df[\"ticker\"].nunique() if \"ticker\" in df.columns else \"N/A\"}') if df is not None else None; \
print(f'  Date range: {df.index.min().date()} to {df.index.max().date()}') if df is not None else None"

# ──────────────────────────────────────────────
# Sentiment pipeline (Phase 3)
# ──────────────────────────────────────────────
score-sentiment: ## Score unscored news articles with FinBERT
	$(PYTHON) -m src.features.sentiment --score-new

score-sentiment-llm: ## Score articles with LLM (requires vLLM running)
	$(PYTHON) -c "from src.features.sentiment_llm import score_unscored_in_db; print(f'Scored {score_unscored_in_db()} articles')"

sentiment-status: ## Show sentiment scoring status
	@$(PYTHON) -c "\
from src.utils.db import read_sql, check_connection; \
print('Sentiment status:') if check_connection() else print('DB not connected'); \
r = read_sql('SELECT COUNT(*) as total, COUNT(sentiment_finbert) as finbert, COUNT(sentiment_llm) as llm FROM news_articles') if check_connection() else None; \
print(f'  Total articles: {r[\"total\"].iloc[0]:,}') if r is not None else None; \
print(f'  FinBERT scored: {r[\"finbert\"].iloc[0]:,}') if r is not None else None; \
print(f'  LLM scored:     {r[\"llm\"].iloc[0]:,}') if r is not None else None; \
print(f'  Unscored:       {r[\"total\"].iloc[0] - r[\"finbert\"].iloc[0]:,}') if r is not None else None"

# ──────────────────────────────────────────────
# Training (Phase 4)
# ──────────────────────────────────────────────
train-baseline: ## Train all baseline models (XGBoost + LightGBM, all horizons)
	$(PYTHON) -m src.training.train --model all

train-lgbm: ## Train LightGBM only (all horizons)
	$(PYTHON) -m src.training.train --model lgbm

train-xgboost: ## Train XGBoost only (all horizons)
	$(PYTHON) -m src.training.train --model xgboost

train-regime: ## Fit HMM regime detector on SPY
	$(PYTHON) -m src.models.regime --ticker SPY

train-status: ## Show trained models
	@echo "==> Trained models:"
	@ls -lh models/*.joblib 2>/dev/null || echo "  No models yet — run: make train-baseline"
	@echo "==> Results files:"
	@ls -lh models/*_results.json 2>/dev/null || echo "  No results yet"

# ──────────────────────────────────────────────
# Foundation models (Phase 5)
# ──────────────────────────────────────────────
eval-chronos: ## Evaluate Chronos-2 zero-shot (downloads ~800MB on first run)
	$(PYTHON) -m src.training.train_foundation --model chronos2

eval-moirai: ## Evaluate Moirai 2.0 zero-shot (requires uni2ts)
	$(PYTHON) -m src.training.train_foundation --model moirai2

compare-models: ## Compare all trained/evaluated models side by side
	$(PYTHON) -m src.training.train_foundation --compare --horizon 5

# ──────────────────────────────────────────────
# TFT + Ensemble (Phase 6)
# ──────────────────────────────────────────────
train-tft: ## Train Temporal Fusion Transformer (5d horizon, full)
	$(PYTHON) -m src.training.train_tft --horizon 5

train-tft-quick: ## Quick TFT training (5 epochs, 2 folds — for testing)
	$(PYTHON) -m src.training.train_tft --horizon 5 --quick --no-wandb

train-ensemble: ## Train stacking ensemble over all models
	$(PYTHON) -m src.models.ensemble --horizon 5

# ──────────────────────────────────────────────
# Dashboard (Phase 7)
# ──────────────────────────────────────────────
dashboard-install: ## Install dashboard dependencies (first time only)
	cd dashboard && npm install

dashboard: ## Start dashboard dev server (http://localhost:3000)
	cd dashboard && npm run dev

# ──────────────────────────────────────────────
# Monitoring & Retraining (Phase 8)
# ──────────────────────────────────────────────
monitor: ## Full drift check (performance + data + regime)
	$(PYTHON) -m src.training.drift

drift-check: ## Quick performance drift check only
	$(PYTHON) -m src.training.drift --perf-only

retrain: ## Retrain if drift detected (check first, then train if needed)
	$(PYTHON) -m src.training.retrain

retrain-force: ## Force retrain all models regardless of drift
	$(PYTHON) -m src.training.retrain --force

smart-pipeline: ## Run the full smart pipeline (data + features + drift + predict)
	bash scripts/smart_pipeline.sh

smart-pipeline-retrain: ## Run smart pipeline with forced retraining
	bash scripts/smart_pipeline.sh --retrain

collect-events: ## Collect commodity, FX, and regional index event signals
	$(PYTHON) -m src.collectors.events_collector

backfill-actuals: ## Fill in actual returns for past predictions
	$(PYTHON) -c "from src.inference.logger import backfill_actuals; print(f'Updated {backfill_actuals()} predictions')"

sentiment-test: ## Test sentiment model with a sample headline
	$(PYTHON) -m src.features.sentiment --test "Federal Reserve raises interest rates by 25 basis points"

# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────
check-health: ## Verify database connection and project setup
	@echo "==> Checking Python..."
	@$(PYTHON) --version
	@echo "==> Checking database..."
	@$(PYTHON) -c "from src.utils.db import check_connection; print('DB OK' if check_connection() else 'DB FAILED')"
	@echo "==> Checking imports..."
	@$(PYTHON) -c "from src.utils.config import get_ticker_list; print(f'Tickers loaded: {len(get_ticker_list())}')"
	@echo "==> All checks passed."

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# Docker Production
# ──────────────────────────────────────────────
docker-build: ## Build all Docker images
	docker compose --profile full build

docker-up: ## Start full stack (DB + API + Dashboard)
	docker compose --profile full up -d

docker-api: ## Start DB + API only
	docker compose --profile api up -d

docker-worker: ## Run pipeline worker once (data + features + train)
	docker compose --profile worker run --rm worker

docker-down: ## Stop all containers
	docker compose --profile full down

docker-logs: ## Tail logs from all services
	docker compose --profile full logs -f

docker-size: ## Show image sizes
	@docker images | grep -E "stock-|timescale" | awk '{printf "  %-40s %s\n", $$1":"$$2, $$7" "$$8}'

clean: ## Remove venv, caches, logs (keeps data and DB)
	rm -rf $(VENV) __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f pipeline.log
