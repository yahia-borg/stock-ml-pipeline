# Stock Forecasting ML Pipeline — Technical Specification

> Version: 1.0.0 | Date: 2026-03-31 | Status: ALL PHASES COMPLETE (0-8)

## 1. Project Overview

### Goal
Predict stock price direction (up/flat/down) and magnitude at 1-day, 5-day, and 20-day horizons using an ensemble of classical ML, foundation models, and deep learning, informed by technical indicators, macroeconomic data, news sentiment, and market regime detection.

### Stack
- **Language:** Python 3.11+
- **ML:** PyTorch, LightGBM, XGBoost, HMMLearn, Amazon Chronos-2, Salesforce Moirai 2.0
- **NLP:** HuggingFace Transformers (Modern-FinBERT-large), OpenAI-compatible vLLM
- **Database:** PostgreSQL 16 + TimescaleDB (time-series hypertables)
- **API:** FastAPI + Uvicorn
- **Infrastructure:** Docker Compose, Make
- **Frontend (planned):** Next.js 15, shadcn/ui, Recharts, TradingView Lightweight Charts
- **MLOps:** Weights & Biases, MLflow, Evidently (drift detection)

### Data Coverage
- **Historical data from:** 2000-01-01 to present
- **Ticker universe:** 21 tickers (10 US mega-cap stocks, 6 index/commodity ETFs, 5 sector ETFs) + VIX
- **Macro indicators:** 10 FRED series with ALFRED vintage data (as-reported, no look-ahead)
- **News:** FNSPID historical dataset (1999-2023, 15.7M records), live RSS feeds, NewsAPI, GDELT 2.0

---

## 2. Architecture

### Directory Structure

```
stock-ml-pipeline/
├── configs/                         # YAML configuration (single source of truth)
│   ├── tickers.yaml                 #   Ticker universe, macro indicators, GDELT keywords
│   ├── data_config.yaml             #   Data pipeline, validation, feature engineering settings
│   └── model_config.yaml            #   All model hyperparameters
├── data/
│   ├── raw/prices/                  #   Per-ticker parquet files (OHLCV)
│   ├── raw/news/                    #   News parquet files (daily + historical)
│   ├── raw/macro/                   #   Macro indicators parquet
│   ├── processed/                   #   Feature matrices (per-ticker + combined)
│   └── splits/                      #   Walk-forward fold definitions
├── models/                          #   Trained model artifacts (.joblib, .pt, results JSON)
│   └── scalers/                     #   Per-fold normalization scalers
├── src/
│   ├── collectors/                  #   Data ingestion (5 collectors)
│   ├── features/                    #   Feature engineering (10 modules)
│   ├── models/                      #   Model definitions (3 modules)
│   ├── training/                    #   Training and evaluation (4 modules)
│   ├── inference/                   #   Prediction serving (planned)
│   └── utils/                       #   Config, logging, database helpers
├── scripts/                         #   Shell scripts (setup, backfill, daily update, DB bootstrap)
├── notebooks/                       #   EDA and experimentation (planned)
├── tests/                           #   Test suite (planned)
├── docker-compose.yml               #   TimescaleDB + API containers
├── Dockerfile                       #   API container with cached dependency layer
├── Makefile                         #   All common commands
├── pyproject.toml                   #   Python project config with optional dep groups
└── .env.example                     #   Environment variables template
```

### Data Flow

```
Raw Sources ──→ Collectors ──→ Raw Parquet + TimescaleDB
                                    │
                              Feature Pipeline
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              Technical        Macro + VIX     Sentiment
              (60+ cols)       (25+ cols)      (20+ cols)
                    │               │               │
                    └───────┬───────┘               │
                            │                       │
                    Calendar + Cross-sectional       │
                      (15 + 20 cols)                │
                            │                       │
                            └───────┬───────────────┘
                                    │
                            Feature Matrix
                          (~130+ features/ticker)
                                    │
                         ┌──────────┼──────────┐
                         │          │          │
                    Regime HMM   Baselines  Foundation
                   (bull/bear)  (XGB/LGBM)  (Chronos-2)
                         │          │          │
                         └──────────┼──────────┘
                                    │
                              Ensemble (planned)
                                    │
                            Predictions + API
```

---

## 3. Implemented Phases

### Phase 0 — Project Setup (Complete)
**Files:** `pyproject.toml`, `Makefile`, `docker-compose.yml`, `Dockerfile`, `.env.example`, `.gitignore`, `scripts/setup.sh`, `scripts/bootstrap_db.sh`, `scripts/bootstrap_db.sql`, `src/utils/config.py`, `src/utils/logger.py`, `src/utils/db.py`, `README.md`

- Python package with optional dependency groups (CPU/GPU PyTorch, deep learning, Moirai)
- Docker Compose with TimescaleDB (auto-initializes schema on first run) and optional API service
- Dockerfile with cached dependency layer (no re-download on code changes)
- Makefile with `make setup` one-command first-run experience
- Central config loader with env var resolution (`${VAR:-default}`)
- Rich-formatted structured logging to console + file
- PostgreSQL connection pool with session management and idempotent upsert helper

### Phase 1 — Data Infrastructure (Complete)
**Files:** `src/collectors/price_collector.py`, `src/collectors/macro_collector.py`, `src/collectors/news_collector.py`, `src/collectors/gdelt_collector.py`, `src/collectors/historical_news_collector.py`, `scripts/backfill_history.sh`, `scripts/daily_update.sh`

**5 data collectors:**

| Collector | Source | Coverage | Key Detail |
|-----------|--------|----------|------------|
| `price_collector` | yfinance | 2000-present, 21 tickers + VIX | Backfill + daily upsert modes, saves parquet + DB |
| `macro_collector` | FRED ALFRED API | 2000-present, 10 indicators | Vintage (as-reported) data, USD index splice (DTWEXB pre-2006 + DTWEXBGS post-2006) |
| `news_collector` | RSS feeds + NewsAPI | Rolling + 30 days | 4 RSS feeds, deduplication, append-only DB storage |
| `gdelt_collector` | GDELT 2.0 DOC API | 30-day rolling | Tone scoring, keyword batching, 1-sec rate limiting |
| `historical_news_collector` | HuggingFace FNSPID | 1999-2023, 15.7M records | One-time backfill, yearly parquet files, batch DB insert |

**Database schema (7 tables):** `prices`, `macro_indicators`, `news_articles`, `gdelt_events`, `feature_store`, `predictions`, `model_registry` — all with TimescaleDB hypertables and appropriate indexes. Plus `daily_prediction_accuracy` continuous aggregate.

### Phase 2 — Feature Engineering (Complete)
**Files:** `src/features/technical.py`, `src/features/cross_sectional.py`, `src/features/calendar.py`, `src/features/macro.py`, `src/features/targets.py`, `src/features/normalizer.py`, `src/features/pipeline.py`

**~130+ features per ticker across 6 categories:**

**Technical indicators (60+ features):**
- Momentum: RSI (7, 14), ROC (5, 20), MACD, Frog-in-the-Pan decomposition (20d, 60d)
- Trend: EMA (20, 50, 200), ADX, EMA crossover signals, 52-week high proximity
- Volatility: ATR, Bollinger Bands, Garman-Klass, Parkinson, Yang-Zhang, volatility-of-volatility, variance ratio (5d/20d)
- Volume: OBV, volume ratio, volume Z-score (60d), VWAP deviation, OBV divergence, Amihud illiquidity (20d, 60d)
- Microstructure: Close Location Value, Corwin-Schultz spread estimator, Roll spread (20d, 60d)
- Returns: lagged (1,2,3,5,10,20 day), log returns

**Cross-sectional features (20+ features):**
- Market breadth: % above 200 EMA, % above 50 EMA, advance-decline ratio (10d), new high-low diff, Zweig breadth thrust
- Dispersion: cross-sectional volatility (1d, 20d), average pairwise correlation (60d)
- Cross-asset regime: SPY-TLT correlation (20d, 60d), SPY-GLD correlation, SPY-HYG correlation, SPY realized vol
- Sector momentum: sector spread (5d, 20d), top/bottom sector return
- Per-stock relative: market-relative return (5d, 20d, 60d), rolling beta to SPY (60d), sector-relative return, sector-relative Z-score

**Calendar features (15 features):**
- OPEX: is_opex_day, is_opex_week, is_quad_witching, days_to_opex, is_day_after_opex
- Quarter: days_to_quarter_end, is_quarter_end_week, is_quarter_start_week
- Holiday: is_pre_holiday, is_post_holiday
- Cyclical: dow_sin, dow_cos, month_sin, month_cos, is_monday, is_friday, sell_in_may

**Macro features (25+ features):**
- Raw: fed_funds_rate, yield_curve_10y2y, yield_curve_10y3m, unemployment, cpi, core_cpi, usd_index, initial_claims, industrial_production, consumer_sentiment
- Derived: recession_signal, inversion_duration, dual_inversion, rate_change (1m, 3m), rate_hiking/cutting flags, cpi_yoy, cpi_mom, cpi_acceleration, cpi_core_spread, real_yield, sahm_indicator, claims_4w_avg, claims_change_mom, usd_roc (20d, 60d), ip_yoy

**VIX features (7 features):**
- vix_level, vix_change (5d, 20d), vix_high_regime, vix_extreme, variance_risk_premium, vix_term_slope

**Target engineering:**
- Multi-horizon forward returns: fwd_return_{1,5,20}d (continuous) + fwd_log_return_{1,5,20}d
- Adaptive volatility-based 3-class labels: label_{1,5,20}d (down=0, flat=1, up=2)
- Threshold formula: `k * sqrt(horizon) * rolling_std(daily_returns, 60d)` where k=0.4
- Ensures ~30-40% flat class regardless of market regime

**Normalization (applied per walk-forward fold):**
- Rolling percentile rank (default) — robust to outliers and fat tails
- Expanding Z-score (alternative) — exponential halflife=126d
- 3-stage feature selection: mutual information filter → SHAP via LightGBM → stability selection across time periods

### Phase 3 — Sentiment Pipeline (Complete)
**Files:** `src/features/sentiment.py`, `src/features/sentiment_llm.py`, `src/features/sentiment_features.py`

**Two sentiment scorers:**

| Scorer | Model | Speed | Use Case |
|--------|-------|-------|----------|
| FinBERT | `beethogedeon/Modern-FinBERT-large` (fallback: `ProsusAI/finbert`) | ~1-3ms/headline | Real-time scoring, batch processing |
| LLM | Any vLLM-compatible model (configurable) | ~100ms/headline | Daily batch enrichment, reasoning + factor extraction |

**Sentiment features (20+ features):**
- Per-source daily aggregation: mean, std, min, max, count, pos_ratio, neg_ratio (for both FinBERT and LLM)
- GDELT tone: mean, std, min, max, event_count, tone_dispersion
- Derived: sentiment momentum (5d, 20d), sentiment surprise (Z-score vs rolling mean), FinBERT-LLM disagreement, ensemble mean, news volume Z-score, GDELT-FinBERT alignment
- All aggregated with exponential decay weighting (halflife=24h)

### Phase 4 — Regime Detection + Baseline Models (Complete)
**Files:** `src/models/regime.py`, `src/models/baseline.py`, `src/training/walk_forward.py`, `src/training/evaluate.py`, `src/training/train.py`

**HMM Regime Detector:**
- 3-state GaussianHMM (bull/bear/sideways) fitted on daily_return + volatility_20d + volume_ratio
- States auto-labeled post-hoc by mean return (highest = bull, lowest = bear)
- Outputs: regime_state, regime_proba_{bear,sideways,bull}, regime_duration, regime_transition
- Saved as `models/hmm_regime.joblib`

**Walk-Forward Validation:**
- Expanding window (anchored) and sliding window (fixed 5-year) modes
- 21-day purged embargo gap between train and test (matches max 20d forecast horizon)
- Config: initial_train_years=3, test_months=3, step_months=1, min_test_samples=50
- Per-fold scaler fitting (normalization applied INSIDE each fold)

**Baseline Models:**
- LightGBM: 500 estimators, max_depth=6, lr=0.05, balanced class weights, early stopping
- XGBoost: 500 estimators, max_depth=6, lr=0.05, hist tree method, early stopping
- Both trained across all 3 horizons (1d, 5d, 20d) via walk-forward

**Evaluation Metrics:**
- Classification: accuracy, directional accuracy (up vs down ignoring flat), macro F1, per-class precision/recall/F1
- Trading simulation: Sharpe ratio (annualized), cumulative return, max drawdown, win rate
- Confidence: average confidence, high-confidence subset accuracy
- Results saved as JSON per model/horizon

### Phase 5 — Foundation Models (Complete)
**Files:** `src/models/foundation.py`, `src/training/train_foundation.py`

**Chronos-2 Forecaster (primary):**
- Amazon Chronos-2 (`amazon/chronos-bolt-base`, 200M params)
- Zero-shot: no training required, auto-downloads ~800MB on first use
- Converts probabilistic price forecasts into directional predictions (up/flat/down)
- Uses adaptive volatility thresholds matching our target engineering
- Supports batch prediction for walk-forward evaluation

**Moirai 2.0 Forecaster (optional):**
- Salesforce Moirai 2.0 (`Salesforce/moirai-2.0-R-small`)
- Quantile-based probabilistic forecasts
- Requires `uni2ts` package (optional dependency)

**Model Comparison:**
- `compare_all_models()` loads all saved results and ranks models side by side
- Metrics: accuracy, directional accuracy, F1, Sharpe, max drawdown, win rate

### Phase 6 — TFT + Ensemble (Complete)
**Files:** `src/models/tft.py`, `src/models/ensemble.py`, `src/training/tft_data.py`, `src/training/train_tft.py`

**Temporal Fusion Transformer:**
- Uses pytorch-forecasting `TemporalFusionTransformer`
- Automatic covariate classification: static (ticker), known-future (calendar), unknown (technicals, macro, sentiment)
- Data preparation converts our feature matrix into TimeSeriesDataSet format with GroupNormalizer
- Encoder length: 60 days (3 months of lookback)
- Walk-forward training with PyTorch Lightning, early stopping, model checkpointing
- W&B logging integration (optional)
- Quick mode (`--quick`) for testing: 5 epochs, 2 folds
- Config: hidden_size=128, 8 attention heads, dropout=0.15, lr=0.0005
- Limits unknown reals to top 80 by variance to avoid memory issues

**Stacking Ensemble:**
- Level-1 models: LightGBM, XGBoost, Chronos-2, TFT (any subset available)
- Level-2 meta-learner: Logistic Regression with isotonic probability calibration
- Weighted vote strategy: models weighted by their walk-forward directional accuracy
- Walk-forward ensemble: per-fold evaluation aligned to level-1 model folds
- `StackingEnsemble` class with fit/predict/save/load interface
- Requires at least 2 level-1 model results to train

---

## 4. Configuration Reference

### tickers.yaml
- `tickers`: 21 symbols (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, V, SPY, QQQ, IWM, GLD, TLT, HYG, XLF, XLE, XLK, XLV, XLI)
- `vix_ticker`: "^VIX"
- `macro_indicators`: 10 FRED series + legacy USD index (DTWEXB) for pre-2006 splice
- `gdelt_keywords`: 10 financial event keyword groups

### data_config.yaml
- `history.start_date`: "2000-01-01"
- `database`: PostgreSQL connection (env var resolved)
- `validation`: initial_train_years=3, test_months=3, step_months=1, gap_days=21
- `horizons`: [1, 5, 20]
- `labeling.flat_threshold_pct`: adaptive (k=0.4 * sqrt(h) * rolling_std)
- `features`: technical indicator periods, macro settings (vintage=true, usd_splice=true), sentiment model IDs, news sources
- `news.historical_news.dataset`: "Zihan1004/FNSPID"

### model_config.yaml
- `regime`: GaussianHMM, n_states=3, features=[daily_return, volatility_20d, volume_ratio]
- `baseline.lgbm`: n_estimators=500, max_depth=6, lr=0.05, balanced
- `baseline.xgboost`: n_estimators=500, max_depth=6, lr=0.05, hist
- `foundation.chronos2`: amazon/chronos-bolt-base, context=512, prediction=20
- `foundation.moirai2`: Salesforce/moirai-2.0-R-small
- `tft`: hidden_size=128, 8 attention heads, dropout=0.15, patience=10
- `ensemble`: logistic regression meta-learner, isotonic calibration
- `monitoring`: drift_window=30d, accuracy_threshold=0.52, regime-triggered retraining

---

## 5. Database Schema

**Tables (all TimescaleDB hypertables):**

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `prices` | OHLCV per ticker | time, ticker, open, high, low, close, volume, adj_close |
| `macro_indicators` | FRED economic data | time, indicator, value, vintage_date |
| `news_articles` | Headlines + sentiment | time, headline, source, sentiment_finbert, sentiment_llm, tickers[] |
| `gdelt_events` | Global events + tone | time, title, tone, domain, keywords[] |
| `feature_store` | Pre-computed features | time, ticker, features (JSONB) |
| `predictions` | Model outputs | time, ticker, model_name, horizon, predicted_direction, predicted_proba, actual_return, regime_state |
| `model_registry` | Trained model versions | model_name, version, stage, metrics (JSONB), artifact_path |

**Continuous aggregate:** `daily_prediction_accuracy` — auto-materialized daily accuracy per model/ticker.

---

## 6. Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `DB_HOST` | No | localhost | PostgreSQL host |
| `DB_PORT` | No | 5432 | PostgreSQL port |
| `DB_NAME` | No | stocks | Database name |
| `DB_USER` | No | postgres | Database user |
| `DB_PASSWORD` | No | stockpass | Database password |
| `FRED_API_KEY` | Phase 1+ | — | FRED macro data access |
| `NEWSAPI_KEY` | Optional | — | NewsAPI headline fetching |
| `VLLM_BASE_URL` | Optional | — | vLLM endpoint for LLM sentiment |
| `VLLM_MODEL` | Optional | — | LLM model name |
| `WANDB_API_KEY` | Phase 6+ | — | Experiment tracking |
| `HF_HOME` | No | ~/.cache/huggingface | HuggingFace model cache |
| `MODELS_DIR` | No | models | Trained model output directory |

---

## 7. Makefile Commands

```
Setup:        make setup / make setup-gpu
Database:     make db / make db-stop / make db-reset
Data:         make backfill / make backfill-quick / make daily-update
              make collect-prices / make collect-macro / make collect-news / make collect-gdelt
              make data-status
Features:     make features / make features-single TICKER=AAPL / make feature-status
Sentiment:    make score-sentiment / make score-sentiment-llm / make sentiment-test / make sentiment-status
Training:     make train-baseline / make train-lgbm / make train-xgboost / make train-regime / make train-status
Foundation:   make eval-chronos / make eval-moirai / make compare-models
TFT+Ensemble: make train-tft / make train-tft-quick / make train-ensemble
Dev:          make test / make lint / make lint-fix / make run / make api
Health:       make check-health / make clean
```

---

## 8. All Phases Complete

### Phase 7 — Serving + Dashboard (Complete)
- FastAPI REST endpoints: /health, /predictions, /prices, /models, /sentiment, /pipeline/status
- Next.js 15 dashboard with TradingView Lightweight Charts (candlestick + line), Recharts
- TanStack Query v5 for data fetching (polling)
- Pages: Overview (candlestick chart + predictions table + market trend), Models, Sentiment, Pipeline Health
- Region filter: US, Egypt, Saudi, UAE, Qatar, Kuwait
- Time range selector: 1W, 1M, 3M, 6M, 1Y, ALL

### Phase 8 — Monitoring + Retraining (Complete)
**Files:** `src/inference/logger.py`, `src/training/drift.py`, `src/training/registry.py`, `src/training/retrain.py`

- **Prediction logger**: logs every prediction to DB, backfills actuals when ground truth available
- **Drift detection**: 3-type monitoring (performance drift via rolling accuracy, data drift via PSI, regime drift via HMM state change)
- **Model registry**: stage transitions (staging → production → archived), promotion logic based on directional accuracy improvement
- **Retraining orchestrator**: drift-triggered + forced retraining, automatic comparison with production model, auto-promote if better

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Start from 2000 (not 2015) | Covers dot-com crash, 9/11, 2008 crisis — 26 years of regime diversity |
| FRED ALFRED vintage data | Prevents look-ahead bias from retroactive macro data revisions |
| Adaptive volatility thresholds | Fixed % thresholds cause class imbalance shifts across regimes |
| 21-day purged embargo gap | Matches max 20d forecast horizon — prevents label leakage |
| Scaler fitted per fold | Prevents feature normalization leakage across train/test |
| Chronos-2 as zero-shot baseline | Gives strong baseline with zero training cost, validates pipeline |
| Modern-FinBERT-large over ProsusAI/finbert | 2025 ModernBERT architecture, 8K context, better accuracy |
| Cross-sectional features | Captures regime, relative value, and breadth — highest academic IC |
| Variance risk premium | VIX minus realized vol — one of strongest documented equity predictors |
| OPEX calendar features | Strongest calendar effect — market maker hedging creates predictable flows |
| FinBERT-LLM disagreement as feature | Model divergence signals uncertainty, valuable for ensemble confidence |

---

## 10. Dependencies

**Core (CPU, ~500MB download):**
yfinance, pandas-datareader, feedparser, requests, fredapi, pandas, pandas-ta, numpy, scikit-learn, lightgbm, xgboost, shap, hmmlearn, transformers, openai, fastapi, uvicorn, pydantic, psycopg2-binary, sqlalchemy, wandb, mlflow, evidently, joblib, pyyaml, python-dotenv, rich

**Deep learning (optional GPU, ~2.5GB with CUDA):**
torch, pytorch-forecasting, pytorch-lightning, chronos-forecasting

**Optional:**
uni2ts (Moirai 2.0), datasets (FNSPID historical news)

**Dev:**
pytest, pytest-cov, ruff, mypy, pre-commit, jupyter
