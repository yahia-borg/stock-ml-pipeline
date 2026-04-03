-- Stock Forecast Pipeline — PostgreSQL + TimescaleDB Schema
-- Runs automatically on first docker-compose up via initdb.d
--
-- NOTE: TimescaleDB hypertables cannot have PRIMARY KEY or UNIQUE constraints
-- unless they include the partitioning column (time). Use regular tables for
-- entities that need serial PKs (e.g., model_registry).

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- PRICES: OHLCV data per ticker (hypertable on time)
-- ============================================================
CREATE TABLE IF NOT EXISTS prices (
    time        TIMESTAMPTZ     NOT NULL,
    ticker      TEXT            NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    adj_close   DOUBLE PRECISION,
    UNIQUE (time, ticker)
);
SELECT create_hypertable('prices', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_time ON prices (ticker, time DESC);

-- ============================================================
-- MACRO: economic indicators from FRED
-- ============================================================
CREATE TABLE IF NOT EXISTS macro_indicators (
    time        TIMESTAMPTZ     NOT NULL,
    indicator   TEXT            NOT NULL,
    value       DOUBLE PRECISION,
    vintage_date DATE,
    UNIQUE (time, indicator, vintage_date)
);
SELECT create_hypertable('macro_indicators', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_indicator ON macro_indicators (indicator, time DESC);

-- ============================================================
-- NEWS: articles and headlines with sentiment scores
-- No serial PK — hypertable requires time in all unique constraints
-- ============================================================
CREATE TABLE IF NOT EXISTS news_articles (
    time                TIMESTAMPTZ     NOT NULL,
    source              TEXT,
    headline            TEXT            NOT NULL,
    summary             TEXT,
    url                 TEXT,
    sentiment_finbert   DOUBLE PRECISION,
    sentiment_llm       DOUBLE PRECISION,
    tickers             TEXT[],
    created_at          TIMESTAMPTZ     DEFAULT NOW()
);
SELECT create_hypertable('news_articles', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_news_tickers ON news_articles USING GIN (tickers);

-- ============================================================
-- GDELT: global events with tone scoring
-- ============================================================
CREATE TABLE IF NOT EXISTS gdelt_events (
    time            TIMESTAMPTZ     NOT NULL,
    title           TEXT,
    url             TEXT,
    tone            DOUBLE PRECISION,
    domain          TEXT,
    keywords        TEXT[],
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);
SELECT create_hypertable('gdelt_events', 'time', if_not_exists => TRUE);

-- ============================================================
-- FEATURES: pre-computed feature matrices (one row per ticker per day)
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_store (
    time            TIMESTAMPTZ     NOT NULL,
    ticker          TEXT            NOT NULL,
    features        JSONB           NOT NULL,
    UNIQUE (time, ticker)
);
SELECT create_hypertable('feature_store', 'time', if_not_exists => TRUE);

-- ============================================================
-- PREDICTIONS: logged model outputs for monitoring & evaluation
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    ticker          TEXT            NOT NULL,
    model_name      TEXT            NOT NULL,
    model_version   TEXT,
    horizon_days    INTEGER         NOT NULL,
    predicted_direction TEXT,
    predicted_proba JSONB,
    confidence      DOUBLE PRECISION,
    actual_return   DOUBLE PRECISION,
    actual_direction TEXT,
    sentiment_score DOUBLE PRECISION,
    regime_state    INTEGER,
    metadata        JSONB
);
SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions (ticker, time DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions (model_name, time DESC);

-- ============================================================
-- MODEL REGISTRY: regular table (not hypertable — needs serial PK)
-- ============================================================
CREATE TABLE IF NOT EXISTS model_registry (
    id              BIGSERIAL       PRIMARY KEY,
    model_name      TEXT            NOT NULL,
    version         TEXT            NOT NULL,
    stage           TEXT            DEFAULT 'staging',
    train_start     DATE,
    train_end       DATE,
    metrics         JSONB,
    artifact_path   TEXT,
    feature_hash    TEXT,
    created_at      TIMESTAMPTZ     DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ,
    UNIQUE (model_name, version)
);
