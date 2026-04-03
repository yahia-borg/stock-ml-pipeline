# Stock Forecasting ML Pipeline — Build Plan

> **Goal:** Predict stock price direction and magnitude using historical OHLCV data, macroeconomic indicators, and real-time global event sentiment.  
> **Horizon:** 1-day, 5-day, 20-day forecasts  
> **Stack:** Python 3.11 · PyTorch · vLLM · PostgreSQL/TimescaleDB · FastAPI

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Phase 1 — Data Infrastructure](#2-phase-1--data-infrastructure)
3. [Phase 2 — Feature Engineering](#3-phase-2--feature-engineering)
4. [Phase 3 — Sentiment & Events Pipeline](#4-phase-3--sentiment--events-pipeline)
5. [Phase 4 — Model Development](#5-phase-4--model-development)
6. [Phase 5 — Training & Validation](#6-phase-5--training--validation)
7. [Phase 6 — Inference & Serving](#7-phase-6--inference--serving)
8. [Phase 7 — Monitoring & Retraining](#8-phase-7--monitoring--retraining)
9. [Data Sources Reference](#9-data-sources-reference)
10. [Stack & Dependencies](#10-stack--dependencies)
11. [Risks & Mitigations](#11-risks--mitigations)
12. [Milestones](#12-milestones)

---

## 1. Project Structure

```
stock-forecast/
├── data/
│   ├── raw/                    # downloaded files, never modified
│   │   ├── prices/             # parquet per ticker
│   │   ├── news/               # raw JSON from APIs
│   │   └── macro/              # FRED, World Bank exports
│   ├── processed/              # cleaned, merged feature matrices
│   └── splits/                 # train/val/test index files
│
├── src/
│   ├── collectors/             # data ingestion scripts
│   │   ├── price_collector.py
│   │   ├── news_collector.py
│   │   ├── gdelt_collector.py
│   │   └── macro_collector.py
│   │
│   ├── features/               # feature engineering
│   │   ├── technical.py        # RSI, MACD, Bollinger etc.
│   │   ├── macro.py            # yield curve, VIX, FX
│   │   └── sentiment.py        # LLM scoring, FinBERT
│   │
│   ├── models/                 # model definitions
│   │   ├── baseline.py         # XGBoost / LightGBM
│   │   ├── transformer.py      # temporal transformer
│   │   ├── event_model.py      # news + RAG model
│   │   └── ensemble.py         # stacking layer
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── walk_forward.py     # validation framework
│   │   └── losses.py           # custom loss functions
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── api.py              # FastAPI serving
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── db.py               # TimescaleDB helpers
│
├── notebooks/                  # EDA and experimentation
│   ├── 01_eda_prices.ipynb
│   ├── 02_eda_sentiment.ipynb
│   ├── 03_feature_selection.ipynb
│   └── 04_model_comparison.ipynb
│
├── tests/
├── configs/
│   ├── tickers.yaml            # universe of stocks to track
│   ├── model_config.yaml
│   └── data_config.yaml
│
├── scripts/
│   ├── bootstrap_db.sh         # init TimescaleDB schema
│   ├── backfill_history.sh     # one-time 2015 backfill
│   └── daily_update.sh         # cron job
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 2. Phase 1 — Data Infrastructure

### 2.1 Database setup

Use **TimescaleDB** (PostgreSQL extension for time series) — it gives SQL familiarity with time-series optimizations (hypertables, compression, continuous aggregates).

```sql
-- bootstrap_db.sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE prices (
    time        TIMESTAMPTZ NOT NULL,
    ticker      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    adj_close   DOUBLE PRECISION
);
SELECT create_hypertable('prices', 'time');
CREATE INDEX ON prices (ticker, time DESC);

CREATE TABLE news_articles (
    id          SERIAL PRIMARY KEY,
    time        TIMESTAMPTZ NOT NULL,
    source      TEXT,
    headline    TEXT,
    url         TEXT,
    sentiment   DOUBLE PRECISION,
    tickers     TEXT[]
);
SELECT create_hypertable('news_articles', 'time');

CREATE TABLE macro_indicators (
    time        TIMESTAMPTZ NOT NULL,
    indicator   TEXT NOT NULL,
    value       DOUBLE PRECISION
);
SELECT create_hypertable('macro_indicators', 'time');
```

### 2.2 Historical price backfill (2015 → today)

```python
# src/collectors/price_collector.py
import yfinance as yf
import pandas as pd
from pathlib import Path
import yaml

def backfill_prices(tickers: list[str], start: str = "2015-01-01"):
    out = Path("data/raw/prices")
    out.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        df.columns = [c[0].lower() for c in df.columns]
        df["ticker"] = ticker
        df.to_parquet(out / f"{ticker}.parquet")
        print(f"  ✓ {ticker}: {len(df)} rows")

def daily_update(tickers: list[str]):
    """Run daily via cron — fetches last 5 days to catch any gaps."""
    for ticker in tickers:
        df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
        # upsert into TimescaleDB
        # ...

if __name__ == "__main__":
    with open("configs/tickers.yaml") as f:
        cfg = yaml.safe_load(f)
    backfill_prices(cfg["tickers"])
```

### 2.3 Ticker universe config

```yaml
# configs/tickers.yaml
tickers:
  # US large cap
  - AAPL
  - MSFT
  - GOOGL
  - AMZN
  - NVDA
  - META
  - TSLA
  # Indices & ETFs
  - SPY       # S&P 500
  - QQQ       # NASDAQ
  - VIX       # volatility index
  - GLD       # gold
  - TLT       # 20yr treasury
  # Sectors
  - XLF       # financials
  - XLE       # energy
  - XLK       # technology

macro_indicators:
  - DFF       # Fed Funds Rate (FRED)
  - T10Y2Y    # yield curve spread
  - UNRATE    # unemployment rate
  - CPIAUCSL  # CPI inflation
  - DTWEXBGS  # USD index
```

---

## 3. Phase 2 — Feature Engineering

### 3.1 Technical indicators

```python
# src/features/technical.py
import pandas as pd
import pandas_ta as ta

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: OHLCV DataFrame. Output: same df + technical columns."""

    # Momentum
    df.ta.rsi(length=14, append=True)        # RSI_14
    df.ta.rsi(length=7,  append=True)        # RSI_7
    df.ta.roc(length=5,  append=True)        # ROC_5
    df.ta.roc(length=20, append=True)        # ROC_20

    # Trend
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.ema(length=20, append=True)        # EMA_20
    df.ta.ema(length=50, append=True)        # EMA_50
    df.ta.ema(length=200, append=True)       # EMA_200
    df.ta.adx(length=14, append=True)        # ADX

    # Volatility
    df.ta.bbands(length=20, append=True)     # BBands
    df.ta.atr(length=14, append=True)        # ATR_14
    df["vol_20d"] = df["return_1d"].rolling(20).std()
    df["vol_5d"]  = df["return_1d"].rolling(5).std()

    # Volume
    df.ta.obv(append=True)                   # OBV
    df.ta.vwap(append=True)                  # VWAP
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Returns (targets and lagged features)
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"return_{lag}d"] = df["close"].pct_change(lag)

    # Price position
    df["dist_52w_high"] = df["close"] / df["high"].rolling(252).max() - 1
    df["dist_52w_low"]  = df["close"] / df["low"].rolling(252).min() - 1

    return df.dropna()
```

### 3.2 Macro features

```python
# src/features/macro.py
import pandas_datareader as pdr
import pandas as pd
from datetime import date

FRED_SERIES = {
    "fed_funds_rate": "DFF",
    "yield_curve_10y2y": "T10Y2Y",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "usd_index": "DTWEXBGS",
}

def fetch_macro_features(start="2015-01-01") -> pd.DataFrame:
    dfs = []
    for name, series_id in FRED_SERIES.items():
        df = pdr.get_data_fred(series_id, start=start)
        df.columns = [name]
        dfs.append(df)

    macro = pd.concat(dfs, axis=1)
    macro = macro.resample("D").ffill()  # forward-fill weekends/holidays

    # Derived macro features
    macro["yield_curve_slope"] = macro["yield_curve_10y2y"]
    macro["rate_change_1m"] = macro["fed_funds_rate"].pct_change(21)
    macro["cpi_yoy"] = macro["cpi"].pct_change(252)
    macro["recession_signal"] = (macro["yield_curve_10y2y"] < 0).astype(int)

    return macro
```

### 3.3 Feature selection strategy

After building the full feature matrix, apply the following selection steps:

1. **Remove near-zero variance** features (threshold < 0.01)
2. **Remove highly correlated** features (Pearson r > 0.95, keep one)
3. **SHAP feature importance** from a quick XGBoost run — drop bottom 20%
4. **Recursive Feature Elimination** with cross-validation on final set

```python
# notebooks/03_feature_selection.ipynb
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
import shap
import xgboost as xgb

# 1. Train quick baseline
model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# 2. SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Plot feature importance
shap.summary_plot(shap_values, X_val, feature_names=feature_cols)

# Drop features with mean |SHAP| < threshold
mean_shap = np.abs(shap_values).mean(axis=0)
keep_features = [feature_cols[i] for i, s in enumerate(mean_shap) if s > 0.001]
```

---

## 4. Phase 3 — Sentiment & Events Pipeline

### 4.1 News ingestion

```python
# src/collectors/news_collector.py
import feedparser
import requests
import pandas as pd
from datetime import datetime, timedelta

FREE_FEEDS = {
    "reuters_markets": "https://feeds.reuters.com/reuters/businessNews",
    "wsj_markets":     "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "seeking_alpha":   "https://seekingalpha.com/feed.xml",
    "yahoo_finance":   "https://finance.yahoo.com/news/rssindex",
}

def fetch_rss_headlines() -> pd.DataFrame:
    articles = []
    for source, url in FREE_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    "source":   source,
                    "title":    entry.get("title", ""),
                    "summary":  entry.get("summary", "")[:500],
                    "url":      entry.get("link", ""),
                    "published":entry.get("published", ""),
                })
        except Exception as e:
            print(f"Feed error {source}: {e}")
    return pd.DataFrame(articles)

def fetch_newsapi(query: str, api_key: str, days=7) -> pd.DataFrame:
    """NewsAPI.org — free tier: 100 req/day, 1 month history."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        "sortBy":   "publishedAt",
        "language": "en",
        "apiKey":   api_key,
        "pageSize": 100,
    }
    r = requests.get(url, params=params)
    articles = r.json().get("articles", [])
    return pd.DataFrame(articles)[["publishedAt", "title", "description", "source"]]
```

### 4.2 GDELT global events

```python
# src/collectors/gdelt_collector.py
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_gdelt_events(keywords: list[str], days=30) -> pd.DataFrame:
    """
    GDELT 2.0 DOC API — completely free, no key required.
    Covers 100+ languages, 250+ countries.
    Tone field: positive = bullish signal, negative = bearish.
    """
    query = " OR ".join(f'"{k}"' for k in keywords)
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query":         query,
        "mode":          "artlist",
        "maxrecords":    250,
        "format":        "json",
        "startdatetime": (datetime.now() - timedelta(days=days)).strftime("%Y%m%d%H%M%S"),
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()

    articles = r.json().get("articles", [])
    df = pd.DataFrame(articles)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ")
    df["tone"] = pd.to_numeric(df.get("tone", 0), errors="coerce").fillna(0)
    return df[["date", "title", "url", "tone", "domain"]]

# Keywords to track for financial impact
MARKET_KEYWORDS = [
    "federal reserve interest rates",
    "inflation CPI",
    "recession GDP",
    "earnings report",
    "oil price OPEC",
    "China economy",
    "geopolitical conflict",
]
```

### 4.3 LLM sentiment scoring

Two approaches — use both and compare:

**Option A: FinBERT (fast, ~1ms/headline, offline)**

```python
# src/features/sentiment.py
from transformers import pipeline
import pandas as pd
import torch

# Load once, reuse
finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
)

LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def score_with_finbert(headlines: list[str]) -> list[float]:
    results = finbert(headlines, batch_size=64)
    return [
        LABEL_MAP.get(r["label"], 0.0) * r["score"]
        for r in results
    ]
```

**Option B: vLLM LLM (richer, ~100ms/headline, context-aware)**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://your-h200-ip:8021/v1",
    api_key="dummy"
)

SENTIMENT_PROMPT = """You are a financial analyst. Rate the market impact of this headline.

Headline: {headline}
Ticker context: {ticker}

Rules:
- Return ONLY a JSON object: {{"score": float, "reason": "one sentence"}}
- Score range: -1.0 (strongly bearish) to +1.0 (strongly bullish)
- 0.0 = neutral / irrelevant
- Consider: earnings, macro policy, geopolitics, sector news"""

def score_with_llm(headlines: list[str], ticker: str = "market") -> list[dict]:
    results = []
    for headline in headlines:
        try:
            resp = client.chat.completions.create(
                model="your-model",
                messages=[{"role": "user",
                           "content": SENTIMENT_PROMPT.format(
                               headline=headline, ticker=ticker)}],
                max_tokens=60,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            import json
            results.append(json.loads(resp.choices[0].message.content))
        except Exception:
            results.append({"score": 0.0, "reason": "parse error"})
    return results
```

### 4.4 Daily sentiment aggregation

```python
def build_daily_sentiment(df_news: pd.DataFrame) -> pd.Series:
    """Aggregate per-article scores to a single daily sentiment signal."""

    df_news["date"] = pd.to_datetime(df_news["published"]).dt.date

    # Weighted average: recent articles weighted higher
    df_news["hours_ago"] = (
        pd.Timestamp.now() - pd.to_datetime(df_news["published"])
    ).dt.total_seconds() / 3600

    df_news["weight"] = 1 / (1 + df_news["hours_ago"] / 24)

    daily = (
        df_news.groupby("date")
        .apply(lambda g: np.average(g["sentiment"], weights=g["weight"]))
        .rename("daily_sentiment")
    )

    return daily
```

---

## 5. Phase 4 — Model Development

### 5.1 Baseline — XGBoost/LightGBM

Start here. Interpretable, fast to train, strong benchmark.

```python
# src/models/baseline.py
import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_lgbm_classifier():
    return lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
    )

def build_xgb_classifier():
    return xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        tree_method="gpu_hist",  # uses your A4500
        random_state=42,
    )
```

### 5.2 Temporal Transformer

For capturing sequential patterns in price/volume time series.

```python
# src/models/transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StockTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        n_classes: int = 3,      # up / flat / down
        forecast_horizons: int = 3,  # predict 1d, 5d, 20d simultaneously
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-horizon heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes),
            )
            for _ in range(forecast_horizons)
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # x: [batch, seq_len, n_features]
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        last = x[:, -1, :]   # use last timestep
        return [head(last) for head in self.heads]
```

### 5.3 Event model (LLM + RAG)

```python
# src/models/event_model.py
"""
Uses RAG to retrieve relevant historical events similar to current news,
then prompts an LLM to reason about likely market impact.
"""
from openai import OpenAI
import numpy as np

client = OpenAI(base_url="http://your-h200-ip:8021/v1", api_key="dummy")

EVENT_REASONING_PROMPT = """You are an expert quantitative analyst.

Current market conditions:
- Ticker: {ticker}
- Current price: {price}
- Recent return (5d): {return_5d:.2%}
- Volatility (20d): {volatility:.2%}

Recent news headlines (last 24h):
{headlines}

Similar historical events and their outcomes:
{rag_context}

Task: Based on the above, predict the likely price direction over the next 5 trading days.

Return JSON only:
{{
  "direction": "up" | "flat" | "down",
  "confidence": 0.0-1.0,
  "magnitude_pct": expected_move_percent,
  "key_factors": ["factor1", "factor2"],
  "risk_factors": ["risk1", "risk2"]
}}"""

def predict_with_events(
    ticker: str,
    headlines: list[str],
    market_data: dict,
    rag_context: str = "",
) -> dict:
    prompt = EVENT_REASONING_PROMPT.format(
        ticker=ticker,
        price=market_data.get("price", "N/A"),
        return_5d=market_data.get("return_5d", 0),
        volatility=market_data.get("volatility", 0),
        headlines="\n".join(f"- {h}" for h in headlines[:10]),
        rag_context=rag_context or "No similar historical events found.",
    )
    resp = client.chat.completions.create(
        model="your-model",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    import json
    return json.loads(resp.choices[0].message.content)
```

### 5.4 Ensemble stacking

```python
# src/models/ensemble.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class StackingEnsemble:
    """
    Level-1 models: XGBoost, LightGBM, Transformer, Event LLM
    Level-2 meta-learner: Logistic Regression (keeps it interpretable)
    """

    def __init__(self):
        self.meta_learner = CalibratedClassifierCV(
            LogisticRegression(C=1.0, max_iter=1000),
            method="isotonic",
        )

    def fit(self, level1_probs: np.ndarray, y: np.ndarray):
        # level1_probs: [n_samples, n_models * n_classes]
        self.meta_learner.fit(level1_probs, y)

    def predict_proba(self, level1_probs: np.ndarray) -> np.ndarray:
        return self.meta_learner.predict_proba(level1_probs)

    def predict(self, level1_probs: np.ndarray) -> np.ndarray:
        return self.meta_learner.predict(level1_probs)
```

---

## 6. Phase 5 — Training & Validation

### 6.1 Walk-forward validation — CRITICAL

> **Never use random train/test splits on time series data.**  
> This causes data leakage and produces over-optimistic results that fail in production.

```python
# src/training/walk_forward.py
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Iterator

@dataclass
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp

def walk_forward_splits(
    df: pd.DataFrame,
    initial_train_years: int = 2,
    test_months: int = 3,
    step_months: int = 1,
    gap_days: int = 5,       # gap between train end and test start
) -> Iterator[WalkForwardSplit]:
    """
    Expanding window walk-forward splits.
    gap_days prevents look-ahead bias from multi-day prediction targets.
    """
    start = df.index.min()
    end   = df.index.max()

    train_end = start + pd.DateOffset(years=initial_train_years)

    while train_end + pd.DateOffset(months=test_months) <= end:
        test_start = train_end + pd.DateOffset(days=gap_days)
        test_end   = test_start + pd.DateOffset(months=test_months)

        yield WalkForwardSplit(
            train_start=start,
            train_end=train_end,
            test_start=test_start,
            test_end=min(test_end, end),
        )

        train_end += pd.DateOffset(months=step_months)
```

### 6.2 Training loop

```python
# src/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

def train_transformer(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> nn.Module:

    wandb.init(project="stock-forecast", config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.LongTensor(y_train).to(device)
    X_v  = torch.FloatTensor(X_val).to(device)
    y_v  = torch.LongTensor(y_val).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=config.get("batch_size", 256),
        shuffle=True,
    )

    # Class-weighted loss (market is usually ~55% up, 25% flat, 20% down)
    weights = torch.FloatTensor([1.5, 2.0, 1.8]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.get("lr", 1e-4),
        epochs=config.get("epochs", 50),
        steps_per_epoch=len(loader),
    )

    best_val_loss = float("inf")
    patience = config.get("patience", 10)
    patience_counter = 0

    for epoch in range(config.get("epochs", 50)):
        model.train()
        train_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = sum(criterion(out, y_batch) for out in outputs) / len(outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_v)
            val_loss = sum(criterion(out, y_v) for out in val_outputs) / len(val_outputs)
            val_preds = val_outputs[1].argmax(dim=1)  # 5-day horizon
            val_acc   = (val_preds == y_v).float().mean().item()

        wandb.log({"train_loss": train_loss/len(loader),
                   "val_loss": val_loss.item(),
                   "val_acc": val_acc, "epoch": epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pt"))
    return model
```

### 6.3 Evaluation metrics

```python
# src/training/evaluate.py
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)
import pandas as pd
import numpy as np

def evaluate_model(y_true, y_pred_proba, y_pred, horizon_label="5d"):
    report = classification_report(
        y_true, y_pred,
        target_names=["Down", "Flat", "Up"],
        output_dict=True,
    )

    # Directional accuracy (up vs down, ignoring flat)
    mask = y_true != 1
    dir_acc = (y_true[mask] == y_pred[mask]).mean()

    # Sharpe-like: did correct predictions correspond to larger moves?
    metrics = {
        "horizon":          horizon_label,
        "accuracy":         report["accuracy"],
        "directional_acc":  dir_acc,
        "up_precision":     report["Up"]["precision"],
        "up_recall":        report["Up"]["recall"],
        "down_precision":   report["Down"]["precision"],
        "down_recall":      report["Down"]["recall"],
        "macro_f1":         report["macro avg"]["f1-score"],
    }
    return pd.Series(metrics)
```

---

## 7. Phase 6 — Inference & Serving

### 7.1 Predictor class

```python
# src/inference/predictor.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path

class StockPredictor:
    def __init__(self, model_path: str, scaler_path: str, feature_cols: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_cols = feature_cols

        # Load model
        from src.models.transformer import StockTransformer
        self.model = StockTransformer(n_features=len(feature_cols))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)

        # Load scaler
        import joblib
        self.scaler = joblib.load(scaler_path)

    def predict(self, df: pd.DataFrame, seq_len: int = 60) -> dict:
        """df: most recent rows of feature-engineered data."""
        features = self.scaler.transform(df[self.feature_cols].tail(seq_len).values)
        x = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)  # list of 3 tensors (1d, 5d, 20d)

        results = {}
        horizons = ["1d", "5d", "20d"]
        labels = ["down", "flat", "up"]
        for horizon, output in zip(horizons, outputs):
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            results[horizon] = {
                "prediction": labels[probs.argmax()],
                "confidence": float(probs.max()),
                "probabilities": {
                    "down": float(probs[0]),
                    "flat": float(probs[1]),
                    "up":   float(probs[2]),
                }
            }
        return results
```

### 7.2 FastAPI server

```python
# src/inference/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Stock Forecast API", version="1.0")

predictor = None  # loaded on startup

@app.on_event("startup")
async def load_models():
    global predictor
    predictor = StockPredictor(
        model_path="models/best_transformer.pt",
        scaler_path="models/scaler.joblib",
        feature_cols=FEATURE_COLS,
    )

class PredictionRequest(BaseModel):
    ticker: str
    include_sentiment: bool = True

class PredictionResponse(BaseModel):
    ticker: str
    timestamp: str
    forecasts: dict
    sentiment_score: float
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    try:
        # Fetch latest data
        df = get_latest_features(req.ticker)

        # Get model prediction
        forecasts = predictor.predict(df)

        # Get sentiment
        sentiment = 0.0
        if req.include_sentiment:
            headlines = fetch_rss_headlines()
            sentiment = score_with_finbert(
                headlines["title"].tolist()
            )
            sentiment = float(np.mean(sentiment))

        return PredictionResponse(
            ticker=req.ticker,
            timestamp=pd.Timestamp.now().isoformat(),
            forecasts=forecasts,
            sentiment_score=sentiment,
            confidence=forecasts["5d"]["confidence"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}
```

```bash
# Run the API
uvicorn src.inference.api:app --host 0.0.0.0 --port 8099 --workers 2
```

---

## 8. Phase 7 — Monitoring & Retraining

### 8.1 Prediction logging

```python
# Log every prediction for later evaluation
import sqlite3
import json

def log_prediction(ticker, prediction, actual=None):
    conn = sqlite3.connect("predictions.db")
    conn.execute("""
        INSERT INTO predictions (ticker, timestamp, forecast_json, actual_return)
        VALUES (?, datetime('now'), ?, ?)
    """, (ticker, json.dumps(prediction), actual))
    conn.commit()
    conn.close()
```

### 8.2 Drift detection

```python
# Check if model performance has degraded — trigger retraining if so
def check_model_drift(window_days=30, accuracy_threshold=0.52):
    df = pd.read_sql("""
        SELECT * FROM predictions
        WHERE timestamp > datetime('now', '-{} days')
        AND actual_return IS NOT NULL
    """.format(window_days), conn)

    if len(df) < 50:
        return False  # not enough data

    recent_acc = (df["predicted_direction"] == df["actual_direction"]).mean()
    if recent_acc < accuracy_threshold:
        print(f"Drift detected: accuracy {recent_acc:.2%} < {accuracy_threshold:.2%}")
        return True
    return False
```

### 8.3 Retraining schedule

```bash
# scripts/daily_update.sh — run via cron at 6:00 AM
#!/bin/bash
set -e

echo "=== Daily stock pipeline update $(date) ==="

# 1. Fetch new price data
python src/collectors/price_collector.py --mode daily

# 2. Fetch news and score sentiment
python src/collectors/news_collector.py
python src/features/sentiment.py --score-new

# 3. Check model drift
python src/training/monitor.py --check-drift

# 4. Weekly retrain (Sundays only)
if [ "$(date +%u)" = "7" ]; then
    echo "Weekly retrain starting..."
    python src/training/train.py --incremental
fi

echo "=== Done ==="
```

```
# Crontab entry
0 6 * * 1-5 /path/to/scripts/daily_update.sh >> /var/log/stock_pipeline.log 2>&1
```

---

## 9. Data Sources Reference

| Source | Data type | Free tier | Coverage |
|---|---|---|---|
| **yfinance** | OHLCV prices | Unlimited | 1970s → today |
| **FRED** | Macro indicators | Unlimited | 1950s → today |
| **GDELT** | Global events + tone | Unlimited | 1979 → today |
| **NewsAPI** | News headlines | 100 req/day, 1 month | 2015 → today |
| **RSS feeds** | Live headlines | Unlimited | Rolling |
| **Stooq** | Prices (backup) | Unlimited | 2000s → today |
| **OpenBB** | Aggregator | Unlimited | Multiple |
| **Tiingo** | OHLCV + news | 500 req/day | 1960s → today |
| **Alpha Vantage** | Fundamentals + FX | 25 req/day | 2000s → today |
| **Polygon.io** | Full market data | Paid ($29/mo) | Real-time |

---

## 10. Stack & Dependencies

```txt
# requirements.txt

# Data collection
yfinance>=0.2.40
pandas-datareader>=0.10.0
feedparser>=6.0.10
requests>=2.31.0

# Feature engineering
pandas>=2.0.0
pandas-ta>=0.3.14b
numpy>=1.26.0
scikit-learn>=1.4.0

# ML models
torch>=2.3.0
lightgbm>=4.3.0
xgboost>=2.0.3
shap>=0.45.0

# NLP / Sentiment
transformers>=4.40.0
openai>=1.30.0          # for vLLM client

# Serving
fastapi>=0.111.0
uvicorn>=0.29.0
pydantic>=2.7.0

# Database
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.30

# Tracking
wandb>=0.17.0
mlflow>=2.12.0          # optional alternative

# Utils
joblib>=1.4.0
pyyaml>=6.0.1
python-dotenv>=1.0.1
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_PASSWORD: stockpass
      POSTGRES_DB: stocks
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

  api:
    build: .
    ports:
      - "8099:8099"
    environment:
      DB_URL: postgresql://postgres:stockpass@timescaledb:5432/stocks
      VLLM_URL: http://host.docker.internal:8021/v1
    depends_on:
      - timescaledb

volumes:
  timescale_data:
```

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Data leakage** | Model looks great, fails live | Always use walk-forward splits, never random |
| **Regime change** | Model trained on bull market fails in bear | Include macro regime features; retrain quarterly |
| **Lookahead bias** | Features computed using future data | Add `gap_days=5` buffer between train/test |
| **Survivorship bias** | Only training on stocks that survived | Include delisted stocks in universe |
| **Overfitting** | Works on backtest, fails live | Use out-of-sample test set held from 2023 onward |
| **Stale model** | Performance drifts over time | Monitor accuracy weekly, retrain on drift detection |
| **API rate limits** | News data gaps | Multiple backup sources + local caching |
| **LLM latency** | Sentiment scoring too slow | Use FinBERT for real-time; LLM for daily batch |

---

## 12. Milestones

```
Week 1-2:   Data infrastructure
            ✓ TimescaleDB running in Docker
            ✓ Historical price backfill (2015 → today) for 20 tickers
            ✓ FRED macro data collected
            ✓ GDELT pipeline working

Week 3:     Feature engineering
            ✓ Technical indicators complete
            ✓ Macro features merged
            ✓ Feature matrix built and saved to parquet

Week 4:     Sentiment pipeline
            ✓ RSS + NewsAPI ingestion working
            ✓ FinBERT scoring running on A4500
            ✓ Daily sentiment aggregation working
            ✓ GDELT tone integrated into feature matrix

Week 5-6:   Baseline models
            ✓ XGBoost / LightGBM trained and evaluated
            ✓ Walk-forward validation framework complete
            ✓ SHAP feature importance analysis done
            ✓ Baseline accuracy benchmarked (target: >54% directional acc)

Week 7-8:   Transformer model
            ✓ StockTransformer implemented and training on A4500
            ✓ Multi-horizon heads (1d / 5d / 20d)
            ✓ W&B training tracking working
            ✓ Model outperforms baseline

Week 9:     Event LLM model
            ✓ Prompt engineering for event reasoning complete
            ✓ RAG context retrieval from historical events
            ✓ LLM predictions integrated into feature set

Week 10:    Ensemble + evaluation
            ✓ Stacking layer trained
            ✓ Full walk-forward evaluation across all splits
            ✓ Final metrics: accuracy, directional acc, Sharpe

Week 11:    Serving
            ✓ FastAPI server deployed
            ✓ Daily update cron running
            ✓ Prediction logging in place

Week 12:    Monitoring + cleanup
            ✓ Drift detection working
            ✓ Weekly retrain pipeline tested
            ✓ Documentation complete
```

---

> **Note:** Stock forecasting is inherently difficult — no model consistently beats the market. The goal of this pipeline is to build a statistically sound system with measurable edge, not guaranteed returns. Always validate with paper trading before any live use.
