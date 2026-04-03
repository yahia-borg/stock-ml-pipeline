"""
FastAPI server — REST API for stock predictions, model info, and pipeline health.

Endpoints:
  GET  /health                         — API health check
  GET  /api/predictions/latest         — latest prediction per ticker
  GET  /api/predictions/{ticker}       — prediction for a specific ticker
  GET  /api/prices/{ticker}            — OHLCV price data
  GET  /api/models                     — list of available models
  GET  /api/models/results             — evaluation metrics for all models
  GET  /api/sentiment/latest           — recent sentiment scores
  GET  /api/pipeline/status            — data pipeline health (row counts, last update)

Run: uvicorn src.inference.api:app --host 0.0.0.0 --port 8099 --reload
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.utils.logger import get_logger

log = get_logger("inference.api")

app = FastAPI(
    title="Stock Forecast API",
    version="0.2.0",
    description="Stock price direction forecasting with ML ensemble",
)

# Allow dashboard frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int

class PredictionResult(BaseModel):
    ticker: str
    horizon: int
    timestamp: str
    best_model: str | None = None
    best_prediction: dict | None = None
    predictions: dict = {}
    regime: dict | None = None
    error: str | None = None

class PricePoint(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class ModelInfo(BaseModel):
    name: str
    type: str
    horizon: str
    status: str

class PipelineStatus(BaseModel):
    database_connected: bool
    tables: dict
    last_price_update: str | None = None
    last_news_update: str | None = None
    models_available: int


# ── Startup ────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Load models on server startup."""
    try:
        from src.inference.predictor import get_predictor
        predictor = get_predictor()
        log.info("API ready with %d models", len(predictor.get_available_models()))
    except Exception as e:
        log.warning("Model loading failed (API still serves health/data): %s", e)


# ── Health ─────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        from src.inference.predictor import get_predictor
        predictor = get_predictor()
        n_models = len(predictor.get_available_models())
    except Exception:
        n_models = 0

    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=n_models,
    )


# ── Predictions ────────────────────────────────────────────

@app.get("/api/predictions/latest", response_model=list[PredictionResult])
async def get_latest_predictions(
    horizon: int = Query(default=5, description="Forecast horizon in days"),
):
    """Get latest predictions for all tickers."""
    from src.inference.predictor import get_predictor

    try:
        predictor = get_predictor()
        results = predictor.predict_all_tickers(horizon=horizon)
        return [PredictionResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/{ticker}", response_model=PredictionResult)
async def get_prediction(
    ticker: str,
    horizon: int = Query(default=5, description="Forecast horizon in days"),
):
    """Get prediction for a specific ticker."""
    from src.inference.predictor import get_predictor

    try:
        predictor = get_predictor()
        result = predictor.predict_ticker(ticker.upper(), horizon=horizon)
        return PredictionResult(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Prices ─────────────────────────────────────────────────

@app.get("/api/prices/{ticker}")
async def get_prices(
    ticker: str,
    days: int = Query(default=365, description="Number of days of history"),
):
    """Get OHLCV price data for charting."""
    from src.utils.config import ROOT_DIR, get_data_config
    import pandas as pd

    cfg = get_data_config()
    path = ROOT_DIR / cfg["storage"]["raw_dir"] / "prices" / f"{ticker.upper()}.parquet"

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").tail(days)

    records = []
    for _, row in df.iterrows():
        records.append({
            "time": row["time"].isoformat(),
            "open": round(float(row["open"]), 2),
            "high": round(float(row["high"]), 2),
            "low": round(float(row["low"]), 2),
            "close": round(float(row["close"]), 2),
            "volume": int(row["volume"]) if not np.isnan(row["volume"]) else 0,
        })

    return {"ticker": ticker.upper(), "count": len(records), "data": records}


# ── Models ─────────────────────────────────────────────────

@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    """List all available models."""
    from src.inference.predictor import get_predictor

    try:
        predictor = get_predictor()
        models = predictor.get_available_models()
        return [ModelInfo(**m) for m in models]
    except Exception:
        return []


@app.get("/api/models/results")
async def get_model_results():
    """Get evaluation metrics for all models."""
    from src.inference.predictor import get_model_results as load_results

    results = load_results()
    if not results:
        return {"models": [], "message": "No results yet. Run make train-baseline first."}

    # Summarize each model
    summaries = []
    for r in results:
        metrics = r.get("fold_metrics", [])
        if not metrics:
            continue

        summary = {
            "model": r.get("model_type", "unknown"),
            "horizon": r.get("horizon", "?"),
            "n_folds": len(metrics),
            "accuracy": round(np.mean([m["accuracy"] for m in metrics]), 4),
            "directional_accuracy": round(np.mean([m.get("directional_accuracy", 0) for m in metrics]), 4),
            "macro_f1": round(np.mean([m["macro_f1"] for m in metrics]), 4),
            "sharpe_ratio": round(np.mean([m.get("sharpe_ratio", 0) for m in metrics]), 4),
            "max_drawdown": round(np.mean([m.get("max_drawdown", 0) for m in metrics]), 4),
            "win_rate": round(np.mean([m.get("win_rate", 0) for m in metrics]), 4),
            "zero_shot": r.get("zero_shot", False),
        }
        summaries.append(summary)

    # Sort by directional accuracy
    summaries.sort(key=lambda x: x["directional_accuracy"], reverse=True)
    return {"models": summaries}


# ── Sentiment ──────────────────────────────────────────────

@app.get("/api/sentiment/latest")
async def get_latest_sentiment(limit: int = Query(default=50)):
    """Get recent news articles with sentiment scores."""
    from src.utils.db import read_sql, check_connection

    if not check_connection():
        return {"articles": [], "message": "Database not connected"}

    try:
        df = read_sql(f"""
            SELECT time, headline, source, sentiment_finbert, sentiment_llm
            FROM news_articles
            WHERE sentiment_finbert IS NOT NULL
            ORDER BY time DESC
            LIMIT {limit}
        """)

        articles = []
        for _, row in df.iterrows():
            articles.append({
                "time": str(row["time"]),
                "headline": row["headline"],
                "source": row["source"],
                "sentiment_finbert": round(float(row["sentiment_finbert"]), 3) if row["sentiment_finbert"] else None,
                "sentiment_llm": round(float(row["sentiment_llm"]), 3) if row["sentiment_llm"] else None,
            })

        return {"count": len(articles), "articles": articles}
    except Exception as e:
        return {"articles": [], "error": str(e)}


# ── Pipeline Status ────────────────────────────────────────

@app.get("/api/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get data pipeline health status."""
    from src.utils.db import read_sql, check_connection
    from src.inference.predictor import get_predictor

    db_connected = check_connection()
    tables = {}
    last_price = None
    last_news = None

    if db_connected:
        try:
            for table in ["prices", "macro_indicators", "news_articles", "gdelt_events", "predictions"]:
                result = read_sql(f"SELECT COUNT(*) as n FROM {table}")
                tables[table] = int(result["n"].iloc[0])

            # Last update timestamps
            price_result = read_sql("SELECT MAX(time) as latest FROM prices")
            if not price_result.empty and price_result["latest"].iloc[0]:
                last_price = str(price_result["latest"].iloc[0])

            news_result = read_sql("SELECT MAX(time) as latest FROM news_articles")
            if not news_result.empty and news_result["latest"].iloc[0]:
                last_news = str(news_result["latest"].iloc[0])
        except Exception as e:
            log.warning("Status query failed: %s", e)

    try:
        predictor = get_predictor()
        n_models = len(predictor.get_available_models())
    except Exception:
        n_models = 0

    return PipelineStatus(
        database_connected=db_connected,
        tables=tables,
        last_price_update=last_price,
        last_news_update=last_news,
        models_available=n_models,
    )
