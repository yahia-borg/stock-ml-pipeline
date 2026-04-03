"""
Price data collector — downloads OHLCV from yfinance.

Modes:
  backfill  — full history from 2015-01-01 for all tickers
  daily     — last 5 trading days to fill gaps (idempotent upsert)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import ROOT_DIR, get_data_config, get_ticker_list, get_tickers_config
from src.utils.db import check_connection, upsert_dataframe
from src.utils.logger import get_logger

log = get_logger("price_collector")


def _normalize_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to our schema."""
    df = df.copy()

    # yfinance returns MultiIndex columns for single ticker: (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Rename adj_close variants
    rename_map = {}
    for col in df.columns:
        if col in ("adj_close", "adj close", "adjclose"):
            rename_map[col] = "adj_close"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure adj_close exists
    if "adj_close" not in df.columns:
        df["adj_close"] = df.get("close")

    df["ticker"] = ticker

    # Index to column
    if df.index.name in ("Date", "date", "Datetime", "datetime"):
        df = df.reset_index()

    # Standardize the time column
    time_col = None
    for candidate in ("Date", "date", "Datetime", "datetime"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col and time_col != "time":
        df = df.rename(columns={time_col: "time"})

    df["time"] = pd.to_datetime(df["time"], utc=True)

    expected = ["time", "ticker", "open", "high", "low", "close", "volume", "adj_close"]
    for col in expected:
        if col not in df.columns:
            df[col] = None

    return df[expected]


def download_ticker(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download OHLCV for a single ticker."""
    log.info("Downloading %s from %s ...", ticker, start)

    kwargs = {"start": start, "auto_adjust": False, "progress": False}
    if end:
        kwargs["end"] = end

    df = yf.download(ticker, **kwargs)

    if df.empty:
        log.warning("No data returned for %s", ticker)
        return pd.DataFrame()

    df = _normalize_columns(df, ticker)
    log.info("  %s: %d rows (%s to %s)", ticker, len(df),
             df["time"].min().date(), df["time"].max().date())
    return df


def save_parquet(df: pd.DataFrame, ticker: str) -> Path:
    """Save raw price data as parquet."""
    cfg = get_data_config()
    out_dir = ROOT_DIR / cfg["storage"]["raw_dir"] / "prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    log.info("  Saved %s", path.relative_to(ROOT_DIR))
    return path


def upsert_to_db(df: pd.DataFrame) -> int:
    """Upsert price rows into TimescaleDB."""
    if df.empty:
        return 0
    return upsert_dataframe(
        df=df,
        table="prices",
        conflict_columns=["time", "ticker"],
    )


def backfill(tickers: list[str] | None = None, start: str | None = None) -> dict[str, int]:
    """Full historical backfill for all tickers."""
    cfg = get_data_config()
    tickers = tickers or get_ticker_list()
    start = start or cfg["history"]["start_date"]
    has_db = check_connection()

    # Also fetch VIX
    vix = get_tickers_config().get("vix_ticker")
    if vix and vix not in tickers:
        tickers = list(tickers) + [vix]

    results = {}
    for ticker in tickers:
        df = download_ticker(ticker, start=start)
        if df.empty:
            results[ticker] = 0
            continue

        save_parquet(df, ticker.replace("^", ""))

        if has_db:
            upsert_to_db(df)

        results[ticker] = len(df)

    total = sum(results.values())
    log.info("Backfill complete: %d tickers, %d total rows", len(results), total)
    return results


def daily_update(tickers: list[str] | None = None) -> dict[str, int]:
    """Fetch last 5 trading days and upsert (catches gaps from weekends/holidays)."""
    tickers = tickers or get_ticker_list()
    has_db = check_connection()

    vix = get_tickers_config().get("vix_ticker")
    if vix and vix not in tickers:
        tickers = list(tickers) + [vix]

    results = {}
    for ticker in tickers:
        log.info("Updating %s ...", ticker)
        df = yf.download(ticker, period="5d", auto_adjust=False, progress=False)

        if df.empty:
            results[ticker] = 0
            continue

        df = _normalize_columns(df, ticker)
        save_parquet(df, ticker.replace("^", ""))

        if has_db:
            upsert_to_db(df)

        results[ticker] = len(df)

    total = sum(results.values())
    log.info("Daily update: %d tickers, %d rows upserted", len(results), total)
    return results


def main():
    parser = argparse.ArgumentParser(description="Stock price collector")
    parser.add_argument("--mode", choices=["backfill", "daily"], default="backfill",
                        help="backfill: full history from 2015. daily: last 5 days.")
    parser.add_argument("--tickers", nargs="*", help="Override ticker list")
    parser.add_argument("--start", help="Override start date (backfill only)")
    args = parser.parse_args()

    if args.mode == "backfill":
        backfill(tickers=args.tickers, start=args.start)
    else:
        daily_update(tickers=args.tickers)


if __name__ == "__main__":
    main()
