"""
Global and regional events collector — aggregates market-moving events.

Sources (all free, no API key needed):
  1. Economic calendar (FRED, central bank meetings, earnings dates)
  2. Geopolitical events (GDELT — already handled by gdelt_collector)
  3. Commodity shocks (oil, gold, shipping via yfinance)
  4. Currency moves (EGP, SAR, AED via yfinance)
  5. Regional indices (EGX30 proxy, TASI proxy)

This collector creates EVENT FEATURES — binary/continuous signals
for market-moving events that complement the news sentiment pipeline.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from src.utils.config import ROOT_DIR, get_data_config, get_tickers_config
from src.utils.logger import get_logger

log = get_logger("collectors.events")


def collect_commodity_signals(days: int = 60) -> pd.DataFrame:
    """Fetch commodity prices that signal regional market moves.

    Oil → Saudi/UAE/Qatar (direct correlation)
    Gold → Egypt (gold is major export + inflation hedge)
    Shipping (Baltic Dry Index proxy) → Suez Canal / global trade
    """
    signals = {}

    commodities = {
        "CL=F": "crude_oil",       # WTI crude oil futures
        "BZ=F": "brent_oil",       # Brent crude (more relevant for MENA)
        "GC=F": "gold",            # Gold futures
        "SI=F": "silver",          # Silver futures
        "NG=F": "natural_gas",     # Natural gas
        "BDRY": "baltic_dry",      # Baltic Dry Index ETF (shipping proxy)
    }

    for ticker, name in commodities.items():
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False)
            if not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                signals[f"commodity_{name}"] = close.pct_change()
                signals[f"commodity_{name}_5d"] = close.pct_change(5)
                signals[f"commodity_{name}_20d"] = close.pct_change(20)
                # Shock detector: >2 std move
                rolling_std = close.pct_change().rolling(20).std()
                daily_ret = close.pct_change()
                signals[f"commodity_{name}_shock"] = (daily_ret.abs() > 2 * rolling_std).astype(int)
                log.info("  %s (%s): %d rows", name, ticker, len(close))
        except Exception as e:
            log.warning("  %s failed: %s", ticker, e)

    if not signals:
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    return result


def collect_fx_signals(days: int = 60) -> pd.DataFrame:
    """Fetch currency moves for MENA markets.

    EGP devaluation → massive EGX impact
    SAR is pegged to USD (moves signal stress)
    AED is pegged to USD
    """
    signals = {}

    fx_pairs = {
        "EGPUSD=X": "egp_usd",     # Egyptian Pound
        "SARUSD=X": "sar_usd",     # Saudi Riyal
        "AEDUSD=X": "aed_usd",     # UAE Dirham
        "QARUSD=X": "qar_usd",     # Qatari Riyal
        "DX-Y.NYB": "dxy",         # US Dollar Index
    }

    for ticker, name in fx_pairs.items():
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False)
            if not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                signals[f"fx_{name}"] = close.pct_change()
                signals[f"fx_{name}_5d"] = close.pct_change(5)
                # EGP devaluation shock (>1% daily move for a pegged currency = huge)
                if name == "egp_usd":
                    signals["fx_egp_deval_shock"] = (close.pct_change().abs() > 0.01).astype(int)
                log.info("  %s (%s): %d rows", name, ticker, len(close))
        except Exception as e:
            log.warning("  %s failed: %s", ticker, e)

    if not signals:
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    return result


def collect_regional_index_signals(days: int = 60) -> pd.DataFrame:
    """Fetch regional index ETFs as market regime signals."""
    signals = {}

    indices = {
        "EGPT": "egx_proxy",       # Egypt market proxy
        "KSA": "tasi_proxy",       # Saudi market proxy
        "UAE": "adx_proxy",        # UAE market proxy
        "QAT": "qse_proxy",        # Qatar market proxy
        "EEM": "emerging_markets", # MSCI Emerging Markets
        "FM": "frontier_markets",  # iShares MSCI Frontier Markets
    }

    for ticker, name in indices.items():
        try:
            df = yf.download(ticker, period=f"{days}d", progress=False)
            if not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                signals[f"idx_{name}_ret"] = close.pct_change()
                signals[f"idx_{name}_ret_5d"] = close.pct_change(5)
                signals[f"idx_{name}_vol_20d"] = close.pct_change().rolling(20).std()
                log.info("  %s (%s): %d rows", name, ticker, len(close))
        except Exception as e:
            log.warning("  %s failed: %s", ticker, e)

    if not signals:
        return pd.DataFrame()

    result = pd.DataFrame(signals)
    result.index = pd.to_datetime(result.index)
    if result.index.tz is not None:
        result.index = result.index.tz_convert(None)
    return result


def collect_all_events(days: int = 60) -> pd.DataFrame:
    """Collect all event signals and combine."""
    log.info("Collecting event signals (%d days)...", days)

    dfs = []

    log.info("Commodity signals...")
    commodities = collect_commodity_signals(days)
    if not commodities.empty:
        dfs.append(commodities)

    log.info("FX signals...")
    fx = collect_fx_signals(days)
    if not fx.empty:
        dfs.append(fx)

    log.info("Regional index signals...")
    indices = collect_regional_index_signals(days)
    if not indices.empty:
        dfs.append(indices)

    if not dfs:
        log.warning("No event signals collected")
        return pd.DataFrame()

    combined = pd.concat(dfs, axis=1)
    combined = combined.sort_index()

    log.info("Event signals: %d rows, %d features", len(combined), len(combined.columns))
    return combined


def save_events(df: pd.DataFrame) -> None:
    """Save event signals to parquet."""
    cfg = get_data_config()
    out_dir = ROOT_DIR / cfg["storage"]["raw_dir"] / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "event_signals.parquet"
    df.to_parquet(path)
    log.info("Saved %s", path.relative_to(ROOT_DIR))


def load_events() -> pd.DataFrame:
    """Load saved event signals."""
    cfg = get_data_config()
    path = ROOT_DIR / cfg["storage"]["raw_dir"] / "events" / "event_signals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def merge_events_to_prices(
    price_df: pd.DataFrame,
    events_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge event signals into price DataFrame."""
    if events_df is None:
        events_df = load_events()

    if events_df.empty:
        return price_df

    price_df = price_df.copy()

    if "time" in price_df.columns:
        price_idx = pd.to_datetime(price_df["time"])
    else:
        price_idx = pd.to_datetime(price_df.index)

    if price_idx.tz is not None:
        price_idx = price_idx.tz_convert(None)
    if events_df.index.tz is not None:
        events_df.index = events_df.index.tz_convert(None)

    reindexed = events_df.reindex(price_idx, method="ffill")

    for col in reindexed.columns:
        price_df[col] = reindexed[col].values

    log.info("Merged %d event features into price data", len(events_df.columns))
    return price_df


def main():
    parser = argparse.ArgumentParser(description="Global/regional events collector")
    parser.add_argument("--days", type=int, default=365, help="Lookback days")
    args = parser.parse_args()

    df = collect_all_events(days=args.days)
    if not df.empty:
        save_events(df)
        print(f"Collected {len(df.columns)} event features over {len(df)} days")


if __name__ == "__main__":
    main()
