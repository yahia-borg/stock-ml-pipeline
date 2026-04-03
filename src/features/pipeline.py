"""
Feature pipeline orchestrator — builds the full feature matrix from raw data.

Orchestrates:
  1. Load raw price data (per ticker)
  2. Add technical features (momentum, vol, microstructure)
  3. Add calendar features (OPEX, quarter-end, holidays)
  4. Merge macro features (FRED indicators, forward-filled)
  5. Add cross-sectional features (breadth, sector momentum, correlations)
  6. Add VIX-based features (variance risk premium)
  7. Compute targets (adaptive thresholds, multi-horizon)
  8. Save final feature matrix

Usage:
  python -m src.features.pipeline                  # build all
  python -m src.features.pipeline --ticker AAPL     # build for one ticker
  python -m src.features.pipeline --skip-cross      # skip cross-sectional (faster)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ROOT_DIR, get_data_config, get_ticker_list, get_tickers_config
from src.utils.logger import get_logger

log = get_logger("features.pipeline")


def load_price_data(ticker: str) -> pd.DataFrame:
    """Load raw price data for a ticker from parquet."""
    cfg = get_data_config()
    path = ROOT_DIR / cfg["storage"]["raw_dir"] / "prices" / f"{ticker}.parquet"

    if not path.exists():
        log.warning("No price data for %s at %s", ticker, path)
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")

    # Ensure standard columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            log.error("Missing column '%s' in %s", col, ticker)
            return pd.DataFrame()

    log.info("Loaded %s: %d rows (%s to %s)",
             ticker, len(df), df.index.min().date(), df.index.max().date())
    return df


def add_vix_features(df: pd.DataFrame, vix_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add VIX-based features, including the variance risk premium.

    The variance risk premium (VRP = VIX - realized vol) is one of the
    strongest documented predictors of forward equity returns.
    """
    if vix_df is None:
        vix_ticker = get_tickers_config().get("vix_ticker", "^VIX")
        safe_name = vix_ticker.replace("^", "")
        vix_path = ROOT_DIR / get_data_config()["storage"]["raw_dir"] / "prices" / f"{safe_name}.parquet"

        if not vix_path.exists():
            log.warning("VIX data not found at %s", vix_path)
            return df

        vix_df = pd.read_parquet(vix_path)
        vix_df["time"] = pd.to_datetime(vix_df["time"])
        vix_df = vix_df.set_index("time").sort_index()

    # VIX level
    vix_close = vix_df["close"].reindex(df.index, method="ffill")
    df["vix_level"] = vix_close

    # VIX momentum
    df["vix_change_5d"] = vix_close.pct_change(5)
    df["vix_change_20d"] = vix_close.pct_change(20)

    # VIX regime (above/below 20 = fear threshold)
    df["vix_high_regime"] = (vix_close > 20).astype(int)
    df["vix_extreme"] = (vix_close > 30).astype(int)

    # Variance Risk Premium: VIX - realized vol (annualized)
    # Positive VRP = market pricing more fear than realized, historically bullish
    if "vol_20d" in df.columns:
        realized_vol_annualized = df["vol_20d"] * np.sqrt(252) * 100
        df["variance_risk_premium"] = vix_close - realized_vol_annualized
    elif "return_1d" in df.columns:
        realized = df["return_1d"].rolling(20).std() * np.sqrt(252) * 100
        df["variance_risk_premium"] = vix_close - realized

    # VIX term structure proxy (using 5d vs 20d VIX change)
    df["vix_term_slope"] = df["vix_change_5d"] - df["vix_change_20d"]

    return df


def build_single_ticker(
    ticker: str,
    macro_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
    sentiment_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the complete feature set for a single ticker."""
    from src.features.technical import add_technical_features
    from src.features.calendar import add_calendar_features
    from src.features.macro import build_macro_features, merge_macro_to_prices
    from src.features.sentiment_features import build_sentiment_features, merge_sentiment_to_prices
    from src.features.targets import add_targets

    df = load_price_data(ticker)
    if df.empty:
        return pd.DataFrame()

    log.info("Building features for %s ...", ticker)

    # 1. Technical features (~60+ columns)
    df = add_technical_features(df)

    # 2. Calendar features (~15 columns)
    df = add_calendar_features(df)

    # 3. Macro features (~25 columns)
    if macro_df is None:
        macro_df = build_macro_features()
    df = merge_macro_to_prices(df, macro_df)

    # 4. VIX features (~7 columns)
    df = add_vix_features(df, vix_df)

    # 5. Sentiment features (~20 columns)
    if sentiment_df is None:
        sentiment_df = build_sentiment_features()
    df = merge_sentiment_to_prices(df, sentiment_df)

    # 6. Fundamental features (GKX top-5 predictors)
    from src.collectors.fundamentals_collector import build_fundamental_features
    df = build_fundamental_features(df, ticker)

    # 7. Targets (multi-horizon)
    df = add_targets(df)

    # 8. Data cleaning
    df = _clean_features(df, ticker)

    # Add ticker column for identification
    df["ticker"] = ticker

    log.info("%s complete: %d rows, %d columns", ticker, len(df), len(df.columns))
    return df


def _clean_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Clean the feature matrix before saving.

    1. Drop warm-up rows where most features are NaN (rolling windows not filled)
    2. Cap extreme outlier values (prevent overflow in models)
    3. Replace inf/-inf with NaN then fill with 0
    4. Drop tickers with insufficient non-NaN data
    """
    n_before = len(df)

    # 1. Drop warm-up rows — require at least 50% of features to be non-NaN
    feature_cols = [c for c in df.columns
                    if not c.startswith(("fwd_", "label_", "threshold_"))
                    and c not in ("ticker", "time")]
    nan_pct = df[feature_cols].isna().mean(axis=1)
    df = df[nan_pct < 0.5]

    # 2. Cap extreme values — clip numeric features to ±1e10
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col.startswith(("fwd_", "label_", "threshold_")):
            continue
        df[col] = df[col].clip(-1e10, 1e10)

    # 3. Replace inf with NaN, then fill NaN with 0
    df = df.replace([np.inf, -np.inf], np.nan)

    n_after = len(df)
    if n_before != n_after:
        log.info("  %s: cleaned %d -> %d rows (dropped %d warm-up rows)",
                 ticker, n_before, n_after, n_before - n_after)

    return df


def build_all(
    tickers: list[str] | None = None,
    include_cross_sectional: bool = True,
) -> dict[str, pd.DataFrame]:
    """Build features for all tickers and optionally add cross-sectional features.

    Returns a dict of {ticker: feature_dataframe}.
    """
    tickers = tickers or get_ticker_list()

    # Pre-load shared data (computed once, reused for all tickers)
    from src.features.macro import build_macro_features
    from src.features.sentiment_features import build_sentiment_features
    log.info("Loading shared data (macro, VIX, sentiment) ...")
    macro_df = build_macro_features()
    sentiment_df = build_sentiment_features()

    vix_ticker = get_tickers_config().get("vix_ticker", "^VIX")
    safe_name = vix_ticker.replace("^", "")
    vix_path = ROOT_DIR / get_data_config()["storage"]["raw_dir"] / "prices" / f"{safe_name}.parquet"
    vix_df = None
    if vix_path.exists():
        vix_df = pd.read_parquet(vix_path)
        vix_df["time"] = pd.to_datetime(vix_df["time"])
        vix_df = vix_df.set_index("time").sort_index()

    # Build per-ticker features
    results = {}
    for ticker in tickers:
        df = build_single_ticker(ticker, macro_df=macro_df, vix_df=vix_df, sentiment_df=sentiment_df)
        if not df.empty:
            results[ticker] = df

    # Cross-sectional features (requires all tickers built first)
    if include_cross_sectional and len(results) > 1:
        log.info("Computing cross-sectional features ...")
        from src.features.cross_sectional import add_cross_sectional_features
        results = add_cross_sectional_features(results)

    return results


def save_feature_matrix(
    results: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> Path:
    """Save the feature matrix as parquet files (one per ticker + one combined)."""
    cfg = get_data_config()
    output_dir = output_dir or ROOT_DIR / cfg["storage"]["processed_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-ticker files
    for ticker, df in results.items():
        path = output_dir / f"{ticker}_features.parquet"
        df.to_parquet(path)
        log.info("Saved %s (%d rows, %d cols)", path.name, len(df), len(df.columns))

    # Combined file (all tickers stacked)
    if results:
        combined = pd.concat(results.values(), ignore_index=False)
        combined_path = output_dir / "feature_matrix.parquet"
        combined.to_parquet(combined_path)
        log.info("Combined matrix: %d rows, %d cols -> %s",
                 len(combined), len(combined.columns), combined_path.name)
        return combined_path

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument("--ticker", nargs="*", help="Build for specific ticker(s) only")
    parser.add_argument("--skip-cross", action="store_true",
                        help="Skip cross-sectional features (faster)")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    args = parser.parse_args()

    tickers = args.ticker if args.ticker else None

    results = build_all(
        tickers=tickers,
        include_cross_sectional=not args.skip_cross,
    )

    if not results:
        log.error("No features built — check that price data exists (run `make backfill` first)")
        return

    save_feature_matrix(results, output_dir=args.output_dir)

    # Summary
    log.info("=" * 50)
    log.info("Feature pipeline complete!")
    for ticker, df in results.items():
        n_features = len([c for c in df.columns
                          if c not in ["ticker"] and not c.startswith("fwd_") and not c.startswith("label_")])
        log.info("  %s: %d rows, %d features, %d target cols",
                 ticker, len(df), n_features,
                 len([c for c in df.columns if c.startswith("fwd_") or c.startswith("label_")]))


if __name__ == "__main__":
    main()
