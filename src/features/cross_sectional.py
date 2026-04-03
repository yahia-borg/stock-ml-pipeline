"""
Cross-sectional features — comparing stocks across the universe on each day.

These are among the highest-value features in academic literature and capture
information that single-stock features miss entirely.

Feature groups:
  1. Sector momentum + rotation
  2. Market breadth (advance-decline, % above EMAs)
  3. Return dispersion + correlation regime
  4. Relative value (market-adjusted, sector-adjusted)
  5. Cross-asset regime indicators (SPY-TLT, SPY-GLD correlations)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger("features.cross_sectional")

# Sector mapping: ticker -> sector ETF
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLK", "NVDA": "XLK", "META": "XLK",
    "AMZN": "XLK",  # Consumer discretionary but correlated with tech
    "TSLA": "XLK",
    "JPM": "XLF", "V": "XLF", "BRK-B": "XLF",
}

# ETFs used as cross-asset references
REFERENCE_ETFS = ["SPY", "QQQ", "TLT", "GLD", "HYG", "IWM"]
SECTOR_ETFS = ["XLF", "XLE", "XLK", "XLV", "XLI"]


def add_cross_sectional_features(
    all_prices: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Compute cross-sectional features across the entire ticker universe.

    Args:
        all_prices: Dict of {ticker: DataFrame} where each DataFrame has
                    at minimum [close, volume] columns indexed by date.

    Returns:
        Same dict with cross-sectional feature columns added to each DataFrame.
    """
    log.info("Computing cross-sectional features for %d tickers...", len(all_prices))

    # Build a combined returns matrix
    closes = {}
    for ticker, df in all_prices.items():
        if "close" in df.columns and len(df) > 0:
            closes[ticker] = df["close"]

    close_matrix = pd.DataFrame(closes)
    return_matrix = close_matrix.pct_change()

    # Compute universe-level features
    breadth = _compute_breadth(close_matrix, return_matrix)
    dispersion = _compute_dispersion(return_matrix)
    cross_asset = _compute_cross_asset_regime(return_matrix)
    sector_momentum = _compute_sector_momentum(return_matrix)

    # Merge back to each ticker
    for ticker, df in all_prices.items():
        df = df.copy()

        # ── Market breadth features ──
        for col in breadth.columns:
            df[col] = breadth[col].reindex(df.index)

        # ── Dispersion features ──
        for col in dispersion.columns:
            df[col] = dispersion[col].reindex(df.index)

        # ── Cross-asset regime ──
        for col in cross_asset.columns:
            df[col] = cross_asset[col].reindex(df.index)

        # ── Sector momentum ──
        for col in sector_momentum.columns:
            df[col] = sector_momentum[col].reindex(df.index)

        # ── Per-stock relative features ──
        df = _add_relative_features(df, ticker, return_matrix, close_matrix)

        all_prices[ticker] = df

    log.info("Cross-sectional features added to all tickers")
    return all_prices


def _compute_breadth(
    close_matrix: pd.DataFrame,
    return_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Market breadth indicators across the universe."""
    breadth = pd.DataFrame(index=close_matrix.index)

    # % of stocks above 200-day EMA
    ema_200 = close_matrix.ewm(span=200, min_periods=100).mean()
    above_200 = (close_matrix > ema_200).sum(axis=1)
    total = close_matrix.notna().sum(axis=1)
    breadth["pct_above_200ema"] = above_200 / total.replace(0, np.nan)

    # % of stocks above 50-day EMA
    ema_50 = close_matrix.ewm(span=50, min_periods=25).mean()
    above_50 = (close_matrix > ema_50).sum(axis=1)
    breadth["pct_above_50ema"] = above_50 / total.replace(0, np.nan)

    # Advance-decline ratio (rolling 10d)
    advances = (return_matrix > 0).sum(axis=1)
    declines = (return_matrix < 0).sum(axis=1)
    ad_ratio = advances / declines.replace(0, np.nan)
    breadth["ad_ratio_10d"] = ad_ratio.rolling(10).mean()

    # New 20d highs minus lows
    high_20 = close_matrix.rolling(20).max()
    low_20 = close_matrix.rolling(20).min()
    new_highs = (close_matrix == high_20).sum(axis=1)
    new_lows = (close_matrix == low_20).sum(axis=1)
    breadth["new_high_low_diff"] = (new_highs - new_lows) / total.replace(0, np.nan)

    # Breadth thrust detector (Zweig)
    breadth["breadth_thrust"] = (
        (breadth["ad_ratio_10d"].shift(10) < 0.4) &
        (breadth["ad_ratio_10d"] > 0.6)
    ).astype(int)

    return breadth


def _compute_dispersion(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional return dispersion (captures stock-picking vs factor regimes)."""
    disp = pd.DataFrame(index=return_matrix.index)

    # Cross-sectional std of returns
    disp["xs_vol_1d"] = return_matrix.std(axis=1)
    disp["xs_vol_20d"] = disp["xs_vol_1d"].rolling(20).mean()

    # Average pairwise correlation (rolling 20d)
    # High correlation = macro-driven, low = stock-specific
    corr_rolling = return_matrix.rolling(60, min_periods=20).corr()
    # Mean of upper triangle of correlation matrix per date
    mean_corr = []
    for date in return_matrix.index:
        try:
            corr = corr_rolling.loc[date]
            if isinstance(corr, pd.DataFrame) and corr.shape[0] > 1:
                mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
                vals = corr.values[mask]
                mean_corr.append(np.nanmean(vals))
            else:
                mean_corr.append(np.nan)
        except (KeyError, ValueError):
            mean_corr.append(np.nan)
    disp["avg_pairwise_corr"] = mean_corr

    return disp


def _compute_cross_asset_regime(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Cross-asset correlation regime indicators."""
    regime = pd.DataFrame(index=return_matrix.index)

    # SPY-TLT rolling correlation — key regime signal
    # Negative = normal (risk-on). Positive = stress (both falling or both rising)
    if "SPY" in return_matrix.columns and "TLT" in return_matrix.columns:
        for w in [20, 60]:
            regime[f"spy_tlt_corr_{w}d"] = (
                return_matrix["SPY"].rolling(w, min_periods=10)
                .corr(return_matrix["TLT"])
            )

    # SPY-GLD correlation
    if "SPY" in return_matrix.columns and "GLD" in return_matrix.columns:
        regime["spy_gld_corr_20d"] = (
            return_matrix["SPY"].rolling(20, min_periods=10)
            .corr(return_matrix["GLD"])
        )

    # SPY-HYG correlation (credit stress)
    if "SPY" in return_matrix.columns and "HYG" in return_matrix.columns:
        regime["spy_hyg_corr_20d"] = (
            return_matrix["SPY"].rolling(20, min_periods=10)
            .corr(return_matrix["HYG"])
        )

    # Variance risk premium: VIX - realized SPY vol (annualized)
    # Positive = fear premium, strong predictor of mean reversion
    if "SPY" in return_matrix.columns:
        realized_vol = return_matrix["SPY"].rolling(20).std() * np.sqrt(252) * 100
        regime["spy_realized_vol_20d"] = realized_vol

    # ── Cross-asset LEAD-LAG features (Tier 1 alpha) ──
    # Credit markets lead equities by 1-3 weeks

    # HYG-TLT spread momentum (credit quality deterioration leads equity drops)
    if "HYG" in return_matrix.columns and "TLT" in return_matrix.columns:
        hyg_ret = return_matrix["HYG"].rolling(5).sum()
        tlt_ret = return_matrix["TLT"].rolling(5).sum()
        regime["credit_spread_momentum_5d"] = hyg_ret - tlt_ret
        regime["credit_spread_momentum_20d"] = (
            return_matrix["HYG"].rolling(20).sum() - return_matrix["TLT"].rolling(20).sum()
        )

    # Bond-equity momentum divergence (TLT leading SPY)
    if "SPY" in return_matrix.columns and "TLT" in return_matrix.columns:
        for w in [5, 20]:
            spy_cum = return_matrix["SPY"].rolling(w).sum()
            tlt_cum = return_matrix["TLT"].rolling(w).sum()
            regime[f"bond_equity_divergence_{w}d"] = tlt_cum - spy_cum

    # Small-cap / large-cap relative strength (IWM/SPY leads market turns)
    if "IWM" in return_matrix.columns and "SPY" in return_matrix.columns:
        for w in [5, 20]:
            regime[f"smallcap_largecap_roc_{w}d"] = (
                return_matrix["IWM"].rolling(w).sum() -
                return_matrix["SPY"].rolling(w).sum()
            )

    # Gold/equity ratio momentum (risk-off signal)
    if "GLD" in return_matrix.columns and "SPY" in return_matrix.columns:
        regime["gold_equity_roc_20d"] = (
            return_matrix["GLD"].rolling(20).sum() -
            return_matrix["SPY"].rolling(20).sum()
        )

    # HYG momentum alone (credit stress early warning)
    if "HYG" in return_matrix.columns:
        regime["hyg_momentum_5d"] = return_matrix["HYG"].rolling(5).sum()
        regime["hyg_momentum_20d"] = return_matrix["HYG"].rolling(20).sum()

    return regime


def _compute_sector_momentum(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """Sector momentum and rotation signals."""
    sector_mom = pd.DataFrame(index=return_matrix.index)

    available_sectors = [s for s in SECTOR_ETFS if s in return_matrix.columns]
    if not available_sectors:
        return sector_mom

    for window in [5, 20]:
        sector_returns = {}
        for sector in available_sectors:
            sector_returns[sector] = return_matrix[sector].rolling(window).sum()

        sector_ret_df = pd.DataFrame(sector_returns)

        # Sector spread (best minus worst)
        sector_mom[f"sector_spread_{window}d"] = (
            sector_ret_df.max(axis=1) - sector_ret_df.min(axis=1)
        )

        # Top sector momentum
        sector_mom[f"top_sector_ret_{window}d"] = sector_ret_df.max(axis=1)
        sector_mom[f"bottom_sector_ret_{window}d"] = sector_ret_df.min(axis=1)

    return sector_mom


def _add_relative_features(
    df: pd.DataFrame,
    ticker: str,
    return_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Per-stock features relative to market and sector."""
    if ticker not in return_matrix.columns:
        return df

    stock_ret = return_matrix[ticker]

    # Market-relative momentum (alpha isolation)
    if "SPY" in return_matrix.columns:
        spy_ret = return_matrix["SPY"]
        for w in [5, 20, 60]:
            stock_cum = stock_ret.rolling(w).sum()
            spy_cum = spy_ret.rolling(w).sum()
            df[f"market_relative_ret_{w}d"] = stock_cum - spy_cum

        # Rolling beta to SPY (60d)
        cov = stock_ret.rolling(60, min_periods=30).cov(spy_ret)
        var = spy_ret.rolling(60, min_periods=30).var()
        df["rolling_beta_60d"] = cov / var.replace(0, np.nan)

    # Sector-relative momentum (industry-adjusted)
    sector_etf = SECTOR_MAP.get(ticker)
    if sector_etf and sector_etf in return_matrix.columns:
        sector_ret = return_matrix[sector_etf]
        for w in [5, 20]:
            stock_cum = stock_ret.rolling(w).sum()
            sector_cum = sector_ret.rolling(w).sum()
            df[f"sector_relative_ret_{w}d"] = stock_cum - sector_cum

        # Z-score of stock vs sector (mean reversion signal at extremes)
        spread = stock_ret - sector_ret
        spread_mean = spread.rolling(60).mean()
        spread_std = spread.rolling(60).std()
        df["sector_relative_zscore"] = (spread - spread_mean) / spread_std.replace(0, np.nan)

    # Relative volume vs universe median
    volumes = {}
    for t, tdf in [(t, close_matrix) for t in return_matrix.columns]:
        pass  # volumes handled differently since close_matrix doesn't have volume

    return df
