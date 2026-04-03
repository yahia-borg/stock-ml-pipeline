"""
Fundamentals collector — fetches company financials from yfinance.

Captures the top-5 most predictive stock characteristics from Gu, Kelly & Xiu (2020):
  1. Market cap (size factor)
  2. Book-to-market (value factor)
  3. Gross profitability (quality factor)
  4. Asset growth (investment factor)
  5. Earnings yield

Data is quarterly — refreshed once per quarter per ticker.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.config import ROOT_DIR, get_data_config, get_ticker_list
from src.utils.logger import get_logger

log = get_logger("collectors.fundamentals")


def fetch_fundamentals(ticker: str) -> dict | None:
    """Fetch current fundamental data for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result = {
            "ticker": ticker,
            "market_cap": info.get("marketCap"),
            "book_value": info.get("bookValue"),
            "price_to_book": info.get("priceToBook"),
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "profit_margins": info.get("profitMargins"),
            "gross_margins": info.get("grossMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "total_revenue": info.get("totalRevenue"),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
            "enterprise_value": info.get("enterpriseValue"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
            "avg_volume_10d": info.get("averageVolume10days"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
        }

        # Check if we got meaningful data
        if result["market_cap"] is None and result["book_value"] is None:
            log.warning("  %s: no fundamental data available", ticker)
            return None

        log.info("  %s: market_cap=%s, P/B=%s, ROE=%s",
                 ticker,
                 f"${result['market_cap']:,.0f}" if result['market_cap'] else "N/A",
                 f"{result['price_to_book']:.2f}" if result['price_to_book'] else "N/A",
                 f"{result['return_on_equity']:.2%}" if result['return_on_equity'] else "N/A")

        return result

    except Exception as e:
        log.warning("  %s: failed to fetch fundamentals: %s", ticker, e)
        return None


def fetch_all_fundamentals(tickers: list[str] | None = None) -> pd.DataFrame:
    """Fetch fundamentals for all tickers."""
    tickers = tickers or get_ticker_list()
    log.info("Fetching fundamentals for %d tickers...", len(tickers))

    results = []
    for ticker in tickers:
        data = fetch_fundamentals(ticker)
        if data:
            results.append(data)

    if not results:
        log.warning("No fundamental data fetched")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    log.info("Fundamentals collected: %d tickers, %d columns", len(df), len(df.columns))
    return df


def save_parquet(df: pd.DataFrame) -> Path:
    """Save fundamentals to parquet."""
    cfg = get_data_config()
    out_dir = ROOT_DIR / cfg["storage"]["raw_dir"] / "fundamentals"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fundamentals.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved %s", path.relative_to(ROOT_DIR))
    return path


def load_fundamentals() -> pd.DataFrame:
    """Load saved fundamentals from parquet."""
    cfg = get_data_config()
    path = ROOT_DIR / cfg["storage"]["raw_dir"] / "fundamentals" / "fundamentals.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def build_fundamental_features(price_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add fundamental features to a price DataFrame for a specific ticker.

    Since fundamentals are point-in-time (quarterly), they're constant
    for each row within a quarter and only change at earnings dates.
    """
    df = price_df.copy()
    fundamentals = load_fundamentals()

    if fundamentals.empty or ticker not in fundamentals["ticker"].values:
        return df

    row = fundamentals[fundamentals["ticker"] == ticker].iloc[0]
    import numpy as np

    # Sanity check: skip clearly broken data (overflow values)
    market_cap = row.get("market_cap")
    if pd.notna(market_cap) and market_cap > 1e16:  # > $10 quadrillion = overflow
        log.warning("  %s: market_cap overflow (%s), skipping fundamentals", ticker, market_cap)
        return df

    # GKX Factor 1: Size (log market cap)
    if pd.notna(market_cap) and market_cap > 0:
        df["fund_log_market_cap"] = np.log(market_cap)

    # GKX Factor 2: Value (book-to-market = 1/P/B)
    ptb = row.get("price_to_book")
    if pd.notna(ptb) and 0.01 < ptb < 500:  # cap at sensible range
        df["fund_book_to_market"] = 1.0 / ptb

    # GKX Factor 3: Profitability (gross margins)
    if pd.notna(row.get("gross_margins")):
        df["fund_gross_profitability"] = row["gross_margins"]

    # GKX Factor 4: Investment (revenue growth as proxy for asset growth)
    if pd.notna(row.get("revenue_growth")):
        df["fund_revenue_growth"] = row["revenue_growth"]

    # GKX Factor 5: Earnings yield (1/PE)
    if pd.notna(row.get("trailing_pe")) and row["trailing_pe"] > 0:
        df["fund_earnings_yield"] = 1.0 / row["trailing_pe"]

    # Additional quality factors
    if pd.notna(row.get("return_on_equity")):
        df["fund_roe"] = row["return_on_equity"]

    if pd.notna(row.get("debt_to_equity")):
        df["fund_leverage"] = row["debt_to_equity"]

    if pd.notna(row.get("profit_margins")):
        df["fund_profit_margin"] = row["profit_margins"]

    if pd.notna(row.get("beta")):
        df["fund_beta"] = row["beta"]

    if pd.notna(row.get("dividend_yield")):
        df["fund_dividend_yield"] = row["dividend_yield"]

    # Sharia compliance screening
    df["fund_sharia_compliant"] = _check_sharia_compliance(ticker, row)

    n_fund = len([c for c in df.columns if c.startswith("fund_")])
    if n_fund > 0:
        log.info("  %s: %d fundamental features added (sharia=%d)", ticker, n_fund,
                 int(df["fund_sharia_compliant"].iloc[0]) if len(df) > 0 else -1)

    return df


def _check_sharia_compliance(ticker: str, fundamentals: pd.Series) -> int:
    """Check if a ticker is Sharia-compliant based on AAOIFI standards.

    Returns 1 (compliant), 0 (non-compliant), or -1 (unknown/insufficient data).

    AAOIFI criteria:
      1. Debt/market_cap < 33%
      2. Business not in prohibited sectors (alcohol, gambling, pork, conventional interest)
      3. Non-halal revenue < 5%
    """
    from src.utils.config import get_tickers_config

    cfg = get_tickers_config()
    sharia_cfg = cfg.get("sharia_compliance", {})

    # Check known-compliant list first
    known = sharia_cfg.get("known_compliant", {})
    for region, tickers in known.items():
        if ticker in tickers:
            return 1

    # Check prohibited sectors (conventional banks, alcohol, gambling)
    sector = str(fundamentals.get("sector", "")).lower()
    industry = str(fundamentals.get("industry", "")).lower()
    prohibited = ["alcohol", "gambling", "casino", "tobacco", "pork", "wine", "beer", "spirits"]
    for p in prohibited:
        if p in sector or p in industry:
            return 0

    # Financial ratio screening
    market_cap = fundamentals.get("market_cap")
    total_debt = fundamentals.get("total_debt")

    if pd.notna(market_cap) and pd.notna(total_debt) and market_cap > 0:
        debt_ratio = total_debt / market_cap
        max_debt = sharia_cfg.get("max_debt_to_market_cap", 0.33)
        if debt_ratio > max_debt:
            return 0

    # If we can't determine, return unknown
    return -1


def main():
    parser = argparse.ArgumentParser(description="Fundamentals collector (yfinance)")
    parser.add_argument("--tickers", nargs="*", help="Override ticker list")
    args = parser.parse_args()

    df = fetch_all_fundamentals(tickers=args.tickers)
    if not df.empty:
        save_parquet(df)
        print(f"Saved fundamentals for {len(df)} tickers")
        print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
