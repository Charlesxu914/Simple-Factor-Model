"""
Equity factor model data loader using Yahoo Finance.

This script implements a data pipeline to fetch historical data for an
equity universe (S&P 500 or Russell 3000), creating all the necessary
DataFrames for the factor model.
"""

import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import random
import time as time_module

from toraniko.ticker_universe import get_tickers, UNIVERSE_SP500, UNIVERSE_RUSSELL3000

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('yfinance_data_loader')

# Configuration — update these to control the data range and universe
START_DATE = "2020-01-01"  # Earliest date to include
END_DATE = datetime.now().strftime("%Y-%m-%d")  # Defaults to today
OUTPUT_DIR = "./data"
UNIVERSE = UNIVERSE_SP500  # Default universe; override via main(universe=...)
TEST_MODE = False  # Set True to use only TEST_TICKERS (faster iteration)
TEST_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']
FORCE_REFRESH = False  # Set True to re-fetch even if cached parquet files exist

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance package not available. Install using: pip install yfinance")
    YFINANCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers: retry, rate-limit params, progress
# ---------------------------------------------------------------------------
def _retry_with_backoff(func, max_retries=3, base_delay=2.0, label=""):
    """Execute *func()*, retrying with exponential backoff on failure."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(
                f"{label} attempt {attempt + 1} failed: {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time_module.sleep(delay)


def _get_rate_params(n_tickers: int) -> dict:
    """Return batch sizes and delays tuned for the universe size."""
    if n_tickers > 1000:
        return {
            "price_batch": 50, "price_delay": 1,
            "info_batch": 10, "info_delay": 2,
            "fund_batch": 10, "fund_delay": 3,
        }
    return {
        "price_batch": 100, "price_delay": 0,
        "info_batch": 20, "info_delay": 1,
        "fund_batch": 20, "fund_delay": 2,
    }


def _eta_str(start_time, done, total):
    """Return a human-readable ETA string."""
    if done == 0:
        return "calculating..."
    elapsed = time_module.time() - start_time
    rate = elapsed / done
    remaining = rate * (total - done)
    if remaining < 60:
        return f"{remaining:.0f}s"
    return f"{remaining / 60:.1f}min"


def _get_output_dir(universe: str) -> str:
    """Return the universe-specific data directory."""
    return os.path.join(OUTPUT_DIR, universe)
    
def get_sp500_tickers():
    """Legacy wrapper — prefer ``ticker_universe.get_tickers()``."""
    if TEST_MODE:
        logger.info("TEST MODE: Using a small set of test tickers")
        return TEST_TICKERS
    return get_tickers(UNIVERSE_SP500)


def fetch_price_data(tickers, start_date=START_DATE, end_date=END_DATE, batch_size=100):
    """
    Fetches historical price data for the given tickers in batches to avoid API limits.
    Returns a Polars DataFrame with columns: date, symbol, adj_close, volume
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("Cannot fetch real price data. yfinance package not available.")
        return None
        
    logger.info(f"Fetching price data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    all_data = []
    # Process tickers in batches
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}: {len(batch_tickers)} tickers")
        
        try:
            # Download data for this batch (with retry)
            data = _retry_with_backoff(
                lambda bt=batch_tickers: yf.download(
                    bt,
                    start=start_date,
                    end=end_date,
                    group_by='ticker',
                    auto_adjust=False,
                    progress=False,
                    threads=True,
                    ignore_tz=True,
                ),
                label=f"price batch {i // batch_size + 1}",
            )
            
            # Handle case of single ticker (different structure)
            if len(batch_tickers) == 1:
                ticker = batch_tickers[0]
                if data.empty:
                    logger.warning(f"No data found for {ticker}")
                    continue
                    
                # For a single ticker, the DataFrame structure is different
                data_subset = data[['Adj Close', 'Volume']].copy()
                data_subset.columns = ['adj_close', 'volume']
                data_subset['symbol'] = ticker
                data_subset.reset_index(inplace=True)  # Make date a column
                all_data.append(data_subset)
            else:
                # For multiple tickers, process each one
                for ticker in batch_tickers:
                    if ticker not in data.columns.levels[0]:
                        logger.warning(f"No data found for {ticker}")
                        continue
                        
                    data_subset = data[ticker][['Adj Close', 'Volume']].copy()
                    data_subset.columns = ['adj_close', 'volume']
                    data_subset['symbol'] = ticker
                    data_subset.reset_index(inplace=True)  # Make date a column
                    all_data.append(data_subset)
        
        except Exception as e:
            logger.error(f"Error fetching batch {i//batch_size + 1}: {e}")

        # Rate-limit delay between batches (important for large universes)
        if i + batch_size < len(tickers):
            price_delay = _get_rate_params(len(tickers))["price_delay"]
            if price_delay > 0:
                time_module.sleep(price_delay)

    if not all_data:
        logger.error("No price data fetched for any tickers")
        return None
        
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Convert to polars
    price_df = pl.from_pandas(combined_df)
    
    # Clean the data
    # First, ensure the date column has a consistent name
    if "Date" in price_df.columns:
        price_df = price_df.rename({"Date": "date"})
    elif "index" in price_df.columns:
        price_df = price_df.rename({"index": "date"})
    
    # Then apply other cleaning operations
    price_df = (
        price_df
        .filter(pl.col("adj_close").is_not_null())  # Remove rows with missing prices
        .filter(pl.col("adj_close") > 0)  # Remove invalid prices
        .sort(["symbol", "date"])  # Sort for efficient processing
    )
    
    logger.info(f"Price data fetched and cleaned. Shape: {price_df.shape}")
    return price_df
    
def create_returns_df(price_df):
    """
    Calculates daily returns from price data.
    Returns a Polars DataFrame with columns: date, symbol, asset_returns
    """
    if price_df is None or price_df.is_empty():
        logger.error("Cannot calculate returns: No price data available")
        return None
        
    logger.info("Calculating daily returns from price data")
    
    returns_df = (
        price_df.sort(["symbol", "date"])
        .with_columns([
            (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1)
            .alias("asset_returns")
        ])
        .select(["date", "symbol", "asset_returns"])
        .filter(~pl.col("asset_returns").is_null())  # Remove first day for each stock (null return)
    )
    
    # Check for extreme returns (could be data errors)
    extreme_returns = returns_df.filter(pl.col("asset_returns").abs() > 0.5)
    if extreme_returns.height > 0:
        logger.warning(f"Found {extreme_returns.height} extreme returns (>50%). Consider investigating.")
        
    logger.info(f"Returns calculation complete. Shape: {returns_df.shape}")
    return returns_df
    
def fetch_ticker_info(tickers, batch_size=20, checkpoint_dir=None):
    """
    Fetches .info for all tickers in a single pass, returning shares outstanding
    and sector data. This avoids making duplicate API calls.

    Supports checkpointing for large universes (Russell 3000).

    Returns: (shares_data: dict, sector_data: list[dict])
    """
    if not YFINANCE_AVAILABLE:
        return {}, []

    logger.info(f"Fetching ticker info (shares + sector) for {len(tickers)} tickers")

    shares_data = {}
    sector_data = []
    completed_tickers = set()

    # Load checkpoint if available
    if checkpoint_dir:
        cp_file = os.path.join(checkpoint_dir, "ticker_info_checkpoint.parquet")
        if os.path.exists(cp_file):
            cp_df = pl.read_parquet(cp_file)
            for row in cp_df.iter_rows(named=True):
                completed_tickers.add(row["symbol"])
                if row.get("shares") and row["shares"] > 0:
                    shares_data[row["symbol"]] = row["shares"]
                if row.get("sector"):
                    sector_data.append({"symbol": row["symbol"], "sector": row["sector"]})
            logger.info(f"Resumed from checkpoint: {len(completed_tickers)} tickers already done")

    remaining = [t for t in tickers if t not in completed_tickers]
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    info_delay = _get_rate_params(len(tickers))["info_delay"]
    start_time = time_module.time()

    for i in range(0, len(remaining), batch_size):
        batch_tickers = remaining[i:i + batch_size]
        batch_num = i // batch_size + 1
        done = len(completed_tickers) + i + len(batch_tickers)
        eta = _eta_str(start_time, i + len(batch_tickers), len(remaining))
        logger.info(
            f"Ticker info batch {batch_num}/{total_batches} "
            f"({done}/{len(tickers)} total, ETA: {eta})"
        )

        for ticker in batch_tickers:
            try:
                info = _retry_with_backoff(
                    lambda t=ticker: yf.Ticker(t).info,
                    label=f"info({ticker})",
                )

                # Shares outstanding
                for field in ['sharesOutstanding', 'impliedSharesOutstanding']:
                    if field in info and info[field] and info[field] > 0:
                        shares_data[ticker] = info[field]
                        break

                # Sector
                if 'sector' in info and info['sector']:
                    sector_data.append({'symbol': ticker, 'sector': info['sector']})
            except Exception as e:
                logger.error(f"Error fetching info for {ticker}: {e}")

        # Save checkpoint every 100 tickers
        if checkpoint_dir and (i + batch_size) % 100 < batch_size:
            _save_info_checkpoint(checkpoint_dir, shares_data, sector_data)

        time_module.sleep(info_delay)

    # Final checkpoint
    if checkpoint_dir:
        _save_info_checkpoint(checkpoint_dir, shares_data, sector_data)

    logger.info(f"Got shares for {len(shares_data)} tickers, sector for {len(sector_data)} tickers")
    return shares_data, sector_data


def _save_info_checkpoint(checkpoint_dir, shares_data, sector_data):
    """Save ticker info progress to a checkpoint file."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    rows = []
    all_symbols = set(shares_data.keys()) | {d["symbol"] for d in sector_data}
    sector_map = {d["symbol"]: d["sector"] for d in sector_data}
    for sym in all_symbols:
        rows.append({
            "symbol": sym,
            "shares": shares_data.get(sym, 0.0),
            "sector": sector_map.get(sym, ""),
        })
    if rows:
        pl.DataFrame(rows).write_parquet(
            os.path.join(checkpoint_dir, "ticker_info_checkpoint.parquet")
        )


def fetch_market_cap_data(tickers, price_df=None, shares_data=None, batch_size=20):
    """
    Computes historical market cap using shares outstanding and price data.
    If shares_data is not provided, fetches it from yfinance.
    """
    if price_df is None or price_df.is_empty():
        logger.error("Price data is required to calculate market cap")
        return None

    if shares_data is None:
        shares_data, _ = fetch_ticker_info(tickers, batch_size)

    if not shares_data:
        logger.error("No shares outstanding data available")
        return None

    logger.info("Calculating market cap using price data and shares outstanding")

    mkt_cap_df = (
        price_df.select(["date", "symbol", "adj_close"])
        .with_columns([
            pl.col("symbol").map_elements(
                lambda s: float(shares_data[s]) if s in shares_data else None,
                return_dtype=pl.Float64
            ).alias("shares_outstanding")
        ])
        .filter(pl.col("shares_outstanding").is_not_null())
        .with_columns([
            (pl.col("adj_close") * pl.col("shares_outstanding")).alias("market_cap")
        ])
        .select(["date", "symbol", "market_cap"])
    )

    logger.info(f"Market cap calculation complete. Shape: {mkt_cap_df.shape}")
    return mkt_cap_df


def fetch_sector_data(tickers, sector_data=None, batch_size=20):
    """
    Returns a DataFrame with ticker -> sector mapping.
    If sector_data list is not provided, fetches it from yfinance.
    """
    if sector_data is None:
        _, sector_data = fetch_ticker_info(tickers, batch_size)

    if not sector_data:
        logger.error("No sector data available")
        return None

    sector_df = pl.DataFrame(sector_data)
    logger.info(f"Sector data: {sector_df.height} tickers, {sector_df['sector'].n_unique()} unique sectors")
    return sector_df
    
def create_sector_exposure_df(sector_df, price_df):
    """
    Creates a daily sector exposure DataFrame with one-hot encoding.
    Returns a DataFrame with columns: date, symbol, SECTOR1, SECTOR2, etc.
    """
    if sector_df is None or sector_df.is_empty() or price_df is None or price_df.is_empty():
        logger.error("Cannot create sector exposure DataFrame: Missing input data")
        return None
        
    logger.info("Creating daily sector exposure DataFrame with one-hot encoding")
    
    try:
        # Get all unique date-symbol combinations from price data
        all_dates_symbols = price_df.select(["date", "symbol"]).unique()
        
        # Join sector information to all date-symbol pairs
        daily_sector_df = all_dates_symbols.join(
            sector_df,
            on="symbol",
            how="left"
        )
        
        # Check for missing sector data
        missing_sectors = daily_sector_df.filter(pl.col("sector").is_null()).select("symbol").unique()
        if not missing_sectors.is_empty():
            logger.warning(f"Missing sector data for {missing_sectors.height} symbols. These will be dropped.")
            daily_sector_df = daily_sector_df.filter(~pl.col("sector").is_null())
        
        # One-hot encode sectors
        if 'sector' in daily_sector_df.columns and not daily_sector_df["sector"].is_null().all():
            unique_sectors = sorted(daily_sector_df["sector"].drop_nulls().unique().to_list())
            
            if len(unique_sectors) == 0:
                logger.error("No valid sectors found after cleaning")
                return None
                
            logger.info(f"Creating one-hot encoding for {len(unique_sectors)} sectors")
            
            # Alternative approach to creating sector dummies if pivot has issues
            sector_dummies_df = all_dates_symbols.clone()
            
            # Add columns for each sector, initialized to 0
            for sector in unique_sectors:
                sector_dummies_df = sector_dummies_df.with_columns(
                    pl.lit(0).cast(pl.Int8).alias(sector)
                )
            
            # Update sector values based on the mapping
            for sector in unique_sectors:
                # For each sector, find symbols belonging to that sector
                sector_symbols = sector_df.filter(pl.col("sector") == sector)["symbol"].unique().to_list()
                
                if sector_symbols:
                    # Update the sector column to 1 for matching symbols
                    sector_dummies_df = sector_dummies_df.with_columns(
                        pl.when(pl.col("symbol").is_in(sector_symbols))
                        .then(pl.lit(1))
                        .otherwise(pl.col(sector))
                        .alias(sector)
                    )
            
            logger.info(f"Sector exposure DataFrame created with {len(unique_sectors)} sectors")
            return sector_dummies_df
        else:
            logger.error("No sector information available after joining")
            return None
            
    except Exception as e:
        logger.error(f"Error creating sector exposure DataFrame: {e}")
        # Create a minimal DataFrame with just date and symbol
        return price_df.select(["date", "symbol"]).unique()
        
def fetch_value_metrics_data(tickers, price_df=None, start_date=START_DATE, end_date=END_DATE,
                             batch_size=20, checkpoint_dir=None):
    """
    Fetches fundamental data needed for value metrics.
    Returns a DataFrame with: date, symbol, book_price, sales_price, cf_price

    Supports checkpointing for large universes.
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("Cannot fetch real value metrics. yfinance package not available.")
        return None

    if price_df is None or price_df.is_empty():
        logger.error("Price data is required to calculate value metrics")
        return None

    logger.info(f"Fetching value metrics data for {len(tickers)} tickers")

    # Collect fundamental snapshots: list of dicts with {date, symbol, book_value, revenue, cash_flow}
    fund_rows = []
    fetched_count = 0
    completed_tickers = set()

    # Load checkpoint if available
    if checkpoint_dir:
        cp_file = os.path.join(checkpoint_dir, "fund_data_checkpoint.parquet")
        if os.path.exists(cp_file):
            cp_df = pl.read_parquet(cp_file)
            completed_tickers = set(cp_df["symbol"].unique().to_list())
            fund_rows = cp_df.to_dicts()
            fetched_count = len(completed_tickers)
            logger.info(f"Resumed from checkpoint: {len(completed_tickers)} tickers already done")

    remaining = [t for t in tickers if t not in completed_tickers]
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    fund_delay = _get_rate_params(len(tickers))["fund_delay"]
    start_time = time_module.time()

    for i in range(0, len(remaining), batch_size):
        batch_tickers = remaining[i:i + batch_size]
        batch_num = i // batch_size + 1
        done = len(completed_tickers) + i + len(batch_tickers)
        eta = _eta_str(start_time, i + len(batch_tickers), len(remaining))
        logger.info(
            f"Fundamental data batch {batch_num}/{total_batches} "
            f"({done}/{len(tickers)} total, ETA: {eta})"
        )

        for ticker in batch_tickers:
            try:
                stock = _retry_with_backoff(
                    lambda t=ticker: yf.Ticker(t),
                    label=f"Ticker({ticker})",
                )

                # Get balance sheet (quarterly preferred, fall back to annual)
                balance_sheet = stock.quarterly_balance_sheet
                if balance_sheet.empty:
                    balance_sheet = stock.balance_sheet
                financials = stock.quarterly_financials
                cash_flow = stock.quarterly_cashflow

                if balance_sheet.empty and financials.empty and cash_flow.empty:
                    continue

                # Collect all reporting dates across the three statements
                all_dates = sorted(set(
                    balance_sheet.columns.tolist() +
                    financials.columns.tolist() +
                    cash_flow.columns.tolist()
                ))

                ticker_has_data = False
                for date in all_dates:
                    row = {'fund_date': pd.Timestamp(date), 'symbol': ticker}

                    # Book value = Total Assets - Total Liabilities
                    if not balance_sheet.empty and 'Total Assets' in balance_sheet.index and date in balance_sheet.columns:
                        assets = balance_sheet.loc['Total Assets', date]
                        liab_key = 'Total Liabilities Net Minority Interest'
                        liabilities = balance_sheet.loc[liab_key, date] if liab_key in balance_sheet.index and date in balance_sheet.columns else None
                        if assets is not None and liabilities is not None and not pd.isna(assets) and not pd.isna(liabilities):
                            row['book_value'] = float(assets - liabilities)

                    # Revenue (annualized from quarterly)
                    if not financials.empty and 'Total Revenue' in financials.index and date in financials.columns:
                        revenue = financials.loc['Total Revenue', date]
                        if revenue is not None and not pd.isna(revenue):
                            row['revenue'] = float(revenue) * 4

                    # Cash flow (annualized from quarterly)
                    if not cash_flow.empty and 'Operating Cash Flow' in cash_flow.index and date in cash_flow.columns:
                        cf = cash_flow.loc['Operating Cash Flow', date]
                        if cf is not None and not pd.isna(cf):
                            row['cash_flow'] = float(cf) * 4

                    if len(row) > 2:  # Has at least one metric beyond fund_date and symbol
                        fund_rows.append(row)
                        ticker_has_data = True

                if ticker_has_data:
                    fetched_count += 1

            except Exception as e:
                logger.error(f"Error fetching fundamental data for {ticker}: {e}")

        # Save checkpoint every 100 tickers
        if checkpoint_dir and (i + batch_size) % 100 < batch_size:
            _save_fund_checkpoint(checkpoint_dir, fund_rows)

        time_module.sleep(fund_delay)

    # Final checkpoint
    if checkpoint_dir and fund_rows:
        _save_fund_checkpoint(checkpoint_dir, fund_rows)

    logger.info(f"Fetched fundamental data for {fetched_count}/{len(tickers)} tickers")

    if not fund_rows:
        logger.error("No fundamental data fetched for any ticker")
        return None

    # Build a fundamentals DataFrame and join with prices using an asof join
    # (much faster than the old row-by-row iteration)
    logger.info("Creating daily value metrics DataFrame via asof join")

    fund_df = pl.DataFrame(fund_rows).sort(['symbol', 'fund_date'])

    # Ensure consistent date types for the asof join
    fund_df = fund_df.with_columns(pl.col('fund_date').cast(pl.Datetime('ns')))

    prices = price_df.select(['date', 'symbol', 'adj_close']).sort(['symbol', 'date'])
    prices = prices.with_columns(pl.col('date').cast(pl.Datetime('ns')))

    # Asof join: for each (symbol, trading_date), find the most recent fund_date <= trading_date
    joined = prices.join_asof(
        fund_df,
        left_on='date',
        right_on='fund_date',
        by='symbol',
        strategy='backward',
    )

    # Calculate value ratios: fundamental / price
    value_exprs = []
    if 'book_value' in joined.columns:
        value_exprs.append(
            pl.when(pl.col('adj_close') > 0)
            .then(pl.col('book_value') / pl.col('adj_close'))
            .otherwise(None)
            .alias('book_price')
        )
    if 'revenue' in joined.columns:
        value_exprs.append(
            pl.when(pl.col('adj_close') > 0)
            .then(pl.col('revenue') / pl.col('adj_close'))
            .otherwise(None)
            .alias('sales_price')
        )
    if 'cash_flow' in joined.columns:
        value_exprs.append(
            pl.when(pl.col('adj_close') > 0)
            .then(pl.col('cash_flow') / pl.col('adj_close'))
            .otherwise(None)
            .alias('cf_price')
        )

    if not value_exprs:
        logger.error("No value metrics could be computed")
        return None

    value_df = joined.select(['date', 'symbol'] + value_exprs)

    # Drop rows where all value columns are null (no fundamental data available yet)
    value_cols = [c for c in ['book_price', 'sales_price', 'cf_price'] if c in value_df.columns]
    value_df = value_df.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in value_cols])
    )

    # Forward-fill remaining nulls within each symbol
    value_df = (
        value_df
        .sort(['symbol', 'date'])
        .with_columns([pl.col(c).forward_fill().over('symbol') for c in value_cols])
        .drop_nulls()
    )

    logger.info(f"Value metrics DataFrame created. Shape: {value_df.shape}")
    return value_df

def _save_fund_checkpoint(checkpoint_dir, fund_rows):
    """Save fundamental data progress to a checkpoint file."""
    if not fund_rows:
        return
    os.makedirs(checkpoint_dir, exist_ok=True)
    pl.DataFrame(fund_rows).write_parquet(
        os.path.join(checkpoint_dir, "fund_data_checkpoint.parquet")
    )


def _to_date_str(date_val) -> str:
    """Convert a date value (datetime, pd.Timestamp, etc.) to 'YYYY-MM-DD' string."""
    if hasattr(date_val, 'strftime'):
        return date_val.strftime("%Y-%m-%d")
    return str(date_val)


def _filter_after_date(df, date_str):
    """Filter a Polars DataFrame to rows on or after the given date string."""
    return df.with_columns(
        pl.col("date").dt.strftime("%Y-%m-%d").alias("_date_str")
    ).filter(
        pl.col("_date_str") >= date_str
    ).drop("_date_str")


def align_and_clean_data(returns_df, mkt_cap_df, sector_df, value_df, start_date=START_DATE):
    """
    Aligns all DataFrames to the same set of symbols and filters to start_date.

    Date alignment is intentionally lenient: returns, market_cap, and sector data
    keep their full history from start_date onward, while value_df may start later
    (yfinance only provides ~2 years of quarterly fundamentals). The factor model
    handles the mismatch — value scores are simply 0 for dates without fundamentals.
    """
    dfs = [returns_df, mkt_cap_df, sector_df, value_df]
    if any(df is None or df.is_empty() for df in dfs):
        logger.error("Cannot align data: One or more DataFrames are empty")
        return None, None, None, None

    logger.info("Aligning and cleaning all DataFrames")

    # Filter each DataFrame to start_date (not to the latest common start)
    returns_df = _filter_after_date(returns_df, start_date)
    mkt_cap_df = _filter_after_date(mkt_cap_df, start_date)
    sector_df = _filter_after_date(sector_df, start_date)
    value_df = _filter_after_date(value_df, start_date)

    # Log per-DataFrame date ranges for transparency
    for name, df in [('returns', returns_df), ('mkt_cap', mkt_cap_df),
                     ('sector', sector_df), ('value', value_df)]:
        logger.info(f"  {name}: {_to_date_str(df['date'].min())} → {_to_date_str(df['date'].max())} ({df.height:,} rows)")

    # Find symbols common to all DataFrames
    common_symbols = set(returns_df['symbol'].unique().to_list())
    for df in [mkt_cap_df, sector_df, value_df]:
        common_symbols &= set(df['symbol'].unique().to_list())

    logger.info(f"Found {len(common_symbols)} symbols common to all datasets")

    if len(common_symbols) == 0:
        logger.error("No common symbols across all DataFrames")
        return None, None, None, None

    # Filter to common symbols
    sym_list = list(common_symbols)
    returns_df = returns_df.filter(pl.col('symbol').is_in(sym_list))
    mkt_cap_df = mkt_cap_df.filter(pl.col('symbol').is_in(sym_list))
    sector_df = sector_df.filter(pl.col('symbol').is_in(sym_list))
    value_df = value_df.filter(pl.col('symbol').is_in(sym_list))

    logger.info(f"Data alignment complete. Shapes - returns: {returns_df.shape}, market_cap: {mkt_cap_df.shape}, sector: {sector_df.shape}, value: {value_df.shape}")

    return returns_df, mkt_cap_df, sector_df, value_df
    
def save_data_to_files(returns_df, mkt_cap_df, sector_df, value_df,
                       output_dir=None, universe=UNIVERSE_SP500):
    """Saves all DataFrames to parquet files in a universe-specific directory."""
    if any(df is None or df.is_empty() for df in [returns_df, mkt_cap_df, sector_df, value_df]):
        logger.error("Cannot save data: One or more DataFrames are empty")
        return None

    if output_dir is None:
        output_dir = _get_output_dir(universe)

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    returns_df.write_parquet(f"{output_dir}/returns_df.parquet")
    mkt_cap_df.write_parquet(f"{output_dir}/mkt_cap_df.parquet")
    sector_df.write_parquet(f"{output_dir}/sector_df.parquet")
    value_df.write_parquet(f"{output_dir}/value_df.parquet")

    logger.info(f"Data saved to {output_dir}/")
    return {
        "returns_df": f"{output_dir}/returns_df.parquet",
        "mkt_cap_df": f"{output_dir}/mkt_cap_df.parquet",
        "sector_df": f"{output_dir}/sector_df.parquet",
        "value_df": f"{output_dir}/value_df.parquet"
    }

def load_saved_data(input_dir=None, universe=UNIVERSE_SP500):
    """
    Loads previously saved data files for use with the factor model.

    Parameters
    ----------
    input_dir : str or None
        Directory containing the parquet files. If None, uses the
        universe-specific default ``./data/{universe}/``.
    universe : str
        Universe name (used to derive *input_dir* when not given explicitly).

    Returns
    -------
    Tuple of DataFrames: (returns_df, mkt_cap_df, sector_df, value_df)
    """
    if input_dir is None:
        input_dir = _get_output_dir(universe)

    # Backward compat: check if legacy flat layout exists (./data/*.parquet)
    legacy_dir = OUTPUT_DIR
    if not os.path.exists(input_dir) and os.path.exists(f"{legacy_dir}/returns_df.parquet"):
        logger.warning(
            f"Universe directory {input_dir} not found, but legacy data files "
            f"exist in {legacy_dir}/. Consider moving them to {input_dir}/."
        )
        input_dir = legacy_dir

    try:
        logger.info(f"Loading data from {input_dir}")

        if not os.path.exists(input_dir):
            logger.error(f"Directory {input_dir} does not exist")
            return None, None, None, None

        files = {
            'returns_df': f"{input_dir}/returns_df.parquet",
            'mkt_cap_df': f"{input_dir}/mkt_cap_df.parquet",
            'sector_df': f"{input_dir}/sector_df.parquet",
            'value_df': f"{input_dir}/value_df.parquet"
        }

        missing_files = [n for n, p in files.items() if not os.path.exists(p)]
        if missing_files:
            logger.error(f"Missing data files: {', '.join(missing_files)}")
            return None, None, None, None

        returns_df = pl.read_parquet(files['returns_df'])
        mkt_cap_df = pl.read_parquet(files['mkt_cap_df'])
        sector_df = pl.read_parquet(files['sector_df'])
        value_df = pl.read_parquet(files['value_df'])

        logger.info(
            f"Data loaded successfully. Shapes - returns: {returns_df.shape}, "
            f"market_cap: {mkt_cap_df.shape}, sector: {sector_df.shape}, "
            f"value: {value_df.shape}"
        )
        return returns_df, mkt_cap_df, sector_df, value_df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def run_loader_and_model(force_refresh=None, universe=None):
    """
    Load data (from cache or by fetching fresh), returning the four DataFrames.

    Parameters
    ----------
    force_refresh : bool or None
        Re-download from Yahoo Finance even when cached parquet files exist.
        Defaults to the module-level FORCE_REFRESH setting.
    universe : str or None
        ``"sp500"`` or ``"russell3000"``. Defaults to the module-level UNIVERSE.
    """
    if force_refresh is None:
        force_refresh = FORCE_REFRESH
    if universe is None:
        universe = UNIVERSE

    data_dir = _get_output_dir(universe)
    parquet_files = [f"{data_dir}/{f}.parquet" for f in
                     ("returns_df", "mkt_cap_df", "sector_df", "value_df")]

    if not force_refresh and all(os.path.exists(f) for f in parquet_files):
        logger.info(f"Found existing {universe} data files, loading them...")
        result = load_saved_data(universe=universe)
        if result[0] is not None:
            logger.info("Successfully loaded existing data")
            return result

    logger.info(f"Fetching new data for universe={universe}...")
    main(universe=universe)
    return load_saved_data(universe=universe)

def main(universe=None):
    """Main entry point for the data collection pipeline.

    Parameters
    ----------
    universe : str or None
        ``"sp500"`` or ``"russell3000"``. Defaults to the module-level UNIVERSE.
    """
    if universe is None:
        universe = UNIVERSE

    try:
        logger.info(f"Starting data collection for {universe} from {START_DATE} to {END_DATE}")

        # 1. Get tickers for the requested universe
        if TEST_MODE:
            tickers = TEST_TICKERS
            logger.info(f"TEST MODE: Using {len(tickers)} test tickers")
        else:
            tickers = get_tickers(universe)
        if not tickers:
            logger.error("No tickers found, exiting.")
            return

        logger.info(f"Universe {universe}: {len(tickers)} tickers")

        # Rate-limit parameters tuned to universe size
        rate = _get_rate_params(len(tickers))
        checkpoint_dir = os.path.join(_get_output_dir(universe), ".checkpoints")

        # 2. Fetch price data and calculate returns
        price_df = fetch_price_data(tickers, batch_size=rate["price_batch"])
        if price_df is None or price_df.is_empty():
            logger.error("Failed to fetch price data, exiting.")
            return

        returns_df = create_returns_df(price_df)
        if returns_df is None or returns_df.is_empty():
            logger.error("Failed to calculate returns, exiting.")
            return

        # 3. Fetch ticker info (shares + sector) in a single pass
        shares_data, sector_list = fetch_ticker_info(
            tickers, batch_size=rate["info_batch"], checkpoint_dir=checkpoint_dir
        )

        # 4. Compute market cap from shares outstanding + prices
        mkt_cap_df = fetch_market_cap_data(tickers, price_df, shares_data=shares_data)
        if mkt_cap_df is None or mkt_cap_df.is_empty():
            logger.warning("Failed to fetch market cap data, creating fallback data...")
            np.random.seed(42)
            mkt_cap_df = price_df.select(['date', 'symbol', 'adj_close']) \
                .with_columns([
                    (pl.col('adj_close') * pl.col('symbol').map_elements(
                        lambda s: np.random.lognormal(mean=np.log(1e8), sigma=0.8),
                        return_dtype=pl.Float64
                    )).alias('market_cap')
                ]) \
                .select(['date', 'symbol', 'market_cap'])

        # 5. Create sector exposure DataFrame
        sector_info_df = fetch_sector_data(tickers, sector_data=sector_list)
        if sector_info_df is None or sector_info_df.is_empty():
            logger.warning("Failed to fetch sector data, creating fallback sectors...")
            sectors = ['Technology', 'Financials', 'Healthcare', 'Energy', 'Consumer']
            sector_data = []
            for ticker in tickers:
                sector_data.append({
                    'symbol': ticker,
                    'sector': sectors[hash(ticker) % len(sectors)]
                })
            sector_info_df = pl.DataFrame(sector_data)

        sector_df = create_sector_exposure_df(sector_info_df, price_df)
        if sector_df is None:
            logger.error("Failed to create sector exposure data, exiting.")
            return

        # 6. Fetch value metrics (fundamentals) data
        value_df = fetch_value_metrics_data(
            tickers, price_df, batch_size=rate["fund_batch"],
            checkpoint_dir=checkpoint_dir,
        )
        if value_df is None or value_df.is_empty():
            logger.error("Failed to create value metrics, exiting.")
            return

        # 7. Align and clean all DataFrames
        aligned_dfs = align_and_clean_data(returns_df, mkt_cap_df, sector_df, value_df)
        if aligned_dfs[0] is None:
            logger.error("Data alignment failed, saving unaligned data...")
            save_data_to_files(returns_df, mkt_cap_df, sector_df, value_df, universe=universe)
        else:
            returns_df, mkt_cap_df, sector_df, value_df = aligned_dfs
            save_data_to_files(returns_df, mkt_cap_df, sector_df, value_df, universe=universe)

        # Clean up checkpoints on success
        if os.path.exists(checkpoint_dir):
            import shutil
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            logger.info("Cleaned up checkpoint files")

        logger.info(f"{universe} data collection complete")

    except Exception as e:
        logger.error(f"Error in data collection process: {str(e)}")
        logger.exception("Stack trace:")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch equity data for factor model")
    parser.add_argument(
        "--universe", default=UNIVERSE_SP500,
        choices=[UNIVERSE_SP500, UNIVERSE_RUSSELL3000],
        help="Ticker universe to fetch (default: sp500)",
    )
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-fetch even if cached data exists")
    parser.add_argument("--test", action="store_true",
                        help="Use a small set of test tickers for quick iteration")
    args = parser.parse_args()
    if args.test:
        TEST_MODE = True
    if args.force_refresh:
        FORCE_REFRESH = True
    main(universe=args.universe)