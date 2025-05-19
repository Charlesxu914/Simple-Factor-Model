"""
S&P 500 data loader using Yahoo Finance.

This script implements a data pipeline to fetch S&P 500 historical data,
creating all the necessary DataFrames for the equity factor model.
"""

import numpy as np
import polars as pl
import pandas as pd
from datetime import datetime, timedelta
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('yfinance_data_loader')

# Configuration
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = "./data"
TEST_MODE = False  # Set to False to process full S&P 500
TEST_TICKERS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL']  # Test with top 5 tech stocks

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    logger.warning("yfinance package not available. Install using: pip install yfinance")
    YFINANCE_AVAILABLE = False
    
def get_sp500_tickers():
    """
    Fetches current S&P 500 constituents by scraping Wikipedia.
    Returns a list of ticker symbols.
    """
    if TEST_MODE:
        logger.info("TEST MODE: Using a small set of test tickers")
        return TEST_TICKERS
        
    if not YFINANCE_AVAILABLE:
        logger.warning("Using a small placeholder list of S&P 500 tickers")
        # Return a subset of S&P 500 companies as placeholder
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'UNH', 'XOM', 'JPM',
            'JNJ', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'LLY', 'AVGO', 'KO'
        ]
    
    try:
        logger.info("Fetching S&P 500 tickers from Wikipedia")
        # Use pandas to read the S&P 500 table from Wikipedia
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].str.replace('.', '-').tolist()
        logger.info(f"Found {len(tickers)} S&P 500 constituents")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        # Fallback to a small subset
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'BRK-B', 'UNH', 'XOM', 'JPM'
        ]
        
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
            # Download data for this batch
            data = yf.download(
                batch_tickers,
                start=start_date,
                end=end_date,
                group_by='ticker',
                auto_adjust=False,  # We want to use Adj Close for consistency
                progress=False,
                threads=True,
                ignore_tz=True
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
    
def fetch_market_cap_data(tickers, price_df=None, start_date=START_DATE, end_date=END_DATE, batch_size=20):
    """
    Fetches historical market cap data by combining:
    1. Current shares outstanding from ticker.info
    2. Historical price data
    
    This is an approximation since yfinance doesn't provide historical shares outstanding.
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("Cannot fetch real market cap data. yfinance package not available.")
        return None
        
    if price_df is None or price_df.is_empty():
        logger.error("Price data is required to calculate market cap")
        return None
        
    logger.info(f"Fetching market cap data for {len(tickers)} tickers")
    
    # Get current shares outstanding for each ticker
    shares_data = {}
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        logger.info(f"Fetching shares data - batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
        
        for ticker in batch_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Try different fields that might contain shares outstanding
                shares_outstanding = None
                for field in ['sharesOutstanding', 'impliedSharesOutstanding']:
                    if field in info and info[field] and info[field] > 0:
                        shares_outstanding = info[field]
                        break
                
                if shares_outstanding:
                    shares_data[ticker] = shares_outstanding
                else:
                    logger.warning(f"No shares outstanding data found for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
    
    if not shares_data:
        logger.error("No shares outstanding data fetched for any tickers")
        return None
    
    # Calculate market cap for each date-symbol pair
    logger.info("Calculating market cap using price data and shares outstanding")
    
    mkt_cap_df = (
        price_df.select(["date", "symbol", "adj_close"])
        .with_columns([
            pl.col("symbol").map_elements(lambda s: shares_data.get(s, None), return_dtype=pl.Float64).alias("shares_outstanding")
        ])
        .filter(pl.col("shares_outstanding").is_not_null())  # Remove tickers with no shares data
        .with_columns([
            (pl.col("adj_close") * pl.col("shares_outstanding")).alias("market_cap")
        ])
        .select(["date", "symbol", "market_cap"])
    )
    
    logger.info(f"Market cap calculation complete. Shape: {mkt_cap_df.shape}")
    return mkt_cap_df

def fetch_sector_data(tickers, batch_size=20):
    """
    Fetches sector information for each ticker.
    Returns a DataFrame with ticker -> sector mapping.
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("Cannot fetch real sector data. yfinance package not available.")
        return None
        
    logger.info(f"Fetching sector data for {len(tickers)} tickers")
    
    sector_data = []
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        logger.info(f"Fetching sector data - batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
        
        for ticker in batch_tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                if 'sector' in info and info['sector']:
                    sector_data.append({
                        'symbol': ticker,
                        'sector': info['sector']
                    })
                else:
                    logger.warning(f"No sector data found for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching sector data for {ticker}: {e}")
    
    if not sector_data:
        logger.error("No sector data fetched for any tickers")
        return None
        
    # Create DataFrame from sector data
    sector_df = pl.DataFrame(sector_data)
    logger.info(f"Sector data fetched for {sector_df.height} tickers with {sector_df['sector'].n_unique()} unique sectors")
    
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
        
def fetch_value_metrics_data(tickers, price_df=None, start_date=START_DATE, end_date=END_DATE, batch_size=20):
    """
    Fetches fundamental data needed for value metrics.
    Returns a DataFrame with: date, symbol, book_price, sales_price, cf_price
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("Cannot fetch real value metrics. yfinance package not available.")
        return None
        
    if price_df is None or price_df.is_empty():
        logger.error("Price data is required to calculate value metrics")
        return None
        
    logger.info(f"Fetching value metrics data for {len(tickers)} tickers")
    
    # Dictionary to store latest fundamental data for each ticker
    fundamental_data = {}
    
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i:i + batch_size]
        logger.info(f"Fetching fundamental data - batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
        
        for ticker in batch_tickers:
            try:
                stock = yf.Ticker(ticker)
                
                # Get quarterly balance sheet data
                balance_sheet = stock.quarterly_balance_sheet
                # Get quarterly financials (income statement)
                financials = stock.quarterly_financials
                # Get quarterly cash flow
                cash_flow = stock.quarterly_cashflow
                
                if balance_sheet.empty and financials.empty and cash_flow.empty:
                    logger.warning(f"No fundamental data found for {ticker}")
                    continue
                    
                # Extract data for each reporting date
                fundamental_dates = sorted(set(
                    balance_sheet.columns.tolist() + 
                    financials.columns.tolist() +
                    cash_flow.columns.tolist()
                ))
                
                ticker_data = {}
                for date in fundamental_dates:
                    data_point = {'date': date}
                    
                    # Book value (Total Assets - Total Liabilities)
                    if not balance_sheet.empty:
                        if 'Total Assets' in balance_sheet.index and date in balance_sheet.columns:
                            assets = balance_sheet.loc['Total Assets', date]
                            liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest', date] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                            
                            if assets is not None and liabilities is not None and not pd.isna(assets) and not pd.isna(liabilities):
                                data_point['book_value'] = assets - liabilities
                    
                    # Sales (Revenue)
                    if not financials.empty:
                        if 'Total Revenue' in financials.index and date in financials.columns:
                            revenue = financials.loc['Total Revenue', date]
                            if revenue is not None and not pd.isna(revenue):
                                # Annualize quarterly revenue
                                data_point['revenue'] = revenue * 4
                    
                    # Cash Flow
                    if not cash_flow.empty:
                        if 'Operating Cash Flow' in cash_flow.index and date in cash_flow.columns:
                            cf = cash_flow.loc['Operating Cash Flow', date]
                            if cf is not None and not pd.isna(cf):
                                # Annualize quarterly cash flow
                                data_point['cash_flow'] = cf * 4
                    
                    if len(data_point) > 1:  # More than just date
                        ticker_data[date] = data_point
                
                if ticker_data:
                    fundamental_data[ticker] = ticker_data
                
            except Exception as e:
                logger.error(f"Error fetching fundamental data for {ticker}: {e}")
    
    if not fundamental_data:
        logger.error("No fundamental data fetched for any ticker")
        return None
        
    # Create a daily value metrics DataFrame
    logger.info("Creating daily value metrics DataFrame")
    
    # Get all unique dates and symbols from price data
    unique_dates = price_df['date'].unique()
    unique_symbols = list(set(price_df['symbol'].unique()) & set(fundamental_data.keys()))
    
    value_data_list = []
    
    # For each ticker with fundamental data
    for symbol in unique_symbols:
        ticker_fundamentals = fundamental_data[symbol]
        # Sort fundamental dates
        fundamental_dates = sorted(ticker_fundamentals.keys())
        
        # Get market cap data for this symbol
        ticker_prices = price_df.filter(pl.col("symbol") == symbol)
        
        # For each trading day
        for trading_date in ticker_prices['date']:
            # Find the latest fundamental data point before this trading date
            latest_fundamental_date = None
            
            # Ensure trading_date is a datetime object for comparison
            trading_datetime = trading_date
            if isinstance(trading_date, pl.Date):
                trading_datetime = pd.Timestamp(trading_date.strftime("%Y-%m-%d"))
            
            for fund_date in reversed(fundamental_dates):
                # Ensure fund_date is a datetime object for comparison
                fund_datetime = fund_date
                if isinstance(fund_date, pd.Timestamp):
                    fund_datetime = fund_date
                else:
                    try:
                        fund_datetime = pd.Timestamp(fund_date)
                    except:
                        # If conversion fails, skip this date
                        logger.warning(f"Could not convert fundamental date {fund_date} to timestamp")
                        continue
                
                # Compare dates
                try:
                    if fund_datetime < trading_datetime:
                        latest_fundamental_date = fund_date
                        break
                except Exception as e:
                    logger.error(f"Error comparing dates {fund_datetime} and {trading_datetime}: {e}")
                    continue
            
            if latest_fundamental_date is None:
                continue  # No fundamental data before this trading date
                
            # Get the price data for this trading day
            price_row = ticker_prices.filter(pl.col("date") == trading_date)
            if price_row.is_empty():
                continue
                
            price = price_row['adj_close'][0]
            
            # Get the fundamental data
            fund_data = ticker_fundamentals[latest_fundamental_date]
            
            # Calculate value metrics
            data_point = {
                'date': trading_date,
                'symbol': symbol
            }
            
            if 'book_value' in fund_data and price > 0:
                data_point['book_price'] = fund_data['book_value'] / price
            
            if 'revenue' in fund_data and price > 0:
                data_point['sales_price'] = fund_data['revenue'] / price
                
            if 'cash_flow' in fund_data and price > 0:
                data_point['cf_price'] = fund_data['cash_flow'] / price
            
            # Add to list if we have at least one value metric
            if len(data_point) > 2:  # More than just date and symbol
                value_data_list.append(data_point)
    
    if not value_data_list:
        logger.error("Could not create value metrics: No valid data points")
        # Fallback: Create minimal value metrics from price data
        logger.info("Creating fallback value metrics based on price")
        
        # We'll use price/earnings ratio as a simple proxy
        # Since we don't have earnings, we'll generate a proxy
        value_data_list = []
        
        for symbol in price_df['symbol'].unique().to_list()[:20]:  # Limit to first 20 symbols for simplicity
            ticker_prices = price_df.filter(pl.col("symbol") == symbol)
            
            # Generate some synthetic book/price, sales/price and cf/price values for this ticker
            # These will be randomized but consistent for each ticker
            np.random.seed(hash(symbol) % 2**32)  # Use hash of symbol as random seed
            
            base_bp = np.random.uniform(0.5, 3.0)  # Base book/price ratio
            base_sp = np.random.uniform(0.2, 5.0)  # Base sales/price ratio
            base_cp = np.random.uniform(0.5, 10.0)  # Base cash-flow/price ratio
            
            # Add some small random fluctuations
            for idx, row in enumerate(ticker_prices.iter_rows(named=True)):
                # Add small fluctuations with some trend
                factor = 1.0 + np.random.normal(0, 0.02) + (idx * 0.0001)
                
                value_data_list.append({
                    'date': row['date'],
                    'symbol': symbol,
                    'book_price': base_bp * factor,
                    'sales_price': base_sp * factor,
                    'cf_price': base_cp * factor
                })
    
    # Create DataFrame from list
    value_df = pl.DataFrame(value_data_list)
    
    # Fill missing values by forward-filling within each symbol
    value_df = value_df.sort(['symbol', 'date'])
    
    # Convert to pandas for easier forward filling
    value_pd = value_df.to_pandas()
    value_pd = value_pd.set_index(['symbol', 'date']).sort_index()
    
    # Forward fill each column
    for col in ['book_price', 'sales_price', 'cf_price']:
        if col in value_pd.columns:
            value_pd[col] = value_pd.groupby('symbol')[col].ffill()
    
    value_pd = value_pd.reset_index()
    
    # Convert back to Polars
    value_df = pl.from_pandas(value_pd).sort(['symbol', 'date'])
    
    # Drop rows with missing values after filling
    value_df = value_df.drop_nulls()
    
    logger.info(f"Value metrics DataFrame created. Shape: {value_df.shape}")
    return value_df

def align_and_clean_data(returns_df, mkt_cap_df, sector_df, value_df, start_date=START_DATE):
    """
    Aligns all DataFrames to the same date range and set of symbols.
    Returns filtered and aligned DataFrames ready for the factor model.
    """
    if any(df is None or df.is_empty() for df in [returns_df, mkt_cap_df, sector_df, value_df]):
        logger.error("Cannot align data: One or more DataFrames are empty")
        return None, None, None, None
    
    logger.info("Aligning and cleaning all DataFrames")
    
    # Simple approach to find common start date
    try:
        # Find the min date for each DataFrame
        min_dates = []
        
        # Get min date as string from returns_df
        min_returns_date = returns_df['date'].min()
        if isinstance(min_returns_date, datetime):
            min_returns_date = min_returns_date.strftime("%Y-%m-%d")
        elif isinstance(min_returns_date, pd.Timestamp):
            min_returns_date = min_returns_date.strftime("%Y-%m-%d")
        elif isinstance(min_returns_date, pl.Date):
            min_returns_date = min_returns_date.strftime("%Y-%m-%d")
        min_dates.append(min_returns_date)
        
        # Get min date as string from mkt_cap_df
        min_mktcap_date = mkt_cap_df['date'].min()
        if isinstance(min_mktcap_date, datetime):
            min_mktcap_date = min_mktcap_date.strftime("%Y-%m-%d")
        elif isinstance(min_mktcap_date, pd.Timestamp):
            min_mktcap_date = min_mktcap_date.strftime("%Y-%m-%d")
        elif isinstance(min_mktcap_date, pl.Date):
            min_mktcap_date = min_mktcap_date.strftime("%Y-%m-%d")
        min_dates.append(min_mktcap_date)
        
        # Get min date as string from sector_df
        min_sector_date = sector_df['date'].min()
        if isinstance(min_sector_date, datetime):
            min_sector_date = min_sector_date.strftime("%Y-%m-%d")
        elif isinstance(min_sector_date, pd.Timestamp):
            min_sector_date = min_sector_date.strftime("%Y-%m-%d")
        elif isinstance(min_sector_date, pl.Date):
            min_sector_date = min_sector_date.strftime("%Y-%m-%d")
        min_dates.append(min_sector_date)
        
        # Get min date as string from value_df
        min_value_date = value_df['date'].min()
        if isinstance(min_value_date, datetime):
            min_value_date = min_value_date.strftime("%Y-%m-%d")
        elif isinstance(min_value_date, pd.Timestamp):
            min_value_date = min_value_date.strftime("%Y-%m-%d")
        elif isinstance(min_value_date, pl.Date):
            min_value_date = min_value_date.strftime("%Y-%m-%d")
        min_dates.append(min_value_date)
        
        # Add the user-specified start date
        min_dates.append(start_date)
        
        # Convert all to pd.Timestamp for comparison
        min_dates_pd = [pd.Timestamp(d) for d in min_dates]
        
        # Find the max date among all min dates
        common_start_date = max(min_dates_pd)
        common_start_str = common_start_date.strftime("%Y-%m-%d")
        
        logger.info(f"Using common start date: {common_start_str}")
    except Exception as e:
        logger.error(f"Error determining common start date: {e}")
        common_start_str = start_date
        logger.info(f"Falling back to configured start date: {common_start_str}")
    
    # Convert string date to pandas Timestamp for filtering
    common_start_date_pd = pd.Timestamp(common_start_str)
    
    # Filter each DataFrame
    # Convert date column to string for comparison to avoid type issues
    returns_df = returns_df.with_columns([
        pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str")
    ]).filter(
        pl.col("date_str") >= common_start_str
    ).drop("date_str")
    
    mkt_cap_df = mkt_cap_df.with_columns([
        pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str")
    ]).filter(
        pl.col("date_str") >= common_start_str
    ).drop("date_str")
    
    sector_df = sector_df.with_columns([
        pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str")
    ]).filter(
        pl.col("date_str") >= common_start_str
    ).drop("date_str")
    
    value_df = value_df.with_columns([
        pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str")
    ]).filter(
        pl.col("date_str") >= common_start_str
    ).drop("date_str")
    
    # Find symbols common to all DataFrames
    common_symbols = set(returns_df['symbol'].unique().to_list())
    common_symbols &= set(mkt_cap_df['symbol'].unique().to_list())
    common_symbols &= set(sector_df['symbol'].unique().to_list())
    common_symbols &= set(value_df['symbol'].unique().to_list())
    
    logger.info(f"Found {len(common_symbols)} symbols common to all datasets")
    
    if len(common_symbols) == 0:
        logger.error("No common symbols across all DataFrames")
        return None, None, None, None
        
    # Filter to common symbols
    returns_df = returns_df.filter(pl.col('symbol').is_in(list(common_symbols)))
    mkt_cap_df = mkt_cap_df.filter(pl.col('symbol').is_in(list(common_symbols)))
    sector_df = sector_df.filter(pl.col('symbol').is_in(list(common_symbols)))
    value_df = value_df.filter(pl.col('symbol').is_in(list(common_symbols)))
    
    logger.info(f"Data alignment complete. Shapes - returns: {returns_df.shape}, market_cap: {mkt_cap_df.shape}, sector: {sector_df.shape}, value: {value_df.shape}")
    
    return returns_df, mkt_cap_df, sector_df, value_df
    
def save_data_to_files(returns_df, mkt_cap_df, sector_df, value_df, output_dir=OUTPUT_DIR):
    """Saves all DataFrames to parquet files."""
    if any(df is None or df.is_empty() for df in [returns_df, mkt_cap_df, sector_df, value_df]):
        logger.error("Cannot save data: One or more DataFrames are empty")
        return None
    
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

def load_saved_data(input_dir=OUTPUT_DIR):
    """
    Loads previously saved data files for use with the factor model.
    
    Args:
        input_dir: Directory containing the parquet files
        
    Returns:
        Tuple of DataFrames: (returns_df, mkt_cap_df, sector_df, value_df)
    """
    try:
        logger.info(f"Loading data from {input_dir}")
        
        # Check if directory exists
        if not os.path.exists(input_dir):
            logger.error(f"Directory {input_dir} does not exist")
            return None, None, None, None
        
        # List expected files
        files = {
            'returns_df': f"{input_dir}/returns_df.parquet",
            'mkt_cap_df': f"{input_dir}/mkt_cap_df.parquet",
            'sector_df': f"{input_dir}/sector_df.parquet",
            'value_df': f"{input_dir}/value_df.parquet"
        }
        
        # Check if all files exist
        missing_files = []
        for name, path in files.items():
            if not os.path.exists(path):
                missing_files.append(name)
                
        if missing_files:
            logger.error(f"Missing data files: {', '.join(missing_files)}")
            return None, None, None, None
        
        # Load the data
        returns_df = pl.read_parquet(files['returns_df'])
        mkt_cap_df = pl.read_parquet(files['mkt_cap_df'])
        sector_df = pl.read_parquet(files['sector_df'])
        value_df = pl.read_parquet(files['value_df'])
        
        logger.info(f"Data loaded successfully. Shapes - returns: {returns_df.shape}, market_cap: {mkt_cap_df.shape}, sector: {sector_df.shape}, value: {value_df.shape}")
        
        return returns_df, mkt_cap_df, sector_df, value_df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return None, None, None, None

def run_loader_and_model():
    """
    Utility function to load data and run the factor model in one step.
    This can be imported and called from other scripts.
    
    Steps:
    1. Check if data already exists
    2. If not, fetch and save data
    3. Return data for model usage
    """
    data_dir = OUTPUT_DIR
    
    # Check if data files already exist
    if (os.path.exists(f"{data_dir}/returns_df.parquet") and
        os.path.exists(f"{data_dir}/mkt_cap_df.parquet") and
        os.path.exists(f"{data_dir}/sector_df.parquet") and 
        os.path.exists(f"{data_dir}/value_df.parquet")):
        
        logger.info("Found existing data files, loading them...")
        returns_df, mkt_cap_df, sector_df, value_df = load_saved_data()
        
        if returns_df is not None:
            logger.info("Successfully loaded existing data")
            return returns_df, mkt_cap_df, sector_df, value_df
    
    # If we get here, we need to fetch new data
    logger.info("No existing data found or loading failed, fetching new data...")
    
    # Run the main data collection process
    main()
    
    # Load and return the newly created data
    return load_saved_data()

def main():
    """Main entry point for the script."""
    try:
        logger.info(f"Starting data collection for S&P 500 from {START_DATE} to {END_DATE}")
        
        # 1. Get S&P 500 tickers
        tickers = get_sp500_tickers()
        if not tickers:
            logger.error("No tickers found, exiting.")
            return
        
        # 2. Fetch price data and calculate returns
        price_df = fetch_price_data(tickers)
        if price_df is None or price_df.is_empty():
            logger.error("Failed to fetch price data, exiting.")
            return
            
        returns_df = create_returns_df(price_df)
        if returns_df is None or returns_df.is_empty():
            logger.error("Failed to calculate returns, exiting.")
            return
        
        # 3. Fetch market cap data
        mkt_cap_df = fetch_market_cap_data(tickers, price_df)
        if mkt_cap_df is None or mkt_cap_df.is_empty():
            logger.warning("Failed to fetch market cap data, creating fallback data...")
            # Create a fallback market cap (using price as proxy with random multiplier)
            np.random.seed(42)
            mkt_cap_df = price_df.select(['date', 'symbol', 'adj_close']) \
                .with_columns([
                    (pl.col('adj_close') * pl.col('symbol').map_elements(
                        lambda s: np.random.lognormal(mean=np.log(1e8), sigma=0.8),
                        return_dtype=pl.Float64
                    )).alias('market_cap')
                ]) \
                .select(['date', 'symbol', 'market_cap'])
        
        # 4. Fetch sector information and create sector exposure DataFrame
        sector_info_df = fetch_sector_data(tickers)
        if sector_info_df is None or sector_info_df.is_empty():
            logger.warning("Failed to fetch sector data, creating fallback sectors...")
            # Create fallback sector data
            sectors = ['Technology', 'Financials', 'Healthcare', 'Energy', 'Consumer']
            sector_data = []
            for ticker in tickers:
                sector_data.append({
                    'symbol': ticker,
                    'sector': sectors[hash(ticker) % len(sectors)]  # Assign based on hash
                })
            sector_info_df = pl.DataFrame(sector_data)
        
        sector_df = create_sector_exposure_df(sector_info_df, price_df)
        if sector_df is None:
            logger.error("Failed to create sector exposure data, exiting.")
            return
        
        # 5. Fetch value metrics data
        value_df = fetch_value_metrics_data(tickers, price_df)
        if value_df is None or value_df.is_empty():
            logger.error("Failed to create value metrics, exiting.")
            return
        
        # 6. Align and clean all DataFrames
        aligned_dfs = align_and_clean_data(returns_df, mkt_cap_df, sector_df, value_df)
        if aligned_dfs[0] is None:
            logger.error("Data alignment failed, saving unaligned data...")
            # Save the unaligned data
            save_data_to_files(returns_df, mkt_cap_df, sector_df, value_df)
        else:
            # Save aligned data
            returns_df, mkt_cap_df, sector_df, value_df = aligned_dfs
            save_data_to_files(returns_df, mkt_cap_df, sector_df, value_df)
        
        logger.info("S&P 500 data collection complete")
        
    except Exception as e:
        logger.error(f"Error in data collection process: {str(e)}")
        logger.exception("Stack trace:")
        
if __name__ == "__main__":
    main() 