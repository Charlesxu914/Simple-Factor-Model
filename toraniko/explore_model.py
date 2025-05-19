"""
Exploring the Equity Factor Model

This script demonstrates how to use the equity factor model implementation, including data preparation,
factor calculations, and model estimation. The comments are structured to be easily converted into
Jupyter notebook cells.
"""

# %% [markdown]
# # Exploring the Equity Factor Model
# 
# This notebook demonstrates how to use the equity factor model implementation, including data preparation,
# factor calculations, and model estimation.

# %% [markdown]
# ## Setup and Imports
# 
# First, let's import the necessary libraries and modules:

# %%
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_market_calendars as mcal
from datetime import datetime
import pandas as pd

from toraniko import factor_math, model, styles

# Set plotting style
plt.style.use('seaborn-v0_8')  # Use the correct seaborn style name
sns.set_theme()  # This will set the seaborn default theme

# %% [markdown]
# ## Data Requirements
# 
# The model requires several input DataFrames with specific columns:
# 
# 1. `returns_df`: Contains asset returns
#    - Required columns: `date`, `symbol`, `asset_returns`
# 
# 2. `mkt_cap_df`: Contains market capitalizations
#    - Required columns: `date`, `symbol`, `market_cap`
# 
# 3. `sector_df`: Contains sector exposures
#    - Required columns: `date`, `symbol`, plus one column for each sector
# 
# 4. `value_df`: Contains value metrics
#    - Required columns: `date`, `symbol`, `book_price`, `sales_price`, `cf_price`
# 
# Let's create some sample data to demonstrate the functionality:

# %%
# Get trading days from 2020 to today
nyse = mcal.get_calendar('NYSE')
start_date = '2020-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')
trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

# Convert trading days to polars datetime
dates_df = pl.DataFrame({
    'date': trading_days
}).with_columns(
    pl.col('date').cast(pl.Datetime)
)
dates = dates_df['date']

# Create sample symbols (100 stocks)
symbols = [f'STOCK_{i}' for i in range(1, 101)]
num_symbols = len(symbols)
num_dates = len(dates)

# --- More Realistic Market Cap Generation ---
base_mkt_caps = np.random.lognormal(mean=np.log(1e9), sigma=1.5, size=num_symbols) # Base caps from ~1M to ~100B
mkt_cap_time_series = []
for base_cap in base_mkt_caps:
    daily_returns = np.random.normal(loc=0.0005, scale=0.02, size=num_dates) # Small daily drift + volatility
    cap_series = base_cap * np.cumprod(1 + daily_returns)
    mkt_cap_time_series.append(cap_series)

mkt_cap_data_list = []
for i, symbol in enumerate(symbols):
    for j, date_val in enumerate(dates):
        mkt_cap_data_list.append({
            'date': date_val,
            'symbol': symbol,
            'market_cap': mkt_cap_time_series[i][j]
        })
mkt_cap_df = pl.DataFrame(mkt_cap_data_list)

# --- More Realistic Value Metrics Generation ---
value_metrics_data_list = []
base_book_price = np.random.uniform(0.5, 3, size=num_symbols)
base_sales_price = np.random.uniform(0.2, 5, size=num_symbols)
base_cf_price = np.random.uniform(5, 20, size=num_symbols)

for i, symbol in enumerate(symbols):
    # Simulate some slow-moving changes for value metrics
    bp_series = base_book_price[i] * (1 + np.random.normal(0, 0.01, size=num_dates).cumsum() * 0.05)
    sp_series = base_sales_price[i] * (1 + np.random.normal(0, 0.01, size=num_dates).cumsum() * 0.05)
    cp_series = base_cf_price[i] * (1 + np.random.normal(0, 0.01, size=num_dates).cumsum() * 0.05)
    for j, date_val in enumerate(dates):
        value_metrics_data_list.append({
            'date': date_val,
            'symbol': symbol,
            'book_price': max(0.01, bp_series[j]), # Ensure positive
            'sales_price': max(0.01, sp_series[j]),
            'cf_price': max(0.01, cp_series[j])
        })
value_df = pl.DataFrame(value_metrics_data_list)

# Create sample returns data (can keep this simpler for now)
returns_data = {
    'date': np.repeat(dates, len(symbols)),
    'symbol': np.tile(symbols, len(dates)),
    'asset_returns': np.random.normal(0.0001, 0.02, len(dates) * len(symbols))
}
returns_df = pl.DataFrame(returns_data)

# Create sample sector data (remains relatively static)
sectors = ['TECH', 'FIN', 'HEALTH']
sector_assignments = np.random.choice(sectors, size=num_symbols)
sector_data_list = []
for i, symbol in enumerate(symbols):
    for date_val in dates:
        row = {'date': date_val, 'symbol': symbol}
        for sector in sectors:
            row[sector] = 1 if sector_assignments[i] == sector else 0
        sector_data_list.append(row)
sector_df = pl.DataFrame(sector_data_list)

print(f"Date range for initial data generation: {dates[0]} to {dates[-1]}")
print(f"Number of trading days: {len(dates)}")
print("\nReturns DataFrame:")
print(returns_df.head())
print("\nMarket Cap DataFrame:")
print(mkt_cap_df.head())
print("\nSector DataFrame:")
print(sector_df.head())
print("\nValue Metrics DataFrame:")
print(value_df.head())

# %% [markdown]
# ## Style Factor Construction
# 
# We'll construct the style factors using the proper implementations from styles.py:
# 
# 1. Momentum (MOM): Uses exponentially weighted returns over trailing period
# 2. Size (SZE): Uses log market cap, centered and standardized
# 3. Value (VAL): Combines multiple value metrics (book/price, sales/price, cf/price)

# %%
# Construct momentum factor
mom_df = styles.factor_mom(
    returns_df=returns_df,
    trailing_days=504,  # ~2 years of trading days
    half_life=126,     # ~6 months decay
    lag=20,            # 1 month lag
    winsor_factor=0.01
).collect()

# Construct size factor
sze_df = styles.factor_sze(
    mkt_cap_df=mkt_cap_df,
    lower_decile=0.2,  # Exclude smallest 20%
    upper_decile=0.8   # Exclude largest 20%
).collect()

# Construct value factor
val_df = styles.factor_val(
    value_df=value_df,
    winsorize_features=0.01  # Winsorize value metrics at 1%
).collect()

# Combine style factors
style_df = (
    mom_df.join(sze_df, on=['date', 'symbol'])
    .join(val_df, on=['date', 'symbol'])
    .rename({
        'mom_score': 'MOMENTUM',
        'sze_score': 'SIZE',
        'val_score': 'VALUE'
    })
)

print("Style factors after construction:")
print(style_df.head())

# %% [markdown]
# ## Data Alignment & Cleaning Post Factor Construction
#
# After constructing style factors, especially momentum, we'll have NaNs at the beginning.
# We need to:
# 1. Drop rows with NaNs from `style_df`.
# 2. Find the minimum common start date across all DataFrames.
# 3. Filter all DataFrames to this common start date.

# %%
# 1. Drop NaNs from style_df (mostly from momentum calculation)
style_df = style_df.drop_nulls()

# 2. Find the minimum common start date
min_style_date = style_df['date'].min()
min_returns_date = returns_df['date'].min()
min_mkt_cap_date = mkt_cap_df['date'].min()
min_sector_date = sector_df['date'].min()

common_start_date = max(min_style_date, min_returns_date, min_mkt_cap_date, min_sector_date)

print(f"Common start date after aligning for momentum: {common_start_date}")

# 3. Filter all DataFrames to the common start date
style_df = style_df.filter(pl.col('date') >= common_start_date)
returns_df = returns_df.filter(pl.col('date') >= common_start_date)
mkt_cap_df = mkt_cap_df.filter(pl.col('date') >= common_start_date)
sector_df = sector_df.filter(pl.col('date') >= common_start_date)

print("Shape of style_df after alignment: ", style_df.shape)
print("Shape of returns_df after alignment: ", returns_df.shape)
print("Shape of mkt_cap_df after alignment: ", mkt_cap_df.shape)
print("Shape of sector_df after alignment: ", sector_df.shape)

# %% [markdown]
# ## Model Estimation
# 
# Now let's estimate the factor model using our properly constructed factors. The model will use:
# - Market factor (implicit in sector decomposition)
# - Sector factors (TECH, FIN, HEALTH)
# - Style factors (MOMENTUM, SIZE, VALUE)

# %%
# Estimate factor returns
factor_returns, residual_returns = model.estimate_factor_returns(
    returns_df=returns_df,
    mkt_cap_df=mkt_cap_df,
    sector_df=sector_df,
    style_df=style_df,
    winsor_factor=None,  # Factors are already properly processed
    residualize_styles=True
)

print("Factor Returns DataFrame:")
print(factor_returns.head())
print("\nResidual Returns DataFrame:")
print(residual_returns.head())

# %% [markdown]
# ## Analysis and Visualization
# 
# Let's analyze and visualize the factor returns:

# %%
# Calculate cumulative returns for each factor (geometric product)
factor_return_cols = [col for col in factor_returns.columns if col != 'date']

# Expressions for geometric cumulative returns
cumulative_returns_exprs = []
for col_name in factor_return_cols:
    cumulative_returns_exprs.append(
        ((pl.col(col_name) + 1).cum_prod() - 1).alias(col_name) # Keep original name for simplicity
    )

cumulative_returns_df = factor_returns.select(
    [pl.col('date')] + cumulative_returns_exprs
)

# Plot cumulative factor returns
plt.figure(figsize=(12, 6))
for col_name in factor_return_cols: # Iterate through original factor names
    plt.plot(cumulative_returns_df['date'], cumulative_returns_df[col_name], label=col_name)
plt.title('Cumulative Factor Returns (Geometric)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate factor return statistics (on the original non-cumulative returns)
stats_exprs = []
for col_name in factor_return_cols:
    stats_exprs.append(pl.col(col_name).mean().alias(f"{col_name}_mean"))
    stats_exprs.append(pl.col(col_name).std().alias(f"{col_name}_std"))
    stats_exprs.append(pl.col(col_name).skew().alias(f"{col_name}_skew"))
    stats_exprs.append(pl.col(col_name).kurtosis().alias(f"{col_name}_kurtosis"))

factor_stats = factor_returns.select(stats_exprs)

print("\nFactor Return Statistics (Wide Format):")
print(factor_stats)

# %% [markdown]
# ## Residual Analysis
# 
# Let's analyze the residual returns to check model fit:

# %%
# Calculate residual statistics for a sample of stocks
sample_stocks = symbols[:5]  # Look at first 5 stocks
residual_stats_exprs = []
for stock_symbol in sample_stocks:
    residual_stats_exprs.append(pl.col(stock_symbol).mean().alias(f"{stock_symbol}_mean"))
    residual_stats_exprs.append(pl.col(stock_symbol).std().alias(f"{stock_symbol}_std"))
    residual_stats_exprs.append(pl.col(stock_symbol).skew().alias(f"{stock_symbol}_skew"))
    residual_stats_exprs.append(pl.col(stock_symbol).kurtosis().alias(f"{stock_symbol}_kurtosis"))

residual_stats = residual_returns.select(residual_stats_exprs)

print("Residual Statistics for Sample Stocks (Wide Format):")
print(residual_stats)

# Plot residual return distribution for a sample stock
if sample_stocks and sample_stocks[0] in residual_returns.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(residual_returns[sample_stocks[0]], kde=True, stat="density", common_norm=False)
    plt.title(f'Residual Return Distribution for {sample_stocks[0]}')
    plt.xlabel('Residual Return')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()
else:
    print(f"Could not plot residual distribution: {sample_stocks[0] if sample_stocks else 'No sample stock'} not in residual_returns columns or no sample stocks.")

# %% [markdown]
# ## Additional Analysis Ideas
# 
# Here are some additional analyses you might want to try:
# 
# 1. Factor correlation analysis
# 2. Rolling factor return analysis
# 3. Factor exposure analysis
# 4. Portfolio construction using factor returns
# 5. Risk attribution analysis
# 
# You can add these analyses by creating new cells below.

if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # (not when imported as a module)
    print("Running equity factor model exploration...")
    # The script will execute all cells above when run directly 