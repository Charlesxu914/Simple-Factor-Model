"""
Equity Factor Model Runner

This script demonstrates how to:
1. Fetch equity data (S&P 500 or Russell 3000) using the Yahoo Finance data loader
2. Construct style factors
3. Run the factor model estimation
4. Analyze and visualize the results
"""

import argparse
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('factor_model_runner')

# Import our modules
from toraniko import styles, model
from toraniko.yfinance_data_loader import run_loader_and_model
from toraniko.ticker_universe import UNIVERSE_SP500, UNIVERSE_RUSSELL3000

def run_full_model(universe=UNIVERSE_SP500):
    """
    Run the complete factor model pipeline from data loading to visualization.

    Parameters
    ----------
    universe : str
        ``"sp500"`` or ``"russell3000"``.
    """
    logger.info(f"Starting the equity factor model pipeline (universe={universe})")

    # 1. Load data (or fetch if not available)
    logger.info(f"Loading {universe} data")
    returns_df, mkt_cap_df, sector_df, value_df = run_loader_and_model(universe=universe)
    
    if any(df is None for df in [returns_df, mkt_cap_df, sector_df, value_df]):
        logger.error("Failed to load required data")
        return
        
    # 2. Construct style factors
    logger.info("Constructing style factors")
    
    # Build each factor, collecting successes
    factor_dfs = {}  # name -> (df, score_col)

    try:
        # Determine available trading days to size the momentum window
        n_trading_days = returns_df['date'].n_unique()
        trailing_days = min(252, n_trading_days - 30)  # ~1 year lookback
        half_life = max(21, trailing_days // 4)
        lag = min(20, max(1, n_trading_days // 20))
        logger.info(f"Constructing momentum factor (trailing={trailing_days}, half_life={half_life}, lag={lag})")
        mom_df = styles.factor_mom(
            returns_df=returns_df,
            trailing_days=trailing_days,
            half_life=half_life,
            lag=lag,
            winsor_factor=0.01
        ).collect()
        factor_dfs['MOMENTUM'] = (mom_df, 'mom_score')
    except Exception as e:
        logger.warning(f"Failed to construct momentum factor: {e}")

    try:
        logger.info("Constructing size factor")
        sze_df = styles.factor_sze(
            mkt_cap_df=mkt_cap_df,
            lower_decile=0.2,
            upper_decile=0.8
        ).collect()
        factor_dfs['SIZE'] = (sze_df, 'sze_score')
    except Exception as e:
        logger.warning(f"Failed to construct size factor: {e}")

    try:
        logger.info("Constructing value factor")
        val_df = styles.factor_val(
            value_df=value_df,
            winsorize_features=0.01
        ).collect()
        factor_dfs['VALUE'] = (val_df, 'val_score')
    except Exception as e:
        logger.warning(f"Failed to construct value factor: {e}")

    if not factor_dfs:
        logger.error("All factor constructions failed. Cannot proceed.")
        return

    created_factors = list(factor_dfs.keys())
    logger.info(f"Successfully created factors: {', '.join(created_factors)}")

    # Combine all style factors by joining onto the base date-symbol pairs
    try:
        style_df = returns_df.select(['date', 'symbol']).unique()
        for name, (df, score_col) in factor_dfs.items():
            renamed = df.rename({score_col: name})
            style_df = style_df.join(renamed, on=['date', 'symbol'], how='left')

        # Fill missing factor scores (null and NaN) with 0
        for name in created_factors:
            style_df = style_df.with_columns(
                pl.col(name).fill_null(0).fill_nan(0)
            )
    except Exception as e:
        logger.error(f"Failed to combine style factors: {e}")
        return
    
    logger.info(f"Style factors combined. Shape: {style_df.shape}")
    
    # 3. Run the factor model estimation
    logger.info("Estimating factor returns with the model")
    factor_returns, residual_returns = model.estimate_factor_returns(
        returns_df=returns_df,
        mkt_cap_df=mkt_cap_df,
        sector_df=sector_df,
        style_df=style_df,
        winsor_factor=None,  # Factors already winsorized
        residualize_styles=True
    )
    
    # 4. Analyze results
    logger.info("Analyzing factor returns")
    
    # Calculate monthly factor returns
    factor_returns = factor_returns.with_columns([
        pl.col('date').dt.strftime("%Y-%m").alias('month')
    ])
    
    # Group by month and calculate average monthly returns
    monthly_factor_returns = factor_returns.group_by('month').agg([
        pl.exclude('date', 'month').mean()
    ]).sort('month')
    
    logger.info("Factor model estimation complete")
    
    # 5. Visualize results
    logger.info("Creating visualizations")
    
    # Calculate cumulative returns for each factor (geometric)
    factor_return_cols = [col for col in factor_returns.columns 
                         if col not in ['date', 'month']]
    
    # Get cumulative returns (including market)
    cumulative_returns_exprs = []
    for col_name in factor_return_cols:
        cumulative_returns_exprs.append(
            ((pl.col(col_name) + 1).cum_prod() - 1).alias(col_name)
        )
    
    cumulative_returns_df = factor_returns.select(
        [pl.col('date')] + cumulative_returns_exprs
    )
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 8))
    
    # Plot market return
    if 'market' in cumulative_returns_df.columns:
        plt.plot(cumulative_returns_df['date'], 
                cumulative_returns_df['market'], 
                label='Market', 
                linewidth=2)
    
    # Plot style factors
    style_factors = [col for col in factor_return_cols 
                    if col not in ['market']]
    
    # Identify sector columns by checking columns in sector_df
    sector_columns = [col for col in sector_df.columns 
                     if col not in ['date', 'symbol']]
    
    # Filter out any sector columns from style_factors
    style_factors = [factor for factor in style_factors 
                    if factor not in sector_columns]
    
    for factor in style_factors:
        plt.plot(cumulative_returns_df['date'], 
                cumulative_returns_df[factor], 
                label=factor,
                linewidth=2)
    
    plt.title('Cumulative Style Factor Returns', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/cumulative_style_returns.png')
    plt.close()
    
    # Calculate factor return statistics
    stats_exprs = []
    for col_name in factor_return_cols:
        stats_exprs.append(pl.col(col_name).mean().alias(f"{col_name}_mean"))
        stats_exprs.append(pl.col(col_name).std().alias(f"{col_name}_std"))
        stats_exprs.append((pl.col(col_name).mean() / pl.col(col_name).std()).alias(f"{col_name}_sharpe"))
    
    factor_stats = factor_returns.select(stats_exprs)
    
    # Save statistics to CSV
    factor_stats_pd = factor_stats.to_pandas()
    factor_stats_pd.to_csv('./plots/factor_statistics.csv')
    
    # Print summary statistics
    logger.info("Factor Model Summary Statistics:")
    logger.info("--------------------------------")
    
    # Create a summary table
    summary_data = []
    for factor in factor_return_cols:
        mean = factor_stats_pd[f"{factor}_mean"].values[0]
        std = factor_stats_pd[f"{factor}_std"].values[0]
        sharpe = factor_stats_pd[f"{factor}_sharpe"].values[0]
        
        summary_data.append({
            'Factor': factor,
            'Mean (%)': mean * 100,
            'Std Dev (%)': std * 100,
            'Sharpe': sharpe
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    logger.info("Analysis complete. Results saved to ./plots/ directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the equity factor model")
    parser.add_argument(
        "--universe", default=UNIVERSE_SP500,
        choices=[UNIVERSE_SP500, UNIVERSE_RUSSELL3000],
        help="Ticker universe to use (default: sp500)",
    )
    args = parser.parse_args()
    run_full_model(universe=args.universe)