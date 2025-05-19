"""
Equity Factor Model Runner

This script demonstrates how to:
1. Fetch S&P 500 data using the Yahoo Finance data loader
2. Construct style factors 
3. Run the factor model estimation
4. Analyze and visualize the results
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

def run_full_model():
    """
    Run the complete factor model pipeline from data loading to visualization.
    """
    logger.info("Starting the equity factor model pipeline")
    
    # 1. Load data (or fetch if not available)
    logger.info("Loading S&P 500 data")
    returns_df, mkt_cap_df, sector_df, value_df = run_loader_and_model()
    
    if any(df is None for df in [returns_df, mkt_cap_df, sector_df, value_df]):
        logger.error("Failed to load required data")
        return
        
    # 2. Construct style factors
    logger.info("Constructing style factors")
    
    try:
        # Momentum factor
        logger.info("Constructing momentum factor")
        mom_df = styles.factor_mom(
            returns_df=returns_df,
            trailing_days=504,  # ~2 years of trading days
            half_life=126,      # ~6 months decay
            lag=20,             # 1 month lag to avoid short-term reversal
            winsor_factor=0.01  # Winsorize at 1%
        ).collect()
    except Exception as e:
        logger.warning(f"Failed to construct momentum factor: {e}. Using empty DataFrame.")
        # Create empty momentum factor DataFrame with the right structure
        mom_df = pl.DataFrame({
            'date': returns_df['date'].unique(),
            'symbol': pl.Series(['PLACEHOLDER']),
            'mom_score': pl.Series([0.0])
        }).explode('symbol')
    
    try:
        # Size factor
        logger.info("Constructing size factor")
        sze_df = styles.factor_sze(
            mkt_cap_df=mkt_cap_df,
            lower_decile=0.2,   # Exclude smallest 20%
            upper_decile=0.8    # Exclude largest 20%
        ).collect()
    except Exception as e:
        logger.warning(f"Failed to construct size factor: {e}. Using empty DataFrame.")
        # Create empty size factor DataFrame with the right structure
        sze_df = pl.DataFrame({
            'date': returns_df['date'].unique(),
            'symbol': pl.Series(['PLACEHOLDER']),
            'sze_score': pl.Series([0.0])
        }).explode('symbol')
    
    try:
        # Value factor
        logger.info("Constructing value factor")
        val_df = styles.factor_val(
            value_df=value_df,
            winsorize_features=0.01  # Winsorize at 1%
        ).collect()
    except Exception as e:
        logger.warning(f"Failed to construct value factor: {e}. Using empty DataFrame.")
        # Create empty value factor DataFrame with the right structure
        val_df = pl.DataFrame({
            'date': returns_df['date'].unique(),
            'symbol': pl.Series(['PLACEHOLDER']),
            'val_score': pl.Series([0.0])
        }).explode('symbol')
    
    # Check if any factors were successfully created
    created_factors = []
    if not mom_df.filter(pl.col('symbol') != 'PLACEHOLDER').is_empty():
        created_factors.append('MOMENTUM')
    if not sze_df.filter(pl.col('symbol') != 'PLACEHOLDER').is_empty():
        created_factors.append('SIZE')
    if not val_df.filter(pl.col('symbol') != 'PLACEHOLDER').is_empty():
        created_factors.append('VALUE')
    
    if not created_factors:
        logger.error("All factor constructions failed. Cannot proceed.")
        return
    
    logger.info(f"Successfully created the following factors: {', '.join(created_factors)}")
    
    try:
        # Combine all style factors using joins with appropriate handling of potential mismatches
        style_df_parts = []
        
        # Start with a base DataFrame derived from symbol-date combinations in the returns DataFrame
        base_df = returns_df.select(['date', 'symbol']).unique()
        
        # Momentum factor
        if 'MOMENTUM' in created_factors:
            mom_subset = mom_df.filter(pl.col('symbol') != 'PLACEHOLDER')
            style_df_parts.append(mom_subset.rename({'mom_score': 'MOMENTUM'}))
            
        # Size factor
        if 'SIZE' in created_factors:
            sze_subset = sze_df.filter(pl.col('symbol') != 'PLACEHOLDER')
            style_df_parts.append(sze_subset.rename({'sze_score': 'SIZE'}))
            
        # Value factor
        if 'VALUE' in created_factors:
            val_subset = val_df.filter(pl.col('symbol') != 'PLACEHOLDER')
            style_df_parts.append(val_subset.rename({'val_score': 'VALUE'}))
        
        # Combine the factors
        style_df = base_df
        for factor_df in style_df_parts:
            style_df = style_df.join(factor_df, on=['date', 'symbol'], how='left')
        
        # Fill any missing values
        for factor in created_factors:
            style_df = style_df.with_columns([
                pl.col(factor).fill_null(0).alias(factor)
            ])
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
    run_full_model() 